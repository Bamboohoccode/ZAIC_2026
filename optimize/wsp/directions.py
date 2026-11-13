"""Tìm hướng gây lỗi trong residual stream của Qwen3-4B (chạy trên GPU, ví dụ Kaggle).

Ý tưởng như blog (Thí nghiệm 8): so trạng thái ẩn khi model sinh lượt lỗi (thinking bị cắt, lặp) với khi sinh lượt đúng.
Với mỗi mẫu trong `--export` (đầu ra analyze.py, nhãn bad/good):
  1. dựng lại prompt lượt đầu đúng như pipeline (build_messages + tool_specs + chat template),
  2. nối lượt sinh đầu tiên đã ghi trong trace, chạy một lượt forward (teacher forcing),
  3. lấy trung bình trạng thái ẩn trên các token của lượt sinh, ở mọi tầng.
Hướng của tầng l = mean(bad) - mean(good) tại hidden_states[l + 1] (đầu ra tầng l, chỗ down_proj tầng l ghi vào), chuẩn hoá.
Độ tách: Cohen's d của hình chiếu lên hướng đó. Tầng có d lớn là ứng viên để sửa.

Đầu ra: `--out` (torch.save {layer: vector float32}), `--stats` (JSON, tầng xếp theo d).
Chạy: python optimize/wsp/directions.py --model /kaggle/working/Qwen3-4B --export runs/base.export.jsonl \
        --out runs/dirs.pt --stats runs/dirs.stats.json [--think off]
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def build_prompt(tok, item, catalog, think):
    from lifemate.pipeline import build_messages, tool_specs
    texts = [h["audio_text"] for h in item["history"] if h["role"] == "user"] + [item["audio_text"]]
    msgs = build_messages(item, texts, len(texts) - 1)
    kw = {} if think == "on" else {"enable_thinking": False}
    return tok.apply_chat_template(msgs, tools=tool_specs(item, catalog), add_generation_prompt=True, tokenize=False, **kw)


def hidden_mean(model, p_ids, o_ids, device="cuda", chunk=1024):
    """Trung bình trạng thái ẩn (mọi tầng) trên các token của lượt sinh o_ids, sau prompt p_ids.
    Chạy theo đoạn có KV cache để prompt dài (5-9 nghìn token) không làm tràn bộ nhớ attention trên T4.
    Kết quả giống một lượt forward đầy đủ (causal), chỉ khác nhiễu số học."""
    import torch
    past = None
    with torch.no_grad():
        for st in range(0, len(p_ids), chunk):
            x = torch.tensor([p_ids[st:st + chunk]], device=device)
            out = model(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            del out
        acc, n = None, 0
        for st in range(0, len(o_ids), chunk):
            x = torch.tensor([o_ids[st:st + chunk]], device=device)
            out = model(input_ids=x, past_key_values=past, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            past = out.past_key_values
            hs = torch.stack([h[0].float().sum(0) for h in out.hidden_states])   # [L+1, H]
            acc = hs if acc is None else acc + hs
            n += x.shape[1]
            del out, hs
    del past
    return (acc / n).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--root", default=str(REPO / "data" / "raw" / "train+public_test"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--max-tokens", type=int, default=2048, help="số token tối đa của lượt sinh dùng để lấy trung bình (prompt không bị cắt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--prompt-only", action="store_true", help="chỉ dựng prompt và đếm token, không nạp model (kiểm tra trên CPU)")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none", help="bitsandbytes, để vừa GPU nhỏ")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    root = Path(a.root)
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8")}
    recs = [json.loads(l) for l in open(a.export, encoding="utf-8")]
    n_bad = sum(r["label"] == "bad" for r in recs)
    print(f"records: {len(recs)} (bad {n_bad}, good {len(recs) - n_bad})")
    if n_bad < 5 and not a.prompt_only:
        print("CẢNH BÁO: dưới 5 mẫu bad, hướng tìm được không đáng tin")

    if a.prompt_only:
        lens = []
        for r in recs:
            p = build_prompt(tok, items[r["id"]], catalog, a.think)
            lens.append((len(tok(p, add_special_tokens=False).input_ids), len(tok(r["first_plan"], add_special_tokens=False).input_ids)))
        print("prompt tokens min/max:", min(x for x, _ in lens), max(x for x, _ in lens),
              "| output tokens max:", max(y for _, y in lens))
        return

    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    q = None
    if a.quant != "none":
        from transformers import BitsAndBytesConfig
        q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
             if a.quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, quantization_config=q, device_map="cuda:0",
                                                 low_cpu_mem_usage=True).eval()
    feats = {"bad": [], "good": []}
    n_nonfinite = 0
    n_oom = 0
    for r in recs:
        p_ids = tok(build_prompt(tok, items[r["id"]], catalog, a.think), add_special_tokens=False).input_ids
        o_ids = tok(r["first_plan"], add_special_tokens=False).input_ids[: a.max_tokens]
        try:
            f = hidden_mean(model, p_ids, o_ids)   # [L+1, H]
        except torch.OutOfMemoryError:
            print(f"  OOM ở {r['id']} (prompt {len(p_ids)} token): bỏ mẫu", flush=True)
            n_oom += 1
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        if not torch.isfinite(f).all():  # fp16 (T4) có thể tràn ở activation: bỏ mẫu, không làm hỏng hướng
            n_nonfinite += 1
            continue
        feats[r["label"]].append(f)
    print(f"bỏ {n_nonfinite} mẫu có NaN/inf trong trạng thái ẩn (dtype {dtype}), {n_oom} mẫu OOM")
    if len(feats["bad"]) < 2 or len(feats["good"]) < 2:
        raise SystemExit(f"không đủ mẫu hợp lệ: bad {len(feats['bad'])}, good {len(feats['good'])}")
    B, G = torch.stack(feats["bad"]), torch.stack(feats["good"])   # [nb, L+1, H], [ng, L+1, H]
    L = B.shape[1] - 1
    dirs, stats = {}, []
    for l in range(L):
        diff = B[:, l + 1].mean(0) - G[:, l + 1].mean(0)
        u = diff / diff.norm()
        pb, pg = B[:, l + 1] @ u, G[:, l + 1] @ u
        pooled = ((pb.var(unbiased=True) + pg.var(unbiased=True)) / 2).sqrt().clamp_min(1e-6)
        dirs[l] = u
        stats.append({"layer": l, "cohen_d": round(float((pb.mean() - pg.mean()) / pooled), 3), "diff_norm": round(float(diff.norm()), 3)})
    torch.save(dirs, a.out)
    stats.sort(key=lambda s: -s["cohen_d"])
    json.dump({"n_bad": len(feats["bad"]), "n_good": len(feats["good"]), "n_nonfinite_skipped": n_nonfinite, "n_oom_skipped": n_oom,
               "dtype": str(dtype), "think": a.think, "layers": stats},
              open(a.stats, "w", encoding="utf-8"), indent=1)
    print("top layers:", stats[:8])


if __name__ == "__main__":
    main()
