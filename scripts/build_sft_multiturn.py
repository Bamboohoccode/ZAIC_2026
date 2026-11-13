"""Dựng bộ data hội thoại nhiều lượt có tool result từ train BTC (không đọc public_test).

Đầu ra (data/processed/): sft_multiturn_{train,dev}_<ngày>.jsonl
Báo cáo (reports/): sft_multiturn_<ngày>.md ; log (logs/): build_sft_multiturn_<ngày>.log
Chạy: python -X utf8 scripts/build_sft_multiturn.py [--tokenizer <thư mục model>] [--date 20261111]
"""
import argparse, collections, copy, hashlib, json, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/train+public_test/train"
sys.path.insert(0, str(RAW))  # zaic_tools của train
from zaic_tools import Executor, canon  # noqa: E402

SYSTEM_TMPL = (
    "Bạn là LifeMate, trợ lý giọng nói tiếng Việt trên điện thoại. Dùng các tool được cung cấp để thực hiện yêu cầu của người dùng. "
    "Gọi tool đọc trước khi cần id hoặc dữ liệu; thiếu tham số bắt buộc người dùng chưa nói thì gọi sys.ask_missing_info; "
    "yêu cầu ngoài các app đã cài thì không gọi tool.\n"
    "Thời điểm hiện tại: {current_time} ({timezone}).\n"
    "Vị trí: lat {lat}, lng {lng}.\n"
    "Người dùng: {profile}.\n"
    "App đã cài: {apps}."
)


def norm_text(s):
    return re.sub(r"[^\w]+", " ", (s or "").casefold()).strip()


def tool_schemas(catalog, installed):
    allowed = {t for a in installed.values() for t in a.get("tools", [])} | {"sys.ask_missing_info"}
    out = []
    for app in catalog["apps"]:
        for t in app["tools"]:
            if t["name"] in allowed:
                out.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]}})
    return out, allowed


def call_id(sid, tag, i):
    return f"call_{sid}_{tag}{i}"


def build_messages(row):
    ctx = row["context"]
    prof = ctx.get("user_profile", {})
    sysmsg = SYSTEM_TMPL.format(
        current_time=ctx["current_time"], timezone=ctx.get("timezone", ""),
        lat=ctx.get("location", {}).get("lat"), lng=ctx.get("location", {}).get("lng"),
        profile=json.dumps(prof, ensure_ascii=False), apps=", ".join(ctx["installed_apps"]))
    msgs = [{"role": "system", "content": sysmsg, "train": False}]
    sid = row["id"]
    for h_i, h in enumerate(row["history"]):
        if h["role"] == "user":
            msgs.append({"role": "user", "content": h["audio_text"], "train": False})
            continue
        calls, results = h.get("tool_calls") or [], h.get("tool_results") or []
        for i, (c, r) in enumerate(zip(calls, results)):
            cid = call_id(sid, f"h{h_i}_", i)
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}], "train": False})
            msgs.append({"role": "tool", "tool_call_id": cid, "name": c["name"], "content": json.dumps(r, ensure_ascii=False), "train": False})
        if h.get("text"):
            msgs.append({"role": "assistant", "content": h["text"], "train": False})
    msgs.append({"role": "user", "content": row["audio_text"], "train": False})
    for i, (c, r) in enumerate(zip(row["label"]["tool_calls"], row["label"]["tool_results"])):
        cid = call_id(sid, "", i)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}], "train": True})
        msgs.append({"role": "tool", "tool_call_id": cid, "name": c["name"], "content": json.dumps(r, ensure_ascii=False), "train": False})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261111")
    ap.add_argument("--tokenizer", default=str(ROOT.parent / "models/Qwen3.8-4B-Distilled"))
    a = ap.parse_args()
    t0 = time.time()
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True); (ROOT / "logs").mkdir(exist_ok=True)
    log = open(ROOT / f"logs/build_sft_multiturn_{a.date}.log", "w", encoding="utf-8")
    def P(*x):
        s = " ".join(str(i) for i in x); print(s); log.write(s + "\n")

    rows = [json.loads(l) for l in open(RAW / "train.jsonl", encoding="utf-8")]
    P("train.jsonl:", len(rows), "dòng (kỳ vọng 1545)")
    assert len(rows) == 1545, "số dòng lệch, dừng theo CLAUDE.md 3b"
    catalog = json.load(open(RAW / "train_tools_catalog.json", encoding="utf-8"))
    dev_ids = set(open(ROOT / "eval/splits/dev_ids.txt", encoding="utf-8").read().split())
    P("dev_ids:", len(dev_ids))
    ex = Executor(str(RAW / "train_environment"), str(RAW / "train_tools_catalog.json"))

    dev_texts = {norm_text(r["audio_text"]) for r in rows if r["id"] in dev_ids}
    dropped, dropped_within, seen_train, within_dup = [], [], collections.Counter(), 0
    keep = []
    for r in rows:
        n = norm_text(r["audio_text"])
        if r["id"] not in dev_ids and n in dev_texts:
            dropped.append(r["id"]); continue  # chống rò dev sang train
        if r["id"] not in dev_ids:
            seen_train[n] += 1
            if seen_train[n] > 1:
                within_dup += 1; dropped_within.append(r["id"]); continue  # giữ mẫu đầu, như B2
        keep.append(r)
    P("loại vì trùng audio_text với dev:", dropped)
    P("loại vì trùng audio_text trong nội bộ train (giữ mẫu đầu):", dropped_within)

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, fix_mistral_regex=True)
        P("tokenizer:", a.tokenizer)
    except Exception as e:
        P("KHÔNG nạp được tokenizer, bỏ đo token:", repr(e)[:200])

    out = {"train": [], "dev": []}
    mism_res, mism_state, bad_schema, not_allowed = [], [], [], []
    stats = collections.Counter()
    lens = []
    for r in keep:
        sid = r["id"]; lab = r["label"]; calls = lab["tool_calls"]
        tools, allowed = tool_schemas(catalog, r["context"]["installed_apps"])
        # a) mọi call phải nằm trong tool được cài
        if any(c["name"] not in allowed for c in calls):
            not_allowed.append(sid)
        # b) replay trên state.json: kết quả và final_state
        s = ex.open(sid)
        res = [s.call(c["name"], c["arguments"]) for c in calls]
        if any(not x.get("ok", True) for x in res):
            bad_schema.append(sid)
        if res != lab["tool_results"]:
            mism_res.append(sid)
        st = s.get_state()
        before = ex.open(sid).get_state()
        changed = {k: v for k, v in st.items() if v != before.get(k)}
        if canon(changed) != canon(lab["final_state"]):
            mism_state.append(sid)
        cat = "ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else ("empty" if not calls else "act")
        msgs = build_messages(r)
        rec = {"id": sid, "split": "dev" if sid in dev_ids else "train", "source": "btc", "category": cat,
               "apps": sorted(r["context"]["installed_apps"]), "n_calls": len(calls), "n_history": len(r["history"]),
               "messages": msgs, "tools": tools}
        if tok is not None:
            try:
                clean = [{k: v for k, v in m.items() if k != "train"} for m in msgs]
                n = len(tok.apply_chat_template(clean, tools=tools, tokenize=True, add_generation_prompt=False, return_dict=False))
                rec["n_tokens"] = n; lens.append(n)
            except Exception as e:
                rec["n_tokens"] = None; stats["template_error"] += 1
                if stats["template_error"] <= 3:
                    P("lỗi chat template", sid, repr(e)[:200])
        out[rec["split"]].append(rec)
        stats[cat] += 1
    for k, v in out.items():
        path = ROOT / f"data/processed/sft_multiturn_{k}_{a.date}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in v:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        P(f"ghi {path.name}: {len(v)} dòng, sha256 {h}")
        with open(ROOT / f"data/manifest/sft_multiturn_sha256_{a.date}.txt", "a", encoding="utf-8") as mf:
            mf.write(f"{h}  {path.name}  rows={len(v)}\n")

    # báo cáo
    allrec = out["train"] + out["dev"]
    L = [f"# Bộ data nhiều lượt có tool result (derived, sinh {a.date})", "",
         "Tự sinh bởi `scripts/build_sft_multiturn.py`. Nguồn: train BTC. Không đọc public_test.", "",
         "## 1. Số đếm", "",
         f"- Nguồn: {len(rows)} mẫu. Loại vì trùng audio_text với dev: {len(dropped)} {dropped}.",
         f"- train: {len(out['train'])}, dev: {len(out['dev'])}, tổng {len(allrec)}.",
         f"- Loại vì trùng audio_text trong nội bộ train (giữ mẫu đầu): {len(dropped_within)} {dropped_within}."]
    for sp in ("train", "dev"):
        c = collections.Counter(r["category"] for r in out[sp])
        L.append(f"- {sp} theo category: act {c['act']}, ask {c['ask']}, empty {c['empty']}.")
    L += ["", "## 2. Kiểm ToolEnv (replay gold trên state.json)", "",
          f"- Call ngoài tool được cài: {len(not_allowed)} {not_allowed[:10]}",
          f"- Mẫu có call bị từ chối schema (ok=false): {len(bad_schema)} {bad_schema[:10]}",
          f"- Mẫu có tool_results lệch nhãn BTC: {len(mism_res)} {mism_res[:20]}",
          f"- Mẫu có final_state lệch nhãn BTC: {len(mism_state)} {mism_state[:20]}",
          "- Không sửa nhãn; mẫu lệch vẫn được xuất, báo để người dùng quyết định."]
    if lens:
        qs = sorted(lens)
        L += ["", "## 3. Độ dài token (chat template của tokenizer " + Path(a.tokenizer).name + ")", "",
              f"- trung vị {int(statistics.median(lens))}, p90 {qs[int(.9*len(qs))]}, p99 {qs[int(.99*len(qs))]}, max {qs[-1]}.",
              f"- mẫu > 4096: {sum(x>4096 for x in lens)}; > 8192: {sum(x>8192 for x in lens)}; > 16384: {sum(x>16384 for x in lens)}.",
              f"- Lỗi chat template: {stats['template_error']}.",
              "- Không cắt tool_results. Cột `n_tokens` trong file để lọc theo ngưỡng."]
    L += ["", "## 4. Phân bố", "",
          "- Số call mỗi mẫu: " + str(dict(sorted(collections.Counter(r['n_calls'] for r in allrec).items()))),
          "- Số lượt history: " + str(dict(sorted(collections.Counter(r['n_history'] for r in allrec).items()))),
          "", "## 5. Chưa xác minh", "",
          "- Nhãn BTC không có câu trả lời bằng lời của assistant sau tool result cuối ở lượt cuối; data kết thúc ở call cuối.",
          "- `arguments` lưu dạng dict (không phải chuỗi JSON). Cần đổi nếu trainer yêu cầu chuỗi.",
          "- `train: true/false` trong messages là mask loss gợi ý, tuỳ trainer có dùng không."]
    (ROOT / f"reports/sft_multiturn_{a.date}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    P("xong trong", round(time.time() - t0, 1), "giây")
    log.close()


if __name__ == "__main__":
    main()
