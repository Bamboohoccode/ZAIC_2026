"""Một vòng WSP = một "forward pass" theo cách đếm của blog:
áp bản sửa (spec JSON) -> lưu model -> chạy agent trên tập id -> chấm bằng analyze.py -> ghi log.

Vòng 0 (`--spec none`) là baseline, dùng model gốc. Mỗi vòng ghi một dòng vào `--log` (JSONL) và in so sánh với vòng 0
cùng cấu hình. Chỉ chạy trên train/dev. Không chạy trên test BTC (CLAUDE.md mục 4).

Bộ chạy (`--runner`):
a) batch (mặc định): optimize/wsp/batch_agent.py, mọi mẫu chạy song song; `--gpus 0,1` chia id cho từng GPU (mỗi GPU một
   tiến trình vLLM). Chi phí đo bằng gen_tokens (số token sinh TB mỗi mẫu); avg_seconds không phải độ trễ một mẫu.
b) seq: lifemate/pipeline.py gốc, từng mẫu một (dùng để đo độ trễ thật).
`--llm-gold` chạy được với cả hai runner (replay nhãn, không GPU), dùng để kiểm luồng.
Chỉ so các vòng cùng runner.

Kaggle:  python optimize/wsp/run_round.py --base-model /tmp/Qwen3-4B --round 3 --spec r03.json --dirs dirs.pt \
           --work /kaggle/working/wsp --model-out /tmp/wsp_model --llm-dtype half --gpus 0,1
Máy cá nhân (kiểm tra luồng, không cần GPU): thêm --llm-gold (pipeline replay nhãn thay vì chạy model).
"""
import argparse, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from weights import ModelWeights, apply_spec  # noqa: E402


def run_batch(a, model_dir, out):
    """Chia id cho các GPU, mỗi GPU một tiến trình batch_agent, rồi gộp submission và trace theo thứ tự id."""
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    ids = [l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()]
    shards = [ids[k::len(gpus)] for k in range(len(gpus))]
    procs, parts = [], []
    for k, (g, sh) in enumerate(zip(gpus, shards)):
        if not sh:
            continue
        idf, part = out.parent / f"{out.stem}.ids{k}.txt", out.parent / f"{out.stem}.part{k}.jsonl"
        idf.write_text("\n".join(sh) + "\n", encoding="utf-8")
        cmd = [sys.executable, str(HERE / "batch_agent.py"), "--llm", "gold" if a.llm_gold else str(model_dir),
               "--root", a.root, "--split", "train",
               "--ids", str(idf), "--out", str(part), "--llm-think", a.think, "--llm-dtype", a.llm_dtype,
               "--gpu-mem", a.gpu_mem, "--verify-rounds", a.verify_rounds, "--seed", "42"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        print(f"RUN gpu{g} ({len(sh)} mẫu)", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=env))
        parts.append(part)
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"batch_agent lỗi, mã thoát {codes}")
    sub, tr = {}, {}
    for part in parts:
        for l in open(part, encoding="utf-8"):
            sub[json.loads(l)["id"]] = l
        for l in open(str(part) + ".trace.jsonl", encoding="utf-8"):
            tr[json.loads(l)["id"]] = l
    with open(out, "w", encoding="utf-8") as f1, open(str(out) + ".trace.jsonl", "w", encoding="utf-8") as f2:
        for i in ids:
            if i in sub:
                f1.write(sub[i]); f2.write(tr[i])
    for part in parts:  # tệp tạm của từng GPU
        for p in (part, Path(str(part) + ".trace.jsonl"), out.parent / part.name.replace(".part", ".ids").replace(".jsonl", ".txt")):
            p.unlink(missing_ok=True)


def cost_stats(trace_path):
    rows = [json.loads(l) for l in open(trace_path, encoding="utf-8")]
    n = max(len(rows), 1)
    if not rows or "gen_tokens" not in rows[0]:
        return None, None
    return round(sum(r["gen_tokens"] for r in rows) / n, 1), round(sum(r["llm_calls"] for r in rows) / n, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--spec", default="none", help='file JSON danh sách phép sửa, hoặc "none" (baseline)')
    ap.add_argument("--dirs", help="đầu ra directions.py (cần cho project_out)")
    ap.add_argument("--ids", default=str(HERE / "dev_subset50_20261107.txt"))
    ap.add_argument("--root", default=str(REPO / "data" / "raw" / "train+public_test"))
    ap.add_argument("--work", default=str(REPO / "runs" / "wsp"))
    ap.add_argument("--model-out", help="nơi ghi model đã sửa (mặc định <work>/wsp_model); trên Kaggle nên để /tmp")
    ap.add_argument("--log", help="mặc định <work>/wsp_log.jsonl")
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--llm-dtype", default="auto")
    ap.add_argument("--gpu-mem", default="0.9")
    ap.add_argument("--runner", choices=["batch", "seq"], default="batch")
    ap.add_argument("--gpus", default="0", help='runner batch: danh sách GPU, ví dụ "0,1"')
    ap.add_argument("--llm-gold", action="store_true", help="kiểm tra luồng: pipeline replay nhãn, không nạp model")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help='runner seq: "hf" = transformers')
    ap.add_argument("--hf-quant", choices=["none", "8bit", "4bit"], default="4bit")
    ap.add_argument("--verify-rounds", default="2", help="số vòng tự kiểm của pipeline; giữ cố định giữa các vòng WSP")
    ap.add_argument("--keep-model", action="store_true", help="không xoá model đã sửa sau khi chạy")
    ap.add_argument("--label-rule", choices=["cutloop", "score"], default="cutloop", help="luật nhãn của analyze.py")
    a = ap.parse_args()

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    log = Path(a.log) if a.log else work / "wsp_log.jsonl"
    tag = f"r{a.round:02d}"
    spec = [] if a.spec == "none" else json.load(open(a.spec, encoding="utf-8"))
    edit_stats = {"n_changed": 0, "fraction_changed": 0.0, "edited_tensors": []}
    model_dir = Path(a.base_model)
    t0 = time.time()
    if spec:
        import torch
        dirs = torch.load(a.dirs) if a.dirs else None
        mw = ModelWeights(a.base_model)
        apply_spec(mw, spec, dirs)
        model_dir = Path(a.model_out) if a.model_out else work / "wsp_model"
        if model_dir.exists():
            shutil.rmtree(model_dir)              # chỉ giữ model của vòng hiện tại
        edit_stats = mw.save(model_dir)
        shutil.copy2(model_dir / "wsp_edits.json", work / f"{tag}.wsp_edits.json")
    t_edit = time.time() - t0

    out = work / f"{tag}.jsonl"
    t1 = time.time()
    if a.runner == "batch":
        run_batch(a, model_dir, out)
    else:
        cmd = [sys.executable, "-m", "lifemate.pipeline", "--root", a.root, "--split", "train", "--text", "--mapper", "none",
               "--ids", a.ids, "--out", str(out), "--llm", "gold" if a.llm_gold else str(model_dir),
               "--llm-think", a.think, "--llm-dtype", a.llm_dtype, "--gpu-mem", a.gpu_mem, "--seed", "42",
               "--backend", a.backend, "--hf-quant", a.hf_quant, "--verify-rounds", a.verify_rounds]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    t_run = time.time() - t1
    if spec and not a.keep_model:
        shutil.rmtree(model_dir, ignore_errors=True)
    summary_path = work / f"{tag}.summary.json"
    export_path = work / f"{tag}.export.jsonl"
    subprocess.run([sys.executable, str(HERE / "analyze.py"), "--root", a.root, "--sub", str(out), "--ids", a.ids,
                    "--out", str(summary_path), "--export", str(export_path), "--label-rule", a.label_rule], cwd=REPO, check=True)
    s = json.load(open(summary_path, encoding="utf-8"))
    gen_tokens, llm_calls = cost_stats(str(out) + ".trace.jsonl")

    row = {"round": a.round, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spec": spec,
           "model_dir": str(model_dir), "think": a.think, "llm_gold": a.llm_gold, "runner": a.runner, "ids": Path(a.ids).name,
           "backend": a.backend, "hf_quant": a.hf_quant if a.backend == "hf" else None, "verify_rounds": a.verify_rounds,
           "n_changed": edit_stats["n_changed"], "fraction_changed": edit_stats["fraction_changed"],
           "edited_tensors": edit_stats["edited_tensors"], "edit_seconds": round(t_edit, 1), "run_seconds": round(t_run, 1),
           "n": s["n"], "mean_score": s["mean_score"], "by_kind": s["by_kind"],
           "flags": {k: v["count"] for k, v in s["flags"].items()}, "gen_tokens_mean": gen_tokens, "llm_calls_mean": llm_calls,
           "avg_seconds": s["avg_seconds"], "max_seconds": s["max_seconds"]}
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def same_setup(r):  # chỉ so với vòng 0 cùng cấu hình chạy
        return (r["round"] == 0 and r["think"] == a.think and r["llm_gold"] == a.llm_gold and r.get("runner", "seq") == a.runner
                and r.get("ids", row["ids"]) == row["ids"] and r.get("backend", "vllm") == a.backend
                and r.get("hf_quant") == row["hf_quant"] and r.get("verify_rounds", "2") == a.verify_rounds)
    base = next((json.loads(l) for l in open(log, encoding="utf-8") if same_setup(json.loads(l))), None)
    print(f"\n[{tag}] mean={row['mean_score']} changed={row['n_changed']} ({row['fraction_changed']:.6%}) "
          f"gen_tokens={gen_tokens} run_s={row['run_seconds']}")
    if base and a.round != 0:
        print(f"      so với vòng 0: mean {base['mean_score']} -> {row['mean_score']} | "
              f"gen_tokens {base.get('gen_tokens_mean')} -> {gen_tokens}")
        for k in sorted(set(base["flags"]) | set(row["flags"])):
            print(f"      {k:22s} {base['flags'].get(k, 0)} -> {row['flags'].get(k, 0)}")


if __name__ == "__main__":
    main()
