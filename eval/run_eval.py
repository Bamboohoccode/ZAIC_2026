"""Báo cáo eval: điểm theo nhóm và app, lỗi, thống kê từng tầng (từ file trace của pipeline), pass^k.

python -m eval.run_eval --root data --split train --subs runs/a.jsonl [runs/b.jsonl ...] [--ids eval/splits/dev_ids.txt] [--json out.json]
Nhiều --subs (cùng pipeline, chạy lại k lần): pass^k = tỉ lệ mẫu đạt điểm 1.0 ở cả k lần.
--ids cố định tập cần chấm; ID thiếu trong submission tính 0. Nếu bỏ --ids, mỗi file được chấm trên ID hiện có.
Cần <sub>.trace.jsonl (pipeline.py ghi cạnh submission) để có thống kê tầng; thiếu file thì bỏ qua phần đó.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, để chạy trực tiếp như tests/test_pipeline.py

from eval.scorer import evaluate_submission
from lifemate.pipeline import TIME_LIMIT_S, parse_calls, parse_verdict


def pass_k(reports, target_ids=None):
    """Tỉ lệ ID đạt điểm 1.0 ở mọi lần chạy; ID thiếu được tính là 0."""
    domain = set(target_ids) if target_ids is not None else set().union(*(set(r["scores"]) for r in reports))
    score = sum(all(r["scores"].get(i, 0.0) >= 1.0 for r in reports) for i in domain) / max(len(domain), 1)
    return score, domain


def layer_stats(trace_path, ids=None):
    """Tỉ lệ lỗi theo tầng: mapper, planner, verifier, executor/ngoại lệ, thời gian."""
    rows = [json.loads(l) for l in open(trace_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not ids or r["id"] in ids]
    steps = [s for r in rows for s in r["steps"]]
    plans = [s["plan"] for s in steps if "plan" in s]
    verdicts = [parse_verdict(s["verify"])[0] for s in steps if "verify" in s]
    n = max(len(rows), 1)
    ratio = lambda a, b: round(a / b, 3) if b else None
    return {
        "mapper_sửa_text": ratio(sum(r["mapped_texts"] != r["raw_texts"] for r in rows), n),
        "planner_không_call": ratio(sum(not parse_calls(p) for p in plans), len(plans)),
        "verify_reject": ratio(verdicts.count(False), len(verdicts)),
        "mẫu_có_exception_hoặc_quá_giờ": ratio(sum(any("error" in s for s in r["steps"]) for r in rows), n),
        f"mẫu_quá_{TIME_LIMIT_S}s": ratio(sum(r["seconds"] > TIME_LIMIT_S for r in rows), n),
        "giây_trung_bình": round(sum(r["seconds"] for r in rows) / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--subs", nargs="+", required=True, help="submission.jsonl, nhiều file = nhiều lần chạy cùng pipeline")
    ap.add_argument("--ids", help="tập ID mục tiêu; ID thiếu trong submission được tính 0")
    ap.add_argument("--min-app", type=int, default=5, help="chỉ in app có ít nhất ngần này mẫu")
    ap.add_argument("--json", help="ghi toàn bộ báo cáo ra file")
    a = ap.parse_args()
    ids = set(open(a.ids, encoding="utf-8").read().split()) if a.ids else None

    reports = []
    for sub in a.subs:
        r = evaluate_submission(a.root, a.split, sub, filter_ids=ids, verbose=False)
        r["file"] = sub
        if Path(sub + ".trace.jsonl").exists():
            r["layers"] = layer_stats(sub + ".trace.jsonl", ids)
        reports.append(r)
        kinds = " | ".join(f"{k} {v['score']:.4f} (n={v['n']})" for k, v in r["by_kind"].items())
        print(f"\n== {sub}: N={r['total_samples']} mean {r['mean_score']:.4f} | {kinds}")
        print("lỗi:", ", ".join(f"{e} {c}" for e, c in Counter(d["error"] for d in r["diagnostics"]).most_common()))
        worst = sorted(((v["score"], k, v["n"]) for k, v in r["by_app"].items() if v["n"] >= a.min_app))[:8]
        print("app kém nhất:", ", ".join(f"{k} {s:.2f} (n={n})" for s, k, n in worst))
        if "layers" in r:
            print("tầng:", json.dumps(r["layers"], ensure_ascii=False))

    if len(reports) > 1:
        # Một ID thiếu ở bất kỳ lần chạy nào được tính là fail; không lấy intersection vì sẽ bỏ mất lỗi coverage.
        passk, domain = pass_k(reports, ids)
        print(f"\npass^{len(reports)} = {passk:.4f} (trên {len(domain)} ID mục tiêu; ID thiếu tính là 0)")
        for r in reports:
            r["pass_k"] = passk
    if a.json:
        json.dump(reports, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
