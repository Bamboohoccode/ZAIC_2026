"""Trần điểm của Verifier: một Verifier đoán đúng 100% (dùng gold) giúp được bao nhiêu điểm?

Từ một file dự đoán của Planner, dựng các bản nộp "Verifier hoàn hảo", replay qua ToolEnv để lấy state_hash,
rồi chấm bằng eval/scorer.py (cùng công thức BTC):
  goc        dự đoán gốc
  hoi_user   gặp call sai đầu tiên thì thay bằng sys.ask_missing_info(tool, tham số của call đó) và dừng
             (thiết kế trong PROJECT_OVERVIEW mục 4c)
  bo_call    bỏ mọi call sai, giữ call khớp gold
  chon_loc   mỗi mẫu lấy điểm cao hơn giữa goc và bo_call (Verifier biết khi nào bỏ có lợi)
Đây là giới hạn trên. Verifier thật luôn thấp hơn.

  python lifemate/verifier/oracle_ceiling.py --root D:/Downloads/ZAIC/train+public_test --split train \
      --pred dev_v4.jsonl --ids eval/splits/dev_ids.txt --out runs/oracle_v4.json

Kết quả 07/11 trên dev_v4 (300 mẫu): goc 0,3597 | hoi_user 0,3063 | bo_call 0,3760 | chon_loc 0,4218.
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ASK_TOOL = "sys.ask_missing_info"


def oracle_calls(pred, gold, catalog, matches):
    """(bo_call, hoi_user): hai chuỗi call của Verifier hoàn hảo cho một mẫu."""
    left, keep, ask, asked = list(gold), [], [], False
    for c in pred:
        j = next((j for j, g in enumerate(left) if matches(c, g, catalog)), None)
        if j is not None:
            left.pop(j)
            keep.append(c)
            if not asked:
                ask.append(c)
        elif not asked:
            ask.append({"name": ASK_TOOL, "arguments": {"tool_name": c["name"], "params": sorted((c.get("arguments") or {}))}})
            asked = True
    return keep, ask


def kind_of(gold):
    return "empty" if not gold else "ask" if any(g["name"] == ASK_TOOL for g in gold) else "act"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ids", help="chỉ chấm các id này, vd eval/splits/dev_ids.txt")
    ap.add_argument("--out", help="ghi kết quả JSON")
    a = ap.parse_args()

    root = Path(a.root)
    sys.path.insert(0, str(root / a.split))
    from zaic_tools import ToolEnv
    from eval.scorer import call_matches, evaluate_submission
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()}
    preds = [json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()]
    ids = set(Path(a.ids).read_text().split()) if a.ids else {p["id"] for p in preds}
    preds = [p for p in preds if p["id"] in ids and p["id"] in items]

    def replay(sid, calls):
        env = ToolEnv(a.split, sid, root=str(root))
        for c in calls:
            env.call(c["name"], c.get("arguments") or {})
        return {"id": sid, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}

    subs = {"goc": [], "hoi_user": [], "bo_call": []}
    for p in preds:
        keep, ask = oracle_calls(p["tool_calls"], items[p["id"]]["label"]["tool_calls"], catalog, call_matches)
        subs["goc"].append(p)
        subs["bo_call"].append(replay(p["id"], keep))
        subs["hoi_user"].append(replay(p["id"], ask))

    sample_ids = sorted(p["id"] for p in preds)
    scores, summary = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in subs.items():
            path = Path(tmp) / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                r = evaluate_submission(str(root), a.split, str(path), filter_ids=set(sample_ids), verbose=False)
            d = {x["id"]: x["score"] for x in r["diagnostics"]}  # mẫu không có trong diagnostics là đúng hoàn toàn
            scores[name] = {i: d.get(i, 1.0) for i in sample_ids}
            summary[name] = {"mean": r["mean_score"], "by_kind": r["by_kind"]}

    kinds = {i: kind_of(items[i]["label"]["tool_calls"]) for i in sample_ids}
    best = {i: max(scores["goc"][i], scores["bo_call"][i]) for i in sample_ids}
    summary["chon_loc"] = {"mean": round(sum(best.values()) / len(best), 4)}
    up = Counter(kinds[i] for i in sample_ids if scores["bo_call"][i] > scores["goc"][i])
    down = Counter(kinds[i] for i in sample_ids if scores["bo_call"][i] < scores["goc"][i])

    base = summary["goc"]["mean"]
    print(f"{len(sample_ids)} mẫu | {a.pred}")
    for name in ("goc", "hoi_user", "bo_call", "chon_loc"):
        m = summary[name]["mean"]
        bk = " | ".join(f"{k} {v['score']:.4f}" for k, v in summary[name].get("by_kind", {}).items())
        print(f"  {name:<9} mean F1 {m:.4f} ({m - base:+.4f})  {bk}")
    print(f"  bỏ call sai làm tăng điểm: {dict(up)} | làm giảm: {dict(down)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "tang": dict(up), "giam": dict(down), "scores": scores},
                                          ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
