"""Chấm và phân loại lỗi một lần chạy pipeline (dùng cho baseline và mỗi vòng WSP).

Đầu vào: submission (`--sub`) và trace (`<sub>.trace.jsonl`, do lifemate/pipeline.py ghi) trên split có nhãn (train).
Điểm: dùng eval/scorer.py của nhóm (công thức BTC). Thêm các lỗi chỉ thấy trong trace:
- think_truncated: có "<think>" mà không có "</think>" (bị cắt ở MAX_NEW token)
- repetitive: một đoạn >= 30 ký tự lặp >= 3 lần trong cùng một lượt sinh (dấu hiệu vòng lặp)
- bad_tool_call_format: có "<tool_call>" nhưng không parse được call nào
- no_call_but_gold: không gọi tool nào trong khi gold có call
Xuất: tóm tắt JSON (`--out`) và danh sách mẫu kèm lượt sinh đầu tiên (`--export`) để directions.py dùng
(nhãn "bad" = mẫu có think_truncated hoặc repetitive; "good" = mẫu điểm 1).

Chạy: python optimize/wsp/analyze.py --sub runs/x.jsonl --ids optimize/wsp/dev_subset50_20261107.txt --out runs/x.summary.json
"""
import argparse, json, re, sys, collections
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "eval"))
from scorer import evaluate_submission  # noqa: E402

TC = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_ok(text):
    n = 0
    for raw in TC.findall(text):
        try:
            c = json.loads(raw)
            if "name" in c:
                n += 1
        except ValueError:
            pass
    return n


def repetitive(text, size=30, times=3):
    t = re.sub(r"\s+", " ", text)
    if len(t) < size * times:
        return False
    seen = collections.Counter(t[i:i + size] for i in range(0, len(t) - size, 10))
    return any(c >= times for c in seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw" / "train+public_test"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--trace")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--export", help="jsonl các mẫu good/bad kèm lượt sinh đầu, cho directions.py")
    ap.add_argument("--label-rule", choices=["cutloop", "score"], default="cutloop",
                    help='cutloop: bad = thinking bị cắt/lặp. score: bad = điểm 0 (hoặc cắt/lặp), good = điểm 1; '
                         "thiếu thì lấy thêm mẫu điểm thấp nhất/cao nhất cho đủ 2 mỗi nhóm")
    a = ap.parse_args()
    ids = {l.strip() for l in open(a.ids, encoding="utf-8") if l.strip()} if a.ids else None
    summ = evaluate_submission(a.root, a.split, a.sub, filter_ids=ids, verbose=False)
    diag = {d["id"]: d for d in summ["diagnostics"]}
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.root) / a.split / f"{a.split}.jsonl", encoding="utf-8")}
    traces = {}
    for l in open(a.trace or a.sub + ".trace.jsonl", encoding="utf-8"):
        t = json.loads(l)
        if ids is None or t["id"] in ids:
            traces[t["id"]] = t
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(a.sub, encoding="utf-8") if l.strip()}

    cat = collections.Counter()
    per, export = [], []
    secs = []
    for sid, t in traces.items():
        plans = [s["plan"] for s in t["steps"] if "plan" in s]
        flags = set()
        for p in plans:
            if "<think>" in p and "</think>" not in p:
                flags.add("think_truncated")
            if repetitive(p):
                flags.add("repetitive")
            if "<tool_call>" in p and parse_ok(p) == 0:
                flags.add("bad_tool_call_format")
        if any("error" in s for s in t["steps"]):
            flags.add("pipeline_error")
        if not sub.get(sid, {}).get("tool_calls") and gold[sid]["label"]["tool_calls"]:
            flags.add("no_call_but_gold")
        err = diag.get(sid, {}).get("error")
        if err:
            flags.add(err)
        for f in flags:
            cat[f] += 1
        secs.append(t.get("seconds", 0))
        score = diag[sid]["score"] if sid in diag else 1.0
        per.append({"id": sid, "score": score, "flags": sorted(flags), "n_steps": len(plans),
                    "max_plan_chars": max((len(p) for p in plans), default=0)})
        cut = bool(flags & {"think_truncated", "repetitive"})
        if a.label_rule == "score":
            label = "bad" if cut or score == 0.0 else "good" if score == 1.0 else None
        else:
            label = "bad" if cut else "good" if score == 1.0 else None
        if plans:
            export.append({"id": sid, "label": label, "score": score, "first_plan": plans[0]})
    if a.label_rule == "score":  # đủ tối thiểu 2 mẫu mỗi nhóm: lấy thêm từ mẫu chưa có nhãn, theo điểm
        for lab, rev in (("good", True), ("bad", False)):
            rest = sorted((e for e in export if e["label"] is None), key=lambda e: e["score"], reverse=rev)
            while sum(e["label"] == lab for e in export) < 2 and rest:
                rest.pop(0)["label"] = lab
    export = [e for e in export if e["label"]]

    n = max(len(traces), 1)
    out = {"n": len(traces), "mean_score": summ["mean_score"], "by_kind": summ["by_kind"],
           "flags": {k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in cat.most_common()},
           "avg_seconds": round(sum(secs) / n, 2), "max_seconds": max(secs, default=0), "per_sample": per}
    print(f"N={out['n']} mean={out['mean_score']} by_kind={out['by_kind']} avg_s={out['avg_seconds']} max_s={out['max_seconds']}")
    for k, v in out["flags"].items():
        print(f"  {k:22s} {v['count']:4d} ({v['pct']}%)")
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.export:
        with open(a.export, "w", encoding="utf-8") as f:
            for e in export:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print("export:", collections.Counter(e["label"] for e in export), "->", a.export)


if __name__ == "__main__":
    main()
