"""Verifier giúp được bao nhiêu điểm: trần (Verifier hoàn hảo, dùng gold) và Verifier thật.

Từ một file dự đoán của Planner, dựng các bản nộp, replay qua ToolEnv để lấy state_hash,
rồi chấm bằng eval/scorer.py (cùng công thức BTC).

Trần (luôn chạy):
  goc        dự đoán gốc
  hoi_user   gặp call sai đầu tiên thì thay bằng sys.ask_missing_info(tool, tham số của call đó) và dừng
             (thiết kế trong PROJECT_OVERVIEW mục 4c)
  bo_call    bỏ mọi call sai, giữ call khớp gold
  chon_loc   mỗi mẫu lấy điểm cao hơn giữa goc và bo_call (Verifier biết khi nào bỏ có lợi)

Verifier thật (chọn một): mỗi call dự đoán được chấm P(ĐÚNG); với từng ngưỡng, bỏ call có P < ngưỡng.
  --rules           diff.py (luật, không cần model): SAI -> P=0
  --model DIR       checkpoint từ train.py (cần torch, transformers); --with-query nếu model train bằng ver_q
  --scores FILE     điểm đã tính sẵn, jsonl {id, step, p}; step = vị trí call trong dự đoán
Input của model giống data.examples_from_pred: state = history + các call dự đoán trước nó (kèm kết quả).
Đây là mô phỏng tĩnh: bỏ một call thì Planner thật sẽ lập lại kế hoạch, ở đây chỉ xoá call đó.

  python lifemate/verifier/oracle_ceiling.py --root D:/Downloads/ZAIC/train+public_test --split train \
      --pred dev_v4.jsonl --ids eval/splits/dev_ids.txt --out runs/oracle_v4.json
  python lifemate/verifier/oracle_ceiling.py ... --model runs/ver_q_mmbert --with-query --save-scores runs/p_v4.jsonl

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
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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


def rule_scores(items, preds, results, catalog):
    """P(ĐÚNG) theo diff.py: 0 nếu verdict WRONG, 1 nếu không."""
    from lifemate.verifier.data import user_texts
    from lifemate.verifier.diff import Action, actions_from_item, diff, verdict
    out = {}
    for p in preds:
        item, done = items[p["id"]], []
        hist = actions_from_item(item)
        for i, c in enumerate(p["tool_calls"]):
            mm = diff([c], hist + done, catalog, texts=user_texts(item))
            out[(p["id"], i)] = 0.0 if verdict(mm)["label"] == "WRONG" else 1.0
            res = results[p["id"]]
            done.append(Action(c["name"], c.get("arguments") or {}, res[i] if i < len(res) else None))
    return out


def model_scores(rows, model_dir, device="auto", bs=16, max_len=1024):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from lifemate.verifier.train import split_text
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    max_len = min(max_len, getattr(tok, "model_max_length", max_len) or max_len)
    out = {}
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            heads, tails = zip(*(split_text(r["text"]) for r in batch))
            enc = tok(list(heads), list(tails), truncation="only_second", max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                probs = torch.softmax(model(**enc).logits.float(), -1)[:, 1].tolist()
            for r, pr in zip(batch, probs):
                out[(r["id"], r["step"])] = pr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ids", help="chỉ chấm các id này, vd eval/splits/dev_ids.txt")
    ap.add_argument("--out", help="ghi kết quả JSON")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rules", action="store_true", help="Verifier thật = diff.py")
    g.add_argument("--model", help="checkpoint từ train.py")
    g.add_argument("--scores", help="jsonl {id, step, p} tính sẵn")
    ap.add_argument("--with-query", action="store_true", help="input model kèm lời user (model train bằng ver_q)")
    ap.add_argument("--thresholds", default=",".join(map(str, THRESHOLDS)))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-scores", help="ghi điểm P(ĐÚNG) từng call ra jsonl")
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
    sample_ids = sorted(p["id"] for p in preds)

    def replay(sid, calls):
        env = ToolEnv(a.split, sid, root=str(root))
        res = [env.call(c["name"], c.get("arguments") or {}) for c in calls]
        return {"id": sid, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}, res

    def score(rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                r = evaluate_submission(str(root), a.split, str(path), filter_ids=set(sample_ids), verbose=False)
        d = {x["id"]: x["score"] for x in r["diagnostics"]}  # mẫu không có trong diagnostics là đúng hoàn toàn
        return {i: d.get(i, 1.0) for i in sample_ids}, {"mean": r["mean_score"], "by_kind": r["by_kind"]}

    subs = {"goc": [], "hoi_user": [], "bo_call": []}
    results = {}
    for p in preds:
        keep, ask = oracle_calls(p["tool_calls"], items[p["id"]]["label"]["tool_calls"], catalog, call_matches)
        subs["goc"].append(p)
        subs["bo_call"].append(replay(p["id"], keep)[0])
        subs["hoi_user"].append(replay(p["id"], ask)[0])
        results[p["id"]] = replay(p["id"], p["tool_calls"])[1]

    scores, summary = {}, {}
    for name, rows in subs.items():
        scores[name], summary[name] = score(rows)

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
    report = {"summary": summary, "tang": dict(up), "giam": dict(down), "scores": scores}

    if a.rules or a.model or a.scores:
        from lifemate.verifier.data import examples_from_pred
        from lifemate.verifier.train import auc
        rows = [r for p in preds for r in examples_from_pred(items[p["id"]], p["tool_calls"], results[p["id"]],
                                                              catalog, call_matches, a.with_query)]
        gold_ok = {(r["id"], r["step"]): r["label"] for r in rows}
        if a.rules:
            name, ps = "luat", rule_scores(items, preds, results, catalog)
        elif a.model:
            name, ps = Path(a.model).name, model_scores(rows, a.model, a.device)
        else:
            name = Path(a.scores).stem
            ps = {(r["id"], r["step"]): r["p"] for r in map(json.loads, open(a.scores, encoding="utf-8")) if r}
        missing = [k for k in gold_ok if k not in ps]
        if missing:
            sys.exit(f"thiếu điểm cho {len(missing)} call, vd {missing[:3]}")
        if a.save_scores:
            Path(a.save_scores).parent.mkdir(parents=True, exist_ok=True)
            Path(a.save_scores).write_text("".join(json.dumps({"id": i, "step": s, "p": round(p, 6), "dung": gold_ok[(i, s)]})
                                                   + "\n" for (i, s), p in sorted(ps.items())), encoding="utf-8")
        keys = sorted(gold_ok)
        auc_real = auc([ps[k] for k in keys], [gold_ok[k] for k in keys])
        n_ok = sum(gold_ok.values())
        print(f"\nVerifier thật: {name} | {len(keys)} call ({n_ok} đúng, {len(keys) - n_ok} sai) | AUC trên call thật {auc_real:.4f}")
        thresholds = [0.5] if a.rules else [float(t) for t in a.thresholds.split(",")]
        sweep = []
        for t in thresholds:
            drop = {k for k in keys if ps[k] < t}
            rows_t = [replay(p["id"], [c for i, c in enumerate(p["tool_calls"]) if (p["id"], i) not in drop])[0] for p in preds]
            sc, sm = score(rows_t)
            r = {"nguong": t, "mean": sm["mean"], "delta": round(sm["mean"] - base, 4), "bo": len(drop),
                 "bo_nham_call_dung": sum(gold_ok[k] for k in drop), "bo_dung_call_sai": sum(1 - gold_ok[k] for k in drop),
                 "mau_tang": sum(sc[i] > scores["goc"][i] for i in sample_ids),
                 "mau_giam": sum(sc[i] < scores["goc"][i] for i in sample_ids)}
            sweep.append(r)
            print(f"  ngưỡng {t:.2f}: mean F1 {r['mean']:.4f} ({r['delta']:+.4f}) | bỏ {r['bo']} call: "
                  f"{r['bo_dung_call_sai']} sai, {r['bo_nham_call_dung']} đúng (bỏ nhầm) | mẫu tăng {r['mau_tang']}, giảm {r['mau_giam']}")
        report["verifier"] = {"ten": name, "auc_call_that": auc_real, "n_call": len(keys), "n_dung": n_ok, "quet_nguong": sweep}

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
