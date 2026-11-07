"""Phân loại lỗi của Planner theo việc Verifier có bắt được hay không (bước 2, task_flow/ha.md).

Câu hỏi: trong các call sai của Planner, bao nhiêu phần trăm phát hiện được CHỈ bằng cách so với
action đã lưu (diff.py, không cần query)? Đây là trần đóng góp của Verifier không dùng query.

Hai chế độ:
  full     có dữ liệu BTC: so dự đoán với gold, replay call qua ToolEnv để có kết quả tool, chạy diff.py
           trên từng call trong ngữ cảnh lúc nó được gọi.
             python lifemate/verifier/analyze_errors.py --root data/raw/train+public_test --split train \
                 --pred runs/dev_v4.jsonl --ids eval/splits/dev_ids.txt --out runs/errors_v4.json
  partial  KHÔNG có dữ liệu: chỉ có file dự đoán + loại read/write của tool (test/results_train.jsonl).
           Chỉ bắt được lỗi theo chuỗi call (lặp write, lặp read, call sau ask), không có gold.
             python lifemate/verifier/analyze_errors.py --pred dev_v4.jsonl --kinds test/results_train.jsonl

Nhóm cho mỗi call dự đoán không khớp gold (call thừa/sai):
  diff_bat_duoc  diff.py báo mức loi -> Verifier luật sửa được
  hoi_thua       dự đoán gọi sys.ask_missing_info nhưng gold không hỏi
  can_query      còn lại: sai ý user nhưng nhất quán với state -> Verifier không query không bắt được
Cho mỗi call gold không được dự đoán: thieu_call / thieu_ask (Verifier không thêm call được, việc của Planner).
Báo nhầm: call dự đoán ĐÚNG (khớp gold) mà diff.py vẫn báo loi.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lifemate.verifier.diff import ASK_TOOL, LOI, Action, actions_from_item, diff  # noqa: E402


def match_calls(pred, gold, catalog, matches):
    """Ghép một-một như scorer BTC. Trả (chỉ số pred khớp, chỉ số gold khớp)."""
    left, mp, mg = list(range(len(gold))), set(), set()
    for i, c in enumerate(pred):
        j = next((j for j in left if matches(c, gold[j], catalog)), None)
        if j is not None:
            left.remove(j)
            mp.add(i)
            mg.add(j)
    return mp, mg


def user_texts(item):
    return [h.get("audio_text", "") for h in item.get("history", []) if h.get("role") == "user"] + [item.get("audio_text", "")]


def analyze_sample(item, pred, results, catalog, matches):
    """item: mẫu dataset (có label). pred: call đã chạy theo thứ tự. results: kết quả ToolEnv tương ứng.
    Trả dict thống kê của mẫu, thuần Python để test được không cần ToolEnv."""
    gold = item["label"]["tool_calls"]
    mp, mg = match_calls(pred, gold, catalog, matches)
    hist, texts = actions_from_item(item), user_texts(item)
    gold_ask = any(g["name"] == ASK_TOOL for g in gold)
    calls, done = [], []
    for i, c in enumerate(pred):
        flags = [m for m in diff([c], hist + done, catalog, texts=texts) if m.severity == LOI]
        if i in mp:
            group = "dung_bi_bao_nham" if flags else "dung"
        elif flags:
            group = "diff_bat_duoc"
        elif c["name"] == ASK_TOOL and not gold_ask:
            group = "hoi_thua"
        else:
            group = "can_query"
        calls.append({"i": i, "name": c["name"], "group": group, "flags": [m.kind for m in flags]})
        done.append(Action(c["name"], c.get("arguments") or {}, results[i] if i < len(results) else None))
    missing = [{"name": g["name"], "group": "thieu_ask" if g["name"] == ASK_TOOL else "thieu_call"}
               for j, g in enumerate(gold) if j not in mg]
    kind = "empty" if not gold else "ask" if gold_ask else "act"
    return {"id": item["id"], "kind": kind, "calls": calls, "missing": missing,
            "exact": len(mp) == len(pred) == len(gold)}


def summarize(samples):
    groups = Counter(c["group"] for s in samples for c in s["calls"])
    flags = Counter(f for s in samples for c in s["calls"] if c["group"] == "diff_bat_duoc" for f in c["flags"])
    missing = Counter(m["group"] for s in samples for m in s["missing"])
    wrong = [s for s in samples if not s["exact"]]
    extra = groups["diff_bat_duoc"] + groups["hoi_thua"] + groups["can_query"]
    ok = groups["dung"] + groups["dung_bi_bao_nham"]
    return {
        "n_mau": len(samples), "n_mau_sai": len(wrong),
        "mau_sai_co_call_diff_bat_duoc": sum(any(c["group"] == "diff_bat_duoc" for c in s["calls"]) for s in wrong),
        "call_du_doan": dict(groups), "call_thua_sai": extra,
        "ti_le_diff_bat_trong_call_sai": groups["diff_bat_duoc"] / extra if extra else None,
        "ti_le_bao_nham_tren_call_dung": groups["dung_bi_bao_nham"] / ok if ok else None,
        "loai_loi_diff": dict(flags), "call_gold_bi_thieu": dict(missing),
        "theo_nhan": {k: sum(s["kind"] == k and not s["exact"] for s in samples) for k in ("act", "ask", "empty")},
    }


def run_full(a):
    root = Path(a.root)
    sys.path.insert(0, str(root / a.split))
    from zaic_tools import ToolEnv
    from eval.scorer import call_matches
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()}
    preds = [json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()]
    ids = set(Path(a.ids).read_text().split()) if a.ids else None
    samples = []
    for p in preds:
        if (ids and p["id"] not in ids) or p["id"] not in items:
            continue
        env = ToolEnv(a.split, p["id"], root=str(root))
        results = [env.call(c["name"], c.get("arguments") or {}) for c in p["tool_calls"]]
        samples.append(analyze_sample(items[p["id"]], p["tool_calls"], results, catalog, call_matches))
    return samples


def run_partial(a):
    kinds = {}
    for l in open(a.kinds, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            kinds[r["tool"]] = r["kind"]
    catalog = {n: {"kind": k, "parameters": {"properties": {}}} for n, k in kinds.items()}  # không có x-ref
    catalog.setdefault(ASK_TOOL, {"kind": "meta", "parameters": {"properties": {}}})
    preds = [json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()]
    ids = set(Path(a.ids).read_text().split()) if a.ids else None
    flagged, per_kind, n_calls, n_unknown = [], Counter(), 0, 0
    for p in preds:
        if ids and p["id"] not in ids:
            continue
        done = []
        for i, c in enumerate(p["tool_calls"]):
            n_calls += 1
            if c["name"] not in catalog:
                n_unknown += 1
            ms = [m for m in diff([c], done, catalog) if m.severity == LOI and m.kind != "UNKNOWN_TOOL"]
            for m in ms:
                per_kind[m.kind] += 1
                flagged.append({"id": p["id"], "i": i, "name": c["name"], "kind": m.kind})
            done.append(Action(c["name"], c.get("arguments") or {}))
    n = len([p for p in preds if not ids or p["id"] in ids])
    return {"n_mau": n, "n_call": n_calls, "call_tool_khong_ro_kind": n_unknown,
            "n_mau_co_loi_diff": len({f["id"] for f in flagged}), "loai_loi_diff": dict(per_kind), "chi_tiet": flagged}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", required=True, help="submission jsonl (id, tool_calls, state_hash)")
    ap.add_argument("--root", help="thư mục chứa các split (chế độ full)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--kinds", help="jsonl có tool + kind, vd test/results_train.jsonl (chế độ partial)")
    ap.add_argument("--ids", help="chỉ phân tích các id trong file này, vd eval/splits/dev_ids.txt")
    ap.add_argument("--out", help="ghi kết quả JSON")
    a = ap.parse_args()
    if a.root:
        samples = run_full(a)
        res = {"che_do": "full", "tong_hop": summarize(samples), "mau": samples}
    elif a.kinds:
        res = {"che_do": "partial", **run_partial(a)}
    else:
        sys.exit("Cần --root (full) hoặc --kinds (partial)")
    show = res.get("tong_hop") or {k: v for k, v in res.items() if k != "chi_tiet"}
    print(json.dumps(show, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
