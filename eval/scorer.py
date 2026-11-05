"""Scorer cục bộ, khớp công thức chấm điểm chính thức của BTC (LifeMate).

Công thức:
1. State match: s['state_hash'] == gold_state_hash (replay gold tool_calls qua zaic_tools).
   Nếu sai hash: điểm = 0.
2. Tool-call match: Tên tool trùng, tham số thường trùng 100%, tham số fuzzy (x-match: fuzzy) có token-F1 > 0.6.
3. Out-of-scope / Empty: Gold rỗng -> Nộp rỗng được 1, nộp có tool được 0.
4. F1 Score: 2 * M / (|Submitted| + |Gold|).
5. Phân loại lỗi chi tiết (diagnostic categorization) phục vụ GEPA / phân tích lỗi.
"""
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def toks(s: Any) -> List[str]:
    return unicodedata.normalize("NFC", str(s)).lower().split()


def token_f1(a: Any, b: Any) -> float:
    wa, wb = toks(a), toks(b)
    common = sum(min(wa.count(w), wb.count(w)) for w in set(wa))
    return 2 * common / (len(wa) + len(wb)) if (wa and wb) else 0.0


def call_matches(c: Dict[str, Any], g: Dict[str, Any], catalog: Dict[str, Any]) -> bool:
    if c["name"] != g["name"]:
        return False
    props = catalog.get(c["name"], {}).get("parameters", {}).get("properties", {})
    ca, ga = c.get("arguments") or {}, g.get("arguments") or {}
    if set(ca) != set(ga):
        return False
    for k in ga:
        is_fuzzy = props.get(k, {}).get("x-match") == "fuzzy"
        if is_fuzzy:
            if token_f1(ca[k], ga[k]) <= 0.6:
                return False
        else:
            if ca[k] != ga[k]:
                return False
    return True


def evaluate_submission(
    root: str,
    split: str,
    sub_path: str,
    filter_ids: Optional[set] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Chấm điểm submission.jsonl đối chiếu với nhãn train/split."""
    split_dir = Path(root) / split
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    catalog = {
        t["name"]: t
        for app in json.load(open(split_dir / f"{split}_tools_catalog.json", encoding="utf-8"))["apps"]
        for t in app["tools"]
    }
    gold = {
        d["id"]: d
        for d in map(json.loads, open(split_dir / f"{split}.jsonl", encoding="utf-8"))
    }

    sub_items = [json.loads(l) for l in open(sub_path, encoding="utf-8") if l.strip()]
    if filter_ids:
        sub_items = [s for s in sub_items if s["id"] in filter_ids]

    total_scores = []
    by_kind = defaultdict(list)
    diagnostics = []

    for s in sub_items:
        sid = s["id"]
        g_item = gold[sid]
        G = g_item["label"]["tool_calls"]
        submitted_calls = s.get("tool_calls", [])

        # Replay gold calls qua zaic_tools để lấy gold state_hash
        env = ToolEnv(split, sid, root=str(root))
        for g in G:
            env.call(g["name"], g["arguments"])
        gold_hash = env.get_state_hash()

        kind = "empty" if not G else "ask" if any(g["name"] == "sys.ask_missing_info" for g in G) else "act"

        # Đánh giá
        error_type = None
        if not G:
            if not submitted_calls:
                score = 1.0
            else:
                score = 0.0
                error_type = "FALSE_ACT"  # Phải rỗng nhưng lại gọi tool
        elif s.get("state_hash") != gold_hash:
            score = 0.0
            error_type = "HASH_MISMATCH"
        else:
            left_gold = list(G)
            matched = 0
            for c in submitted_calls:
                j = next((idx for idx, g in enumerate(left_gold) if call_matches(c, g, catalog)), None)
                if j is not None:
                    left_gold.pop(j)
                    matched += 1
            denom = len(submitted_calls) + len(G)
            score = 2 * matched / denom if denom else 0.0

            if score < 1.0:
                if any(c["name"] == "sys.ask_missing_info" for c in submitted_calls) and kind == "act":
                    error_type = "FALSE_ASK"
                elif not any(c["name"] == "sys.ask_missing_info" for c in submitted_calls) and kind == "ask":
                    error_type = "MISSED_ASK"
                elif len(submitted_calls) > len(G):
                    error_type = "EXTRA_CALL"
                elif len(submitted_calls) < len(G):
                    error_type = "MISSING_CALL"
                else:
                    error_type = "WRONG_TOOL_OR_ARGS"

        total_scores.append(score)
        by_kind[kind].append(score)

        if error_type:
            diagnostics.append({
                "id": sid,
                "kind": kind,
                "error": error_type,
                "score": round(score, 4),
                "gold_calls": G,
                "sub_calls": submitted_calls
            })

    avg_total = sum(total_scores) / max(len(total_scores), 1)
    summary = {
        "total_samples": len(total_scores),
        "mean_score": round(avg_total, 4),
        "by_kind": {k: {"score": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(by_kind.items())},
        "error_count": len(diagnostics),
        "diagnostics": diagnostics
    }

    if verbose:
        print(f"\n===== KẾT QUẢ ĐÁNH GIÁ (N={len(total_scores)}) =====")
        print(f"Mean Score: {avg_total:.4f}")
        for k, info in summary["by_kind"].items():
            print(f"  {k:5s}: {info['score']:.4f} (n={info['n']})")
        if diagnostics:
            err_counts = defaultdict(int)
            for d in diagnostics:
                err_counts[d["error"]] += 1
            print("\nPhân loại lỗi:")
            for err, c in sorted(err_counts.items(), key=lambda x: -x[1]):
                print(f"  {err:20s}: {c} ({c / len(total_scores) * 100:.1f}%)")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sub", required=True, help="Path to submission.jsonl")
    ap.add_argument("--dev-ids", default="eval/splits/dev_ids.txt", help="Optional filter by dev_ids")
    a = ap.parse_args()

    filter_set = set(open(a.dev_ids, encoding="utf-8").read().splitlines()) if Path(a.dev_ids).exists() else None
    evaluate_submission(a.root, a.split, a.sub, filter_ids=filter_set)
