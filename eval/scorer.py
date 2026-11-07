"""Scorer cục bộ, khớp công thức chấm điểm chính thức của BTC (LifeMate).

Công thức:
1. State match: s['state_hash'] == gold_state_hash (replay gold tool_calls qua zaic_tools).
   Nếu sai hash: điểm = 0.
2. Tool-call match: Tên tool trùng, tham số thường trùng 100%, tham số fuzzy (x-match: fuzzy) có token-F1 > 0.6.
3. Out-of-scope / Empty: Gold rỗng -> Nộp rỗng được 1, nộp có tool được 0.
4. F1 Score: 2 * M / (|Submitted| + |Gold|).
5. Phân loại lỗi chi tiết (diagnostic categorization) phục vụ GEPA / phân tích lỗi.

Hàm dùng chung (metric GEPA, run_eval): load_split, score_sample, match_calls, task_kind, task_app, gold_hash.
"""
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASK = "sys.ask_missing_info"


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


def task_kind(G: List[Dict[str, Any]]) -> str:
    """empty (gold rỗng) | ask (có sys.ask_missing_info) | act."""
    return "empty" if not G else "ask" if any(g["name"] == ASK for g in G) else "act"


def task_app(G: List[Dict[str, Any]]) -> str:
    """App của mẫu: tiền tố tool đầu tiên không phải sys.*, hoặc tool_name trong ask; '(none)' nếu gold rỗng."""
    for g in G:
        name = (g.get("arguments") or {}).get("tool_name", "") if g["name"] == ASK else g["name"]
        if name and not name.startswith("sys."):
            return name.split(".")[0]
    return "(none)"


def match_calls(calls: List[Dict[str, Any]], G: List[Dict[str, Any]], catalog: Dict[str, Any]):
    """Ghép tham lam, không xét thứ tự. -> (cặp khớp, call thừa, gold bị thiếu)."""
    left, pairs, extra = list(G), [], []
    for c in calls:
        j = next((i for i, g in enumerate(left) if call_matches(c, g, catalog)), None)
        if j is None:
            extra.append(c)
        else:
            pairs.append((c, left.pop(j)))
    return pairs, extra, left


def score_sample(G, calls, state_hash, gold_state_hash, catalog) -> Tuple[float, Optional[str]]:
    """Điểm một mẫu theo BTC -> (score, loại lỗi hoặc None)."""
    if state_hash != gold_state_hash:
        return 0.0, "HASH_MISMATCH"
    if not G:
        return (1.0, None) if not calls else (0.0, "FALSE_ACT")  # phải rỗng nhưng lại gọi tool
    score = 2 * len(match_calls(calls, G, catalog)[0]) / (len(calls) + len(G))
    if score >= 1.0:
        return score, None
    asked, kind = any(c["name"] == ASK for c in calls), task_kind(G)
    if asked and kind == "act":
        return score, "FALSE_ASK"
    if not asked and kind == "ask":
        return score, "MISSED_ASK"
    if len(calls) != len(G):
        return score, "EXTRA_CALL" if len(calls) > len(G) else "MISSING_CALL"
    return score, "WRONG_TOOL_OR_ARGS"


def load_split(root: str, split: str):
    """-> (catalog {tên tool: spec}, các mẫu của split, lớp ToolEnv của BTC)."""
    d = Path(root) / split
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from zaic_tools import ToolEnv
    catalog = {t["name"]: t for app in json.load(open(d / f"{split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(d / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    return catalog, items, ToolEnv


def gold_hash(ToolEnv, root: str, split: str, sid: str, G: List[Dict[str, Any]]) -> str:
    """Replay gold tool_calls qua zaic_tools để lấy state_hash chuẩn."""
    env = ToolEnv(split, sid, root=str(root))
    for g in G:
        env.call(g["name"], g["arguments"])
    return env.get_state_hash()


def evaluate_submission(
    root: str,
    split: str,
    sub_path: str,
    filter_ids: Optional[set] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Chấm submission; filter_ids cố định tập cần chấm, ID mục tiêu bị thiếu nhận 0 điểm.

    Nếu filter_ids=None, chỉ chấm các ID có trong submission (hữu ích cho pipeline chạy --limit).
    """
    catalog, items, ToolEnv = load_split(root, split)
    gold = {d["id"]: d for d in items}

    sub_items = [json.loads(l) for l in open(sub_path, encoding="utf-8") if l.strip()]
    seen = set()
    for s in sub_items:
        sid = s.get("id")
        if sid not in gold:
            raise ValueError(f"Submission có ID không có trong {split}: {sid!r}")
        if sid in seen:
            raise ValueError(f"Submission có ID trùng: {sid}")
        seen.add(sid)
    # Khi chấm một subset (vd pipeline --limit 100), domain mặc định là các ID có trong file.
    # Truyền filter_ids để cố định domain và tính mẫu thiếu là 0, tránh score/pass^k bị thổi phồng.
    requested_ids = seen if filter_ids is None else set(filter_ids)
    unknown_filter_ids = requested_ids - set(gold)
    if unknown_filter_ids:
        raise ValueError(f"ID lọc không có trong {split}: {sorted(unknown_filter_ids)[:10]}")
    sub_by_id = {s["id"]: s for s in sub_items if s["id"] in requested_ids}

    scores = {}
    by_kind, by_app = defaultdict(list), defaultdict(list)
    diagnostics = []

    for sid in sorted(requested_ids):
        G = gold[sid]["label"]["tool_calls"]
        s = sub_by_id.get(sid)
        submitted_calls = s.get("tool_calls", []) if s else []
        if s is None:
            score, error_type = 0.0, "MISSING_SUBMISSION"
        else:
            score, error_type = score_sample(G, submitted_calls, s.get("state_hash"),
                                             gold_hash(ToolEnv, root, split, sid, G), catalog)
        kind = task_kind(G)
        scores[sid] = score
        by_kind[kind].append(score)
        by_app[task_app(G)].append(score)
        if error_type:
            diagnostics.append({
                "id": sid,
                "kind": kind,
                "error": error_type,
                "score": round(score, 4),
                "gold_calls": G,
                "sub_calls": submitted_calls,
                **({"missing_submission": True} if s is None else {})
            })

    avg_total = sum(scores.values()) / max(len(scores), 1)
    agg = lambda d: {k: {"score": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(d.items())}
    summary = {
        "total_samples": len(scores),
        "mean_score": round(avg_total, 4),
        "by_kind": agg(by_kind),
        "by_app": agg(by_app),
        "scores": scores,
        "error_count": len(diagnostics),
        "diagnostics": diagnostics
    }

    if verbose:
        print(f"\n===== KẾT QUẢ ĐÁNH GIÁ (N={len(scores)}) =====")
        print(f"Mean Score: {avg_total:.4f}")
        for k, info in summary["by_kind"].items():
            print(f"  {k:5s}: {info['score']:.4f} (n={info['n']})")
        if diagnostics:
            err_counts = defaultdict(int)
            for d in diagnostics:
                err_counts[d["error"]] += 1
            print("\nPhân loại lỗi:")
            for err, c in sorted(err_counts.items(), key=lambda x: -x[1]):
                print(f"  {err:20s}: {c} ({c / len(scores) * 100:.1f}%)")

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
