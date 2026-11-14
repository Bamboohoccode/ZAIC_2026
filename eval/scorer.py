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


HELD_OUT_VAL_APPS = {
    "bike_rental", "calculator", "exchange_rate", "fitness",
    "lottery", "package_tracking", "fuel_price", "gold_price", "tv",
}
HELD_OUT_TEST_APPS = {"library", "home_energy", "medication"}


def extract_sample_apps(gold_calls: List[Dict[str, Any]]) -> List[str]:
    """Trích xuất danh sách app mục tiêu từ nhãn gold (bao gồm sys.ask_missing_info -> tool_name)."""
    if not gold_calls:
        return ["_empty_"]
    apps = []
    for c in gold_calls:
        name = c["name"]
        if "." in name and not name.startswith("sys."):
            apps.append(name.split(".")[0])
        elif name == "sys.ask_missing_info":
            tn = (c.get("arguments") or {}).get("tool_name", "")
            if "." in tn:
                apps.append(tn.split(".")[0])
    return list(dict.fromkeys(apps)) or ["_empty_"]


def app_group_label(app: str) -> str:
    if app in HELD_OUT_VAL_APPS:
        return "held_out_val"
    if app in HELD_OUT_TEST_APPS:
        return "held_out_test"
    if app == "_empty_":
        return "empty"
    return "seen_train"


def evaluate_submission(
    root: str,
    split: str,
    sub_path: str,
    filter_ids: Optional[set] = None,
    verbose: bool = True,
    by_app: bool = False,
) -> Dict[str, Any]:
    """Chấm điểm submission.jsonl đối chiếu với nhãn train/split."""
    split_dir = Path(root) / split
    if str(split_dir) not in sys.path:
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
    by_app_scores = defaultdict(list)
    err_counts = defaultdict(int)
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
        for app in extract_sample_apps(G):
            by_app_scores[app].append(score)

        if error_type:
            err_counts[error_type] += 1
            diagnostics.append({
                "id": sid,
                "kind": kind,
                "apps": extract_sample_apps(G),
                "error": error_type,
                "score": round(score, 4),
                "gold_calls": G,
                "sub_calls": submitted_calls
            })

    avg_total = sum(total_scores) / max(len(total_scores), 1)
    by_app_summary = {
        app: {
            "score": round(sum(vals) / len(vals), 4),
            "n": len(vals),
            "group": app_group_label(app),
        }
        for app, vals in sorted(by_app_scores.items())
    }
    by_app_group = defaultdict(list)
    for app, info in by_app_summary.items():
        by_app_group[info["group"]].extend(by_app_scores[app])

    summary = {
        "total_samples": len(total_scores),
        "mean_score": round(avg_total, 4),
        "by_kind": {k: {"score": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(by_kind.items())},
        "by_app": by_app_summary,
        "by_app_group": {
            grp: {"score": round(sum(v) / len(v), 4), "n_evals": len(v)}
            for grp, v in sorted(by_app_group.items())
        },
        "error_counts": dict(err_counts),
        "error_count": len(diagnostics),
        "diagnostics": diagnostics
    }

    if verbose:
        print(f"\n===== KẾT QUẢ ĐÁNH GIÁ (N={len(total_scores)}) =====")
        print(f"Mean Score: {avg_total:.4f}")
        for k, info in summary["by_kind"].items():
            print(f"  {k:5s}: {info['score']:.4f} (n={info['n']})")
        if err_counts:
            print("\nPhân loại lỗi:")
            for err, c in sorted(err_counts.items(), key=lambda x: -x[1]):
                print(f"  {err:20s}: {c} ({c / len(total_scores) * 100:.1f}%)")
        if by_app and by_app_summary:
            print("\nĐiểm theo App (chia theo app, ưu tiên nhóm held-out):")
            group_order = {"held_out_val": 0, "held_out_test": 1, "seen_train": 2, "empty": 3}
            for grp_name in sorted(summary["by_app_group"], key=lambda g: group_order.get(g, 9)):
                ginfo = summary["by_app_group"][grp_name]
                print(f"  [{grp_name}] avg={ginfo['score']:.4f} (n_evals={ginfo['n_evals']})")
                for app, ainfo in sorted(by_app_summary.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
                    if ainfo["group"] == grp_name:
                        print(f"    - {app:20s}: {ainfo['score']:.4f} (n={ainfo['n']})")

    return summary


def print_comparison(
    before_path: str,
    after_path: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    by_app: bool = False,
) -> None:
    """In bảng so sánh Trước / Sau giữa 2 file submission."""
    print(f"\n===== SO SÁNH TRƯỚC vs SAU (N={before['total_samples']} vs {after['total_samples']}) =====")
    print(f"Trước : {before_path}")
    print(f"Sau   : {after_path}")

    delta = after["mean_score"] - before["mean_score"]
    print(f"\nMean Score: {before['mean_score']:.4f} -> {after['mean_score']:.4f} ({delta:+.4f})")
    for k in sorted(set(before["by_kind"]) | set(after["by_kind"])):
        b = before["by_kind"].get(k, {"score": 0.0, "n": 0})
        a = after["by_kind"].get(k, {"score": 0.0, "n": 0})
        print(f"  {k:5s} (n={a['n']:3d}): {b['score']:.4f} -> {a['score']:.4f} ({a['score'] - b['score']:+.4f})")

    all_errs = sorted(set(before["error_counts"]) | set(after["error_counts"]))
    if all_errs:
        print("\nSo sánh lỗi:")
        for e in all_errs:
            bc = before["error_counts"].get(e, 0)
            ac = after["error_counts"].get(e, 0)
            print(f"  {e:20s}: {bc:3d} -> {ac:3d} ({ac - bc:+d})")

    if by_app:
        print("\nSo sánh theo từng App:")
        group_order = {"held_out_val": 0, "held_out_test": 1, "seen_train": 2, "empty": 3}
        all_apps = sorted(
            set(before["by_app"]) | set(after["by_app"]),
            key=lambda app: (
                group_order.get(app_group_label(app), 9),
                -after["by_app"].get(app, before["by_app"].get(app, {})).get("n", 0),
                app,
            ),
        )
        print(f"  {'App':20s} | {'Group':13s} | {'n':>3s} | {'Before':>7s} | {'After':>7s} | {'Delta':>7s}")
        print("  " + "-" * 72)
        for app in all_apps:
            b = before["by_app"].get(app, {"score": 0.0, "n": 0, "group": app_group_label(app)})
            a = after["by_app"].get(app, {"score": 0.0, "n": b["n"], "group": app_group_label(app)})
            d = a["score"] - b["score"]
            print(f"  {app:20s} | {a['group']:13s} | {a['n']:3d} | {b['score']:7.4f} | {a['score']:7.4f} | {d:+7.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sub", required=True, help="Path to baseline/before submission.jsonl")
    ap.add_argument("--compare", default="", help="Optional path to second (after) submission.jsonl for Before vs After")
    ap.add_argument("--dev-ids", default="eval/splits/dev_ids.txt", help="Optional filter by ids file (e.g., dev_ids.txt or test_ids.txt)")
    ap.add_argument("--by-app", action="store_true", help="Print per-app breakdown table")
    a = ap.parse_args()

    filter_set = set(open(a.dev_ids, encoding="utf-8").read().splitlines()) if a.dev_ids and Path(a.dev_ids).exists() else None
    if a.compare:
        res_before = evaluate_submission(a.root, a.split, a.sub, filter_ids=filter_set, verbose=False, by_app=a.by_app)
        res_after = evaluate_submission(a.root, a.split, a.compare, filter_ids=filter_set, verbose=False, by_app=a.by_app)
        print_comparison(a.sub, a.compare, res_before, res_after, by_app=a.by_app)
    else:
        evaluate_submission(a.root, a.split, a.sub, filter_ids=filter_set, by_app=a.by_app)
