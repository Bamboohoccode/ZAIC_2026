"""Sinh dữ liệu ĐÚNG/SAI cho Verifier từ train (bước 3, task_flow/ha.md; PROJECT_OVERVIEW.md mục 5.3c).

Mỗi mẫu train có gold tool_calls g_0..g_n. Với bước k: state = action trong history + g_0..g_{k-1}
(kèm kết quả tool, replay qua ToolEnv), plan = [g_k] -> nhãn 1 (ĐÚNG). Mẫu SAI (nhãn 0) làm hỏng plan:

  Phát hiện được chỉ từ state (luôn sinh):
    dup_write    lặp lại một call ghi đã chạy trong lượt
    dup_read     lặp lại một call đọc đã chạy trong lượt
    unseen_id    đổi tham số x-ref (ID) thành giá trị chưa xuất hiện
    after_ask    thêm call sau khi đã hỏi user
  Cần query mới biết sai (chỉ sinh với --with-query, input có kèm lời user):
    swap_tool    đổi sang tool khác cùng app, cùng loại read/write
    change_value đổi giá trị tham số: số, ngày giờ ISO, enum
    drop_arg     bỏ một tham số không bắt buộc
    ask_instead  hỏi user tham số mà gold đã có giá trị

Lỗi thật (--pred): call của Planner trên mẫu train (không thuộc dev) khớp gold -> 1, không khớp -> 0.

Pipeline hiện tại đã chặn sẵn ID bịa, đọc lặp và call sau ask (lifemate/pipeline.py run_agent), và phân
tích dev_v4 cho thấy 0 lỗi loại state-only lọt ra (analyze_errors.py, 07/11). Chế độ --with-query có để so
xem Verifier có query bắt được thêm bao nhiêu.

Chạy (cần dữ liệu BTC):
  python lifemate/verifier/data.py --root data/raw/train+public_test --out runs/verifier_data.jsonl
  python lifemate/verifier/data.py --root ... --with-query --out runs/verifier_data_q.jsonl
Mỗi dòng: {id, split (train|dev theo eval/splits/dev_ids.txt), app, step, kind, label, text}.
"""
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lifemate.verifier.diff import ASK_TOOL, Action, _blob, _seen, actions_from_item, to_text  # noqa: E402

STATE_KINDS = ("dup_write", "dup_read", "unseen_id", "after_ask")
QUERY_KINDS = ("swap_tool", "change_value", "drop_arg", "ask_instead")


def app_of(name):
    return name.split(".", 1)[0]


def user_texts(item):
    return [h.get("audio_text", "") for h in item.get("history", []) if h.get("role") == "user"] + [item.get("audio_text", "")]


def render(actions, plan, query=None):
    text = to_text(actions, plan)
    return f"Yêu cầu mới nhất của người dùng: {query}\n{text}" if query is not None else text


# ---------------------------------------------------------------- làm hỏng plan

def _fake_id(value, blob, rng):
    """ID cùng dạng (giữ tiền tố, đổi phần số/chữ cuối) chưa xuất hiện ở đâu."""
    for _ in range(20):
        m = re.match(r"^(.*?)(\d+)$", value)
        cand = f"{m.group(1)}{int(m.group(2)) + rng.randint(1, 97)}" if m else f"{value}_{rng.randint(2, 99)}"
        if not _seen(cand, blob):
            return cand
    return None


def _shift_iso(s):
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%H:%M"):
        try:
            d = datetime.strptime(s, fmt)
        except ValueError:
            continue
        d += timedelta(hours=1) if fmt == "%H:%M" else timedelta(days=1)
        out = d.strftime(fmt)
        return out[:-2] + ":" + out[-2:] if fmt.endswith("%z") and out[-5] in "+-" else out
    return None


def _changed(v, prop, rng):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v * 2 if v else 1
    if isinstance(v, str):
        enum = [e for e in prop.get("enum", []) if e != v]
        if enum:
            return rng.choice(enum)
        return _shift_iso(v)
    return None


def corruptions(item, gold, k, actions, catalog, rng, with_query):
    """Các plan SAI cho bước k: list (kind, plan). actions: state trước bước k (đã có kết quả)."""
    g = gold[k]
    name, args = g["name"], g.get("arguments") or {}
    spec = catalog.get(name, {})
    props = spec.get("parameters", {}).get("properties", {})
    required = set(spec.get("parameters", {}).get("required", []))
    blob = _blob(actions, user_texts(item))
    cur = [a for a in actions if a.current_turn and a.name != ASK_TOOL]
    gkey = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
    out = []

    def other(a):
        return json.dumps([a.name, a.arguments], sort_keys=True, ensure_ascii=False) != gkey

    writes = [a for a in cur if catalog.get(a.name, {}).get("kind") == "write" and other(a)]
    reads = [a for a in cur if catalog.get(a.name, {}).get("kind") == "read" and other(a)]
    if writes:
        a = rng.choice(writes)
        out.append(("dup_write", [{"name": a.name, "arguments": a.arguments}]))
    if reads:
        a = rng.choice(reads)
        out.append(("dup_read", [{"name": a.name, "arguments": a.arguments}]))
    ids = [p for p, v in args.items() if props.get(p, {}).get("x-ref") and isinstance(v, str)]
    if ids:
        p = rng.choice(ids)
        fake = _fake_id(args[p], blob, rng)
        if fake:
            out.append(("unseen_id", [{"name": name, "arguments": {**args, p: fake}}]))
    if not with_query:
        return out

    app = app_of(name)
    siblings = [n for n, t in catalog.items() if n != name and app_of(n) == app and t.get("kind") == spec.get("kind")]
    if siblings and name != ASK_TOOL:
        out.append(("swap_tool", [{"name": rng.choice(siblings), "arguments": copy.deepcopy(args)}]))
    vals = [(p, _changed(v, props.get(p, {}), rng)) for p, v in args.items() if not props.get(p, {}).get("x-ref")]
    vals = [(p, nv) for p, nv in vals if nv is not None and nv != args[p]]
    if vals and name != ASK_TOOL:
        p, nv = rng.choice(vals)
        out.append(("change_value", [{"name": name, "arguments": {**args, p: nv}}]))
    optional = [p for p in args if p not in required]
    if optional and name != ASK_TOOL:
        p = rng.choice(optional)
        out.append(("drop_arg", [{"name": name, "arguments": {q: v for q, v in args.items() if q != p}}]))
    if args and name != ASK_TOOL:
        out.append(("ask_instead", [{"name": ASK_TOOL, "arguments": {"tool_name": name, "params": [rng.choice(list(args))]}}]))
    return out


# ---------------------------------------------------------------- sinh mẫu

def make_examples(item, results, catalog, rng, with_query=False, neg_per_pos=1, split="train"):
    """results: kết quả ToolEnv của từng gold call (cùng thứ tự). Trả list dict mẫu."""
    gold = item["label"]["tool_calls"]
    hist = actions_from_item(item)
    query = item.get("audio_text", "") if with_query else None
    out, done = [], []

    def row(step, kind, label, actions, plan):
        return {"id": item["id"], "split": split, "app": app_of(plan[0]["name"]), "step": step,
                "kind": kind, "label": label, "text": render(actions, plan, query)}

    for k, g in enumerate(gold):
        actions = hist + done
        out.append(row(k, "dung", 1, actions, [g]))
        negs = corruptions(item, gold, k, actions, catalog, rng, with_query)
        for kind, plan in rng.sample(negs, min(neg_per_pos, len(negs))):
            out.append(row(k, kind, 0, actions, plan))
        done.append(Action(g["name"], g.get("arguments") or {}, results[k] if k < len(results) else None))
    if gold and gold[-1]["name"] == ASK_TOOL and len(gold) > 1:  # call sau khi đã hỏi
        other = next(g for g in gold if g["name"] != ASK_TOOL)
        out.append(row(len(gold), "after_ask", 0, hist + done, [other]))
    return out


def examples_from_pred(item, pred, results, catalog, matches, with_query=False, split="train"):
    """Lỗi thật của Planner: mỗi call dự đoán, state = history + các call dự đoán trước nó."""
    gold = list(item["label"]["tool_calls"])
    hist = actions_from_item(item)
    query = item.get("audio_text", "") if with_query else None
    out, done = [], []
    for i, c in enumerate(pred):
        j = next((j for j, g in enumerate(gold) if matches(c, g, catalog)), None)
        if j is not None:
            gold.pop(j)
        out.append({"id": item["id"], "split": split, "app": app_of(c["name"]), "step": i,
                    "kind": "that_dung" if j is not None else "that_sai", "label": int(j is not None),
                    "text": render(hist + done, [c], query)})
        done.append(Action(c["name"], c.get("arguments") or {}, results[i] if i < len(results) else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="thư mục chứa các split của BTC")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev-ids", default=str(REPO / "eval/splits/dev_ids.txt"))
    ap.add_argument("--with-query", action="store_true", help="kèm lời user và sinh lỗi cần query")
    ap.add_argument("--neg-per-pos", type=int, default=1, help="số mẫu SAI tối đa mỗi bước gold")
    ap.add_argument("--pred", help="submission jsonl của Planner trên train để thêm lỗi thật (bỏ mẫu dev)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.root)
    sys.path.insert(0, str(root / a.split))
    from zaic_tools import ToolEnv
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    dev = set(Path(a.dev_ids).read_text().split())
    rng = random.Random(a.seed)
    rows = []
    for item in items:
        env = ToolEnv(a.split, item["id"], root=str(root))
        results = [env.call(g["name"], g.get("arguments") or {}) for g in item["label"]["tool_calls"]]
        rows += make_examples(item, results, catalog, rng, a.with_query, a.neg_per_pos,
                              "dev" if item["id"] in dev else "train")
    if a.pred:
        from eval.scorer import call_matches
        by_id = {i["id"]: i for i in items}
        for l in open(a.pred, encoding="utf-8"):
            p = json.loads(l) if l.strip() else None
            if not p or p["id"] in dev or p["id"] not in by_id:  # lỗi thật trên dev sẽ rò rỉ vào đánh giá
                continue
            env = ToolEnv(a.split, p["id"], root=str(root))
            results = [env.call(c["name"], c.get("arguments") or {}) for c in p["tool_calls"]]
            rows += examples_from_pred(by_id[p["id"]], p["tool_calls"], results, catalog, call_matches, a.with_query)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} mẫu -> {out}")
    for sp in ("train", "dev"):
        c = Counter((r["kind"]) for r in rows if r["split"] == sp)
        print(f"  {sp}: {sum(c.values())} | ĐÚNG {sum(r['label'] for r in rows if r['split'] == sp)} | {dict(c)}")


if __name__ == "__main__":
    main()
