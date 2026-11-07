"""Sinh bộ 3 nhãn từ train (có nhãn sẵn, không đụng test, không gán nhãn tay): full | missing_slot | no_app.

python -m synth.gen_three_labels --root data --split train          # -> data/synth/train_synth.jsonl (data/ đã gitignore)
- full:         mẫu act gốc, giữ nguyên.
- no_app:       gỡ app mà đáp án dùng (và các app cùng group trong catalog) khỏi installed_apps -> đáp án [].
- missing_slot: xoá khỏi câu nói giá trị của một tham số bắt buộc của call ghi cuối (chuỗi xuất hiện đúng 1 lần,
                không có trong lịch sử) -> đáp án = các call đọc trước đó + sys.ask_missing_info(tool_name, [param]).
Mỗi mẫu giữ id gốc để ToolEnv dùng đúng môi trường, thêm synth_id, synth, synth_app; audio = None nên chỉ chạy --text.
"""
import argparse
import copy
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, để chạy trực tiếp như tests/test_pipeline.py

from eval.scorer import ASK, task_app, task_kind

norm = lambda s: unicodedata.normalize("NFC", s).lower()


def apps_used(G):
    names = [(c.get("arguments") or {}).get("tool_name", "") if c["name"] == ASK else c["name"] for c in G]
    return {n.split(".")[0] for n in names if n and not n.startswith("sys.")}


def tag(item, label, src):
    item.update(synth=label, synth_id=f"{src['id']}#{label}", synth_app=task_app(src["label"]["tool_calls"]))
    return item


def no_app(item, groups):
    used = apps_used(item["label"]["tool_calls"])
    if not used:
        return None
    used_groups = {groups[u] for u in used if groups.get(u)}
    drop = used | {a for a in item["context"]["installed_apps"] if groups.get(a) in used_groups}
    if drop & {c["name"].split(".")[0] for h in item["history"] for c in h.get("tool_calls") or []}:
        return None  # lịch sử đã dùng app này: gỡ app sẽ mâu thuẫn ngữ cảnh
    new = copy.deepcopy(item)
    new["context"]["installed_apps"] = {a: v for a, v in item["context"]["installed_apps"].items() if a not in drop}
    new["label"], new["audio"] = {"tool_calls": []}, None
    return tag(new, "no_app", item)


def missing_slot(item, catalog):
    G = item["label"]["tool_calls"]
    if task_kind(G) != "act":
        return None
    *before, last = G
    spec = catalog.get(last["name"], {})
    if spec.get("kind") != "write" or any(catalog.get(c["name"], {}).get("kind") != "read" for c in before):
        return None  # chỉ nhận chuỗi đọc rồi một call ghi cuối, để đáp án mới không đổi state
    text, low = item["audio_text"], norm(item["audio_text"])
    hist = norm(" ".join(h.get("audio_text") or h.get("text") or "" for h in item["history"]))
    used = norm(json.dumps([c["arguments"] for c in before], ensure_ascii=False))
    for p in spec.get("parameters", {}).get("required", []):
        v = (last.get("arguments") or {}).get(p)
        if not isinstance(v, str) or len(v.strip()) < 2:
            continue
        hits = list(re.finditer(rf"(?<!\w){re.escape(norm(v))}(?!\w)", low))
        if len(hits) != 1 or norm(v) in hist or norm(v) in used:
            continue
        i, j = hits[0].span()  # NFC giữ nguyên độ dài với text đã chuẩn NFC
        new = copy.deepcopy(item)
        new["audio_text"] = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)[:i] + " " + unicodedata.normalize("NFC", text)[j:]).strip()
        new["label"] = {"tool_calls": before + [{"name": ASK, "arguments": {"tool_name": last["name"], "params": [p]}}]}
        new["audio"] = None
        return tag(new, "missing_slot", item)
    return None


def generate(items, catalog, groups):
    out = []
    for it in items:
        if task_kind(it["label"]["tool_calls"]) == "act":
            full = copy.deepcopy(it)
            full["audio"] = None
            out.append(tag(full, "full", it))
        out += [s for s in (no_app(it, groups), missing_slot(it, catalog)) if s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", help="mặc định data/synth/<split>_synth.jsonl")
    ap.add_argument("--per-label", type=int, default=0, help="giới hạn số mẫu mỗi nhãn (0 = tất cả)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.root) / a.split
    apps = json.load(open(d / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
    catalog = {t["name"]: t for app in apps for t in app["tools"]}
    groups = {t["name"].split(".")[0]: app.get("group") for app in apps for t in app["tools"]}  # khoá = tiền tố tool = key installed_apps
    items = [json.loads(l) for l in open(d / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    out = generate(items, catalog, groups)
    if a.per_label:
        rng, by = random.Random(a.seed), {}
        for s in out:
            by.setdefault(s["synth"], []).append(s)
        out = [s for v in by.values() for s in rng.sample(v, min(a.per_label, len(v)))]
    path = Path(a.out or Path(a.root) / "synth" / f"{a.split}_synth.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in out)
    print(f"{len(out)} mẫu -> {path}", dict(Counter(s["synth"] for s in out)))


if __name__ == "__main__":
    main()
