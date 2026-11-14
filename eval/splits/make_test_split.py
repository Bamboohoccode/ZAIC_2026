"""Tạo danh sách split test 100 mẫu (eval/splits/test_ids.txt) và DPO train (eval/splits/dpo_train_ids.txt) cho LifeMate.

Nguyên tắc (đồng nhất với eval/splits/make_dev_split.py):
- Loại trừ hoàn toàn 300 mẫu của eval/splits/dev_ids.txt (chứa 9 held-out dev apps).
- Giữ 100% mẫu của 3 app lạ chưa từng xuất hiện trong tập train DPO:
  `library` (24), `home_energy` (27), `medication` (27) -> tổng 78 mẫu (68 act, 10 ask).
- Bổ sung 22 mẫu stratified từ phần còn lại để đạt đúng 100 mẫu giữ nguyên tỉ lệ nhãn:
  Act: 79% (79), Ask: 14% (14), Empty: 7% (7).
- Phần còn lại (1,145 mẫu, 34 app) được xuất ra eval/splits/dpo_train_ids.txt,
  đảm bảo 0% rò rỉ với dev (300) và test (100).
- Seed cố định: 42.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


HELD_OUT_TEST_APPS = ("library", "home_energy", "medication")


def extract_intent_and_apps(item: dict) -> tuple[str, set[str]]:
    calls = item["label"]["tool_calls"]
    if not calls:
        return "empty", set()
    intent = "ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "act"
    apps = set()
    for c in calls:
        t = c["name"]
        if "." in t:
            apps.add(t.split(".")[0])
        if t == "sys.ask_missing_info":
            tn = (c.get("arguments") or {}).get("tool_name", "")
            if "." in tn:
                apps.add(tn.split(".")[0])
    apps.discard("sys")
    return intent, apps


def generate_test_and_dpo_splits(
    root: Path = Path("data"),
    dev_file: Path = Path("eval/splits/dev_ids.txt"),
    test_out: Path = Path("eval/splits/test_ids.txt"),
    train_out: Path = Path("eval/splits/dpo_train_ids.txt"),
    seed: int = 42,
):
    rng = random.Random(seed)
    train_path = root / "train" / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {train_path}")
    if not dev_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {dev_file}")

    train = [json.loads(l) for l in open(train_path, encoding="utf-8") if l.strip()]
    dev_ids = {l.strip() for l in open(dev_file, encoding="utf-8") if l.strip()}

    sample_info = {}
    app_to_samples = defaultdict(set)
    for item in train:
        sid = item["id"]
        if sid in dev_ids:
            continue
        intent, apps = extract_intent_and_apps(item)
        sample_info[sid] = {"intent": intent, "apps": apps}
        for a in apps:
            app_to_samples[a].add(sid)

    held_out_samples = set()
    for a in HELD_OUT_TEST_APPS:
        held_out_samples |= app_to_samples[a]

    cnt = Counter(sample_info[s]["intent"] for s in held_out_samples)

    remaining = sorted(sid for sid in sample_info if sid not in held_out_samples)
    by_intent = defaultdict(list)
    for sid in remaining:
        by_intent[sample_info[sid]["intent"]].append(sid)

    # Tỉ lệ mục tiêu trong 100 mẫu: 79 act, 14 ask, 7 empty
    target_act, target_ask, target_empty = 79, 14, 7
    need_act = max(0, target_act - cnt["act"])
    need_ask = max(0, target_ask - cnt["ask"])
    need_empty = max(0, target_empty - cnt["empty"])

    for k in ("act", "ask", "empty"):
        rng.shuffle(by_intent[k])

    selected_test = set(held_out_samples)
    selected_test.update(by_intent["act"][:need_act])
    selected_test.update(by_intent["ask"][:need_ask])
    selected_test.update(by_intent["empty"][:need_empty])

    test_list = sorted(selected_test)
    dpo_train_list = sorted(sid for sid in sample_info if sid not in selected_test)

    # Kiểm tra chéo không rò rỉ
    assert len(dev_ids) == 300, f"Expected 300 dev_ids, got {len(dev_ids)}"
    assert len(test_list) == 100, f"Expected 100 test_ids, got {len(test_list)}"
    assert len(dpo_train_list) == len(train) - len(dev_ids) - len(test_list)
    assert not (set(test_list) & dev_ids)
    assert not (set(dpo_train_list) & dev_ids)
    assert not (set(dpo_train_list) & set(test_list))
    for a in HELD_OUT_TEST_APPS:
        leaked = app_to_samples[a] & set(dpo_train_list)
        assert not leaked, f"App {a} leaked into dpo_train_ids: {leaked}"

    test_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.write_text("\n".join(test_list) + "\n", encoding="utf-8")
    train_out.write_text("\n".join(dpo_train_list) + "\n", encoding="utf-8")

    test_cnt = Counter(sample_info[s]["intent"] for s in test_list)
    train_cnt = Counter(sample_info[s]["intent"] for s in dpo_train_list)
    print(f"Đã lưu {len(test_list)} test IDs vào {test_out}: {dict(test_cnt)}")
    print(f"  Held-out test apps ({len(HELD_OUT_TEST_APPS)}): {list(HELD_OUT_TEST_APPS)} -> {len(held_out_samples)} mẫu")
    print(f"Đã lưu {len(dpo_train_list)} DPO train IDs vào {train_out}: {dict(train_cnt)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--dev-file", type=Path, default=Path("eval/splits/dev_ids.txt"))
    ap.add_argument("--test-out", type=Path, default=Path("eval/splits/test_ids.txt"))
    ap.add_argument("--train-out", type=Path, default=Path("eval/splits/dpo_train_ids.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_test_and_dpo_splits(args.root, args.dev_file, args.test_out, args.train_out, args.seed)
