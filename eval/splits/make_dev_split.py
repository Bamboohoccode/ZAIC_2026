"""Tạo danh sách split dev (eval/splits/dev_ids.txt) cho LifeMate.

Nguyên tắc (theo eval/splits/README.md và docs/PROJECT_OVERVIEW.md mục 5.4b):
- Giữ ~20% app lạ (9/46 app trong train: bike_rental, calculator, exchange_rate, fitness, lottery, package_tracking, fuel_price, gold_price, tv)
  để đo khả năng tổng quát sang app chưa từng huấn luyện (mô phỏng public test có 11 app mới).
- Bổ sung mẫu stratified từ các app còn lại để giữ nguyên tỉ lệ:
  Act: ~78.7%, Ask: ~14.0%, Empty: ~7.3%.
- Tổng số mẫu dev: 300 / 1545 (19.4%).
- Seed cố định: 42 để đảm bảo tính tái lập (reproducibility).
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def generate_dev_split(root: Path = Path("data"), out_file: Path = Path("eval/splits/dev_ids.txt"), seed: int = 42):
    random.seed(seed)
    train_path = root / "train" / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {train_path}")

    train = [json.loads(l) for l in open(train_path, encoding="utf-8")]

    # Phân loại intent và trích xuất app được gọi
    sample_info = {}
    app_to_samples = defaultdict(set)
    for item in train:
        sid = item["id"]
        calls = item["label"]["tool_calls"]
        if not calls:
            intent = "empty"
            apps = set()
        elif any(c["name"] == "sys.ask_missing_info" for c in calls):
            intent = "ask"
            apps = set()
            for c in calls:
                t = c["name"]
                if "." in t:
                    apps.add(t.split(".")[0])
                if c["name"] == "sys.ask_missing_info":
                    tn = c["arguments"].get("tool_name", "")
                    if "." in tn:
                        apps.add(tn.split(".")[0])
            apps.discard("sys")
        else:
            intent = "act"
            apps = set(c["name"].split(".")[0] for c in calls if "." in c["name"])
            apps.discard("sys")

        sample_info[sid] = {"intent": intent, "apps": apps}
        for a in apps:
            app_to_samples[a].add(sid)

    # 9 app giữ lại (~20% của 46 app)
    held_out_apps = [
        "bike_rental", "calculator", "exchange_rate", "fitness",
        "lottery", "package_tracking", "fuel_price", "gold_price", "tv"
    ]

    held_out_samples = set()
    for a in held_out_apps:
        held_out_samples |= app_to_samples[a]

    cnt = Counter(sample_info[s]["intent"] for s in held_out_samples)

    # Lấy thêm từ phần còn lại để đạt 300 mẫu và giữ tỉ lệ act/ask/empty
    remaining = [sid for sid in sample_info if sid not in held_out_samples]
    by_intent = defaultdict(list)
    for sid in remaining:
        by_intent[sample_info[sid]["intent"]].append(sid)

    # Tỉ lệ mục tiêu trong 300 mẫu: 236 act, 42 ask, 22 empty
    target_act = int(300 * (1217 / 1545))
    target_ask = int(300 * (217 / 1545))
    target_empty = 300 - target_act - target_ask

    need_act = max(0, target_act - cnt["act"])
    need_ask = max(0, target_ask - cnt["ask"])
    need_empty = max(0, target_empty - cnt["empty"])

    random.shuffle(by_intent["act"])
    random.shuffle(by_intent["ask"])
    random.shuffle(by_intent["empty"])

    selected_dev = set(held_out_samples)
    selected_dev.update(by_intent["act"][:need_act])
    selected_dev.update(by_intent["ask"][:need_ask])
    selected_dev.update(by_intent["empty"][:need_empty])

    dev_list = sorted(selected_dev)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for sid in dev_list:
            f.write(f"{sid}\n")

    print(f"Đã lưu {len(dev_list)} dev IDs vào {out_file}")
    dev_cnt = Counter(sample_info[s]["intent"] for s in dev_list)
    print(f"Phân bố nhãn dev: {dict(dev_cnt)}")
    print(f"Held-out apps ({len(held_out_apps)}): {held_out_apps}")


if __name__ == "__main__":
    generate_dev_split()
