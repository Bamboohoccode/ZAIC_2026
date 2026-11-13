"""Tập con cố định cho WSP (seed 42), không đụng test BTC:
- `--pool dev` (mặc định): 50 mẫu từ eval/splits/dev_ids.txt, chỉ để ĐO mỗi vòng sửa.
- `--pool train`: mẫu train KHÔNG thuộc dev, làm dữ liệu practice để TÌM hướng (directions.py), tách khỏi tập đo.

Phân tầng theo (nhãn act/ask/empty) x (có dùng app chỉ có trong dev hay không), giữ tỉ lệ của dev. Seed 42.
Đầu ra: optimize/wsp/dev_subset50_20261107.txt, optimize/wsp/practice100_20261107.txt
Chạy: python optimize/wsp/make_subset.py
      python optimize/wsp/make_subset.py --pool train --n 100 --out optimize/wsp/practice100_20261107.txt
"""
import argparse, json, random, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEV_APPS = {"bike_rental", "calculator", "exchange_rate", "fitness", "lottery", "package_tracking", "fuel_price", "gold_price", "tv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw" / "train+public_test"))
    ap.add_argument("--pool", choices=["dev", "train"], default="dev")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(HERE / "dev_subset50_20261107.txt"))
    a = ap.parse_args()
    dev_ids = [l.strip() for l in open(REPO / "eval" / "splits" / "dev_ids.txt", encoding="utf-8") if l.strip()]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(Path(a.root) / "train" / "train.jsonl", encoding="utf-8")}
    dev = dev_ids if a.pool == "dev" else sorted(set(rows) - set(dev_ids))
    strata = collections.defaultdict(list)
    for sid in dev:
        calls = rows[sid]["label"]["tool_calls"]
        kind = "empty" if not calls else "ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "act"
        held = any(c["name"].split(".")[0] in DEV_APPS for c in calls)
        strata[(kind, held)].append(sid)
    rng = random.Random(42)
    quota = {k: round(a.n * len(v) / len(dev)) for k, v in strata.items()}
    while sum(quota.values()) != a.n:  # sửa sai số làm tròn vào tầng lớn nhất
        big = max(quota, key=lambda k: len(strata[k]))
        quota[big] += 1 if sum(quota.values()) < a.n else -1
    pick = []
    for k in sorted(strata):
        pick += rng.sample(sorted(strata[k]), min(quota[k], len(strata[k])))
    pick.sort()
    Path(a.out).write_text("\n".join(pick) + "\n", encoding="utf-8")
    print("strata (kind, held_out_app): dev ->", {str(k): len(v) for k, v in sorted(strata.items())})
    print("quota ->", {str(k): v for k, v in sorted(quota.items())}, "| total", len(pick), "->", a.out)


if __name__ == "__main__":
    main()
