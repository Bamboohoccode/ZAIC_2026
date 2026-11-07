"""Fine-tune bộ phân loại ĐÚNG/SAI cho Verifier, đổi được model nền (bước 4, task_flow/ha.md).

Input: jsonl từ data.py ({split, kind, label, text}); train trên split=train, chọn epoch tốt nhất theo AUC
trên split=dev (app lạ). Ra: checkpoint HF + metrics.json (AUC, ngưỡng, P(ĐÚNG) theo kiểu lỗi, độ trễ, tham số).

Model nền (--backbone, hoặc HF id bất kỳ):
  mmbert-base    jhu-clsp/mmBERT-base         MIT, ~307M, chính là nền của Laya-multilingual
  mmbert-small   jhu-clsp/mmBERT-small        MIT, ~140M
  xlmr-base      FacebookAI/xlm-roberta-base  MIT, 279M, context 512
  mdeberta-base  microsoft/mdeberta-v3-base   MIT, ~278M, context 512
Laya dùng lớp riêng (LayaTypedDecisions), fine-tune theo notebook của gói laya, không qua script này.

Chạy trên Kaggle GPU:
  pip install -U transformers
  python lifemate/verifier/train.py --data runs/verifier_data.jsonl --backbone mmbert-base --out runs/ver_mmbert
"""
import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

BACKBONES = {
    "mmbert-base": "jhu-clsp/mmBERT-base",
    "mmbert-small": "jhu-clsp/mmBERT-small",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "mdeberta-base": "microsoft/mdeberta-v3-base",
}
PLAN_MARK = "Kế hoạch mới:"
QUERY_MARK = "Yêu cầu mới nhất của người dùng:"


def split_text(text):
    """(phần luôn giữ: lời user + plan, phần được cắt: action đã lưu). Action cũ nhất bị cắt trước."""
    head, _, plan = text.partition(PLAN_MARK)
    query = ""
    if head.startswith(QUERY_MARK):
        query, _, head = head.partition("\n")
    return (query + "\n" + PLAN_MARK + plan).strip(), head.strip()


def auc(scores, labels):
    pos = sorted(s for s, y in zip(scores, labels) if y == 1)
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    import bisect
    total = sum(bisect.bisect_left(pos, n) for n in neg)  # số cặp mà điểm dương < điểm âm
    ties = sum(bisect.bisect_right(pos, n) - bisect.bisect_left(pos, n) for n in neg)
    return 1 - (total + 0.5 * ties) / (len(pos) * len(neg))


def best_threshold(scores, labels):
    best = (0.0, 0.5)
    for t in sorted(set(scores)):
        acc = sum((s >= t) == bool(y) for s, y in zip(scores, labels)) / len(labels)
        best = max(best, (acc, t))
    return best


def evaluate(rows, probs):
    labels = [r["label"] for r in rows]
    acc, thr = best_threshold(probs, labels)
    by_kind = defaultdict(list)
    for r, p in zip(rows, probs):
        by_kind[r["kind"]].append(p)
    return {"auc": auc(probs, labels), "acc_best": acc, "nguong": thr, "n": len(rows),
            "p_dung_theo_kieu": {k: round(statistics.mean(v), 4) for k, v in sorted(by_kind.items())},
            "bat_sai_theo_kieu": {k: round(sum(p < thr for p in v) / len(v), 4)
                                  for k, v in sorted(by_kind.items()) if not k.endswith("dung")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", default="mmbert-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N mẫu train (debug)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    name = BACKBONES.get(a.backbone, a.backbone)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if a.limit:
        random.shuffle(train)
        train = train[:a.limit]
    print(f"{name} | {device} | train {len(train)} (ĐÚNG {sum(r['label'] for r in train)}) | dev {len(dev)}")

    tok = AutoTokenizer.from_pretrained(name)
    tok.truncation_side = "left"  # cắt action cũ nhất của phần thứ hai
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
    max_len = min(a.max_len, getattr(tok, "model_max_length", a.max_len) or a.max_len)
    n_params = sum(p.numel() for p in model.parameters())

    def encode(batch):
        heads, tails = zip(*(split_text(r["text"]) for r in batch))
        enc = tok(list(heads), list(tails), truncation="only_second", max_length=max_len,
                  padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def predict(data, bs=32):
        model.eval()
        out = []
        for i in range(0, len(data), bs):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = model(**encode(data[i:i + bs])).logits.float()
            out += torch.softmax(logits, -1)[:, 1].tolist()
        return out

    n_pos = sum(r["label"] for r in train)
    w = torch.tensor([len(train) / (2 * max(len(train) - n_pos, 1)), len(train) / (2 * max(n_pos, 1))], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    warm = int(a.warmup * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (steps - s) / max(steps - warm, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, history = None, []
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        t0, losses = time.time(), []
        for i in range(0, len(train), a.bs):
            batch = train[i:i + a.bs]
            y = torch.tensor([r["label"] for r in batch], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                loss = loss_fn(model(**encode(batch)).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        m = evaluate(dev, predict(dev)) if dev else {"auc": float("nan")}
        m.update(epoch=ep + 1, loss=round(statistics.mean(losses), 4), sec=round(time.time() - t0, 1))
        history.append(m)
        print(f"epoch {ep + 1}: loss {m['loss']} | dev AUC {m['auc']:.4f} | acc {m.get('acc_best', float('nan')):.4f} | {m['sec']}s")
        if best is None or not (m["auc"] <= best["auc"]):  # nan (không có dev) thì lưu epoch cuối
            best = m
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    ms = []
    for r in (dev or train)[:50]:  # độ trễ 1 mẫu, batch 1 như lúc suy luận
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        predict([r], bs=1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
    ms = sorted(ms[3:] or ms)  # bỏ vài lượt khởi động
    metrics = {"backbone": name, "device": device, "params": n_params, "max_len": max_len, "best": best,
               "history": history, "latency_ms": {"p50": statistics.median(ms), "p95": ms[int(0.95 * (len(ms) - 1))]},
               "args": vars(a)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tốt nhất: epoch {best.get('epoch')} AUC {best['auc']:.4f} | {n_params:,} tham số | "
          f"độ trễ p50 {metrics['latency_ms']['p50']:.1f} ms -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
