"""Smoke-test Laya-multilingual trên tiếng Việt (mục 5.3a, PROJECT_OVERVIEW.md).

Trả lời 3 câu hỏi trước khi đầu tư fine-tune:
  A. Laya có hiểu tiếng Việt không? Phân loại miền (choice, 5 lựa chọn) trên cùng một câu
     viết 4 kiểu: có dấu, không dấu, phương ngữ, tiếng Anh (đối chứng).
  B. Laya có bắt được chi tiết không? Câu hỏi có/không (noul) về số tiền, giờ, phủ định,
     cả bản tiếng Việt và tiếng Anh.
  C. Zero-shot trên đúng bài toán Verifier: action đã lưu + plan mới -> nhất quán hay không.
     Model card nói bản gốc gần như đoán bừa khi chưa fine-tune, nên phần này chỉ là mốc
     trước fine-tune, không dùng để kết luận bỏ Laya.
Ngoài ra đo độ trễ mỗi lượt predict và đếm tham số.

Cài đặt:  pip install laya torch
Chạy:     python lifemate/verifier/smoke_test_laya.py --out runs/laya_smoke.json
          (thêm --device cuda trên Kaggle/A30; mặc định tự chọn cuda nếu có)

API theo model card convaiinnovations/laya-multilingual (gói laya 0.4.27):
  agent = laya.load(repo); agent.predict(state, questions)["answers"][qid] -> {"choice"|"noul", ...}
Laya ship chưa hiệu chỉnh nhiệt độ và card ghi "noul can under-report true", nên phần có/không
chấm bằng AUC (không phụ thuộc ngưỡng) và độ chính xác ở ngưỡng tốt nhất, không cố định 0,5.
"""

import argparse
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

MODEL = "convaiinnovations/laya-multilingual"
HF_PARAMS = 321_908_998  # số tham số theo Hub API, dùng khi không đếm được từ object

# ---------------------------------------------------------------- A. phân loại miền
DOMAINS = {
    "ngan_hang": "chuyển tiền, thanh toán hoá đơn, xem số dư tài khoản",
    "goi_xe": "đặt xe, gọi taxi hoặc xe máy để di chuyển",
    "do_an": "đặt đồ ăn, gọi món, giao đồ ăn",
    "nha_thong_minh": "bật tắt đèn, điều hoà, quạt, thiết bị trong nhà",
    "email": "soạn, gửi, đọc hoặc trả lời email",
}
DOMAINS_EN = {
    "ngan_hang": "money transfers, bill payments, account balance",
    "goi_xe": "booking a ride, taxi or motorbike",
    "do_an": "ordering food, food delivery",
    "nha_thong_minh": "lights, air conditioner, fans, home devices",
    "email": "writing, sending, reading or replying to email",
}
# (nhãn, có dấu, phương ngữ, tiếng Anh). Bản không dấu sinh tự động.
DOMAIN_ITEMS = [
    ("ngan_hang", "Chuyển cho Lan hai trăm nghìn tiền ăn trưa nhé",
     "Chuyển cho con Lan hai trăm ngàn tiền cơm trưa dùm tui nha", "Send Lan two hundred thousand for lunch"),
    ("ngan_hang", "Tài khoản của tôi còn bao nhiêu tiền",
     "Tài khoản của tui còn mô nhiêu tiền rứa", "How much money is left in my account"),
    ("goi_xe", "Đặt cho tôi một xe ra sân bay Tân Sơn Nhất lúc sáu giờ sáng",
     "Kêu giùm tui chiếc xe ra sân bay Tân Sơn Nhất lúc sáu giờ sáng nghen", "Book me a ride to Tan Son Nhat airport at six a.m."),
    ("goi_xe", "Gọi xe máy đưa tôi về nhà",
     "Gọi xe ôm chở tui về nhà hỉ", "Get a motorbike to take me home"),
    ("do_an", "Đặt hai bát phở bò giao về công ty",
     "Đặt hai tô phở bò giao về công ty giùm nha", "Order two bowls of beef pho delivered to the office"),
    ("do_an", "Gọi một ly trà sữa trân châu ít đường",
     "Gọi cho tui một ly trà sữa trân châu ít đường nghe", "Order a bubble milk tea with less sugar"),
    ("nha_thong_minh", "Tắt hết đèn phòng khách đi",
     "Tắt hết đèn phòng khách giùm cái", "Turn off all the living room lights"),
    ("nha_thong_minh", "Bật điều hoà phòng ngủ lên hai mươi sáu độ",
     "Mở máy lạnh phòng ngủ lên hai mươi sáu độ hỉ", "Set the bedroom air conditioner to twenty-six degrees"),
    ("email", "Gửi email cho anh Minh báo là chiều nay tôi nghỉ họp",
     "Gởi mail cho anh Minh nói chiều nay tui nghỉ họp nha", "Email Minh that I am skipping the meeting this afternoon"),
    ("email", "Đọc cho tôi email mới nhất trong hộp thư",
     "Đọc cho tui cái email mới nhất trong hộp thư coi", "Read me the newest email in my inbox"),
]

# ---------------------------------------------------------------- B. có/không về chi tiết
# (câu user vi, câu hỏi vi, câu user en, câu hỏi en, đáp án)
YESNO_ITEMS = [
    ("Chuyển cho Lan hai trăm nghìn rồi đặt xe ra sân bay", "Người dùng có yêu cầu đặt xe không?",
     "Send Lan two hundred thousand then book a ride to the airport", "Does the user ask to book a ride?", 1),
    ("Chuyển cho Lan hai trăm nghìn rồi đặt xe ra sân bay", "Người dùng có yêu cầu đặt đồ ăn không?",
     "Send Lan two hundred thousand then book a ride to the airport", "Does the user ask to order food?", 0),
    ("Chuyển cho Lan hai trăm nghìn rồi đặt xe ra sân bay", "Số tiền cần chuyển là hai trăm nghìn phải không?",
     "Send Lan two hundred thousand then book a ride to the airport", "Is the amount to send two hundred thousand?", 1),
    ("Chuyển cho Lan hai trăm nghìn rồi đặt xe ra sân bay", "Số tiền cần chuyển là năm trăm nghìn phải không?",
     "Send Lan two hundred thousand then book a ride to the airport", "Is the amount to send five hundred thousand?", 0),
    ("Tắt đèn phòng khách nhưng để đèn phòng ngủ sáng", "Người dùng muốn tắt đèn phòng khách phải không?",
     "Turn off the living room light but keep the bedroom light on", "Does the user want the living room light off?", 1),
    ("Tắt đèn phòng khách nhưng để đèn phòng ngủ sáng", "Người dùng muốn tắt đèn phòng ngủ phải không?",
     "Turn off the living room light but keep the bedroom light on", "Does the user want the bedroom light off?", 0),
    ("Đừng gửi email cho anh Minh nữa, để mai tính", "Người dùng muốn hoãn việc gửi email phải không?",
     "Don't send the email to Minh, let's deal with it tomorrow", "Does the user want to postpone sending the email?", 1),
    ("Đừng gửi email cho anh Minh nữa, để mai tính", "Người dùng muốn gửi email ngay bây giờ phải không?",
     "Don't send the email to Minh, let's deal with it tomorrow", "Does the user want to send the email right now?", 0),
    ("Đặt hai bát phở, một bát không hành", "Người dùng đặt hai bát phở phải không?",
     "Order two bowls of pho, one without onions", "Does the user order two bowls of pho?", 1),
    ("Đặt hai bát phở, một bát không hành", "Người dùng đặt ba bát phở phải không?",
     "Order two bowls of pho, one without onions", "Does the user order three bowls of pho?", 0),
    ("Hẹn taxi lúc bảy giờ tối mai", "Chuyến xe được hẹn vào buổi tối phải không?",
     "Schedule a taxi at seven tomorrow evening", "Is the ride scheduled in the evening?", 1),
    ("Hẹn taxi lúc bảy giờ tối mai", "Chuyến xe được hẹn vào buổi sáng phải không?",
     "Schedule a taxi at seven tomorrow evening", "Is the ride scheduled in the morning?", 0),
]

# ---------------------------------------------------------------- C. đúng bài toán Verifier
VERIFY_Q = ("Kế hoạch mới có nhất quán với các action đã thực hiện không "
            "(không lặp lại việc ghi đã làm, không dùng ID chưa xuất hiện, không mâu thuẫn)?")
# (action đã thực hiện, [(plan, nhãn 1=ĐÚNG / 0=SAI, kiểu lỗi)])
VERIFY_SCENARIOS = [
    (['bank.search_contact(name="Lan") -> [{"id": "C12", "name": "Lan"}]'],
     [('bank.transfer(contact_id="C12", amount=200000)', 1, "đúng"),
      ('bank.transfer(contact_id="C99", amount=200000)', 0, "ID chưa xuất hiện")]),
    (['bank.search_contact(name="Lan") -> [{"id": "C12", "name": "Lan"}]',
      'bank.transfer(contact_id="C12", amount=200000) -> {"ok": true, "tx": "T7"}'],
     [('ride.book(destination="sân bay Tân Sơn Nhất")', 1, "đúng"),
      ('bank.transfer(contact_id="C12", amount=200000)', 0, "lặp lại write")]),
    (['home.list_devices() -> [{"id": "D1", "name": "đèn phòng khách", "on": true}, {"id": "D2", "name": "đèn phòng ngủ", "on": true}]',
      'home.turn_off(device_id="D1") -> {"ok": true}'],
     [('home.turn_off(device_id="D2")', 1, "đúng"),
      ('home.turn_off(device_id="D1")', 0, "lặp lại write"),
      ('home.turn_off(device_id="D7")', 0, "ID chưa xuất hiện")]),
    (['food.search_restaurant(keyword="phở") -> [{"id": "R3", "menu": [{"id": "M1", "name": "phở bò", "price": 50000}]}]',
      'food.add_to_cart(restaurant_id="R3", item_id="M1", qty=2) -> {"ok": true, "cart": "K1"}'],
     [('food.checkout(cart_id="K1")', 1, "đúng"),
      ('food.add_to_cart(restaurant_id="R3", item_id="M1", qty=2)', 0, "lặp lại write"),
      ('food.add_to_cart(restaurant_id="R8", item_id="M1", qty=2)', 0, "ID chưa xuất hiện")]),
    (['mail.search_contact(name="Minh") -> [{"email": "minh.nguyen@x.vn"}, {"email": "minh.tran@x.vn"}]'],
     [('sys.ask_missing_info(tool_name="mail.send", params=["to"])', 1, "đúng"),
      ('mail.send(to="minh.le@x.vn", body="Chiều nay tôi nghỉ họp")', 0, "ID chưa xuất hiện")]),
    (['ride.estimate(destination="sân bay") -> {"quote_id": "Q5", "price": 180000}',
      'ride.book(quote_id="Q5") -> {"ok": true, "booking_id": "B1"}'],
     [('ride.get_status(booking_id="B1")', 1, "đúng"),
      ('ride.book(quote_id="Q5")', 0, "lặp lại write"),
      ('ride.cancel(booking_id="B9")', 0, "ID chưa xuất hiện")]),
]


def bo_dau(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def verify_state(done, plan):
    lines = ["Các action đã thực hiện:"] + [f"{i}. {a}" for i, a in enumerate(done, 1)]
    return "\n".join(lines + ["Kế hoạch mới:", f"1. {plan}"])


def auc(scores, labels):
    """Xác suất một mẫu dương ngẫu nhiên có điểm cao hơn một mẫu âm ngẫu nhiên (hoà tính 0,5)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def best_threshold_acc(scores, labels):
    cands = sorted(set(scores)) + [1.01]
    return max((sum((s >= t) == bool(y) for s, y in zip(scores, labels)) / len(labels), t) for t in cands)


class Timed:
    def __init__(self, agent, max_len):
        self.agent, self.max_len, self.ms = agent, max_len, []

    def predict(self, state, questions):
        kw = {"max_len": self.max_len} if self.max_len else {}
        t = time.perf_counter()
        out = self.agent.predict(state, questions, **kw)
        self.ms.append((time.perf_counter() - t) * 1000)
        return out["answers"]


def noul(ans):
    v = ans.get("noul")
    if v is None:  # phòng khi bản laya khác trả về probabilities thay vì noul
        v = (ans.get("probabilities") or {}).get("yes", float("nan"))
    return float(v)


def count_params(agent):
    try:
        import torch
    except ImportError:
        return None
    seen, total = set(), 0
    objs = [agent] + list(vars(agent).values())
    for o in objs:
        if isinstance(o, torch.nn.Module):
            for p in o.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total or None


def run_domain(m):
    q_vi = {"mien": {"type": "choice", "instructions": "Yêu cầu này thuộc ứng dụng nào?", "criteria": DOMAINS}}
    q_en = {"mien": {"type": "choice", "instructions": "Which app does this request belong to?", "criteria": DOMAINS_EN}}
    variants = {"co_dau": 1, "khong_dau": 1, "phuong_ngu": 2, "tieng_anh": 3}
    rows, acc = [], {}
    for name, idx in variants.items():
        hit = 0
        for item in DOMAIN_ITEMS:
            text = bo_dau(item[1]) if name == "khong_dau" else item[idx]
            a = m.predict(text, q_en if name == "tieng_anh" else q_vi)["mien"]
            ok = a.get("choice") == item[0]
            hit += ok
            rows.append({"bien_the": name, "text": text, "gold": item[0], "pred": a.get("choice"),
                         "prob_gold": (a.get("probabilities") or {}).get(item[0]), "dung": ok})
        acc[name] = hit / len(DOMAIN_ITEMS)
    return acc, rows


def run_yesno(m):
    res, rows = {}, []
    for lang, (ti, qi) in {"vi": (0, 1), "en": (2, 3)}.items():
        scores, labels = [], []
        for item in YESNO_ITEMS:
            p = noul(m.predict(item[ti], {"q": {"type": "noul", "instructions": item[qi]}})["q"])
            scores.append(p)
            labels.append(item[4])
            rows.append({"lang": lang, "text": item[ti], "question": item[qi], "gold": item[4], "p_yes": p})
        acc, thr = best_threshold_acc(scores, labels)
        res[lang] = {"auc": auc(scores, labels), "acc_best": acc, "nguong_best": thr,
                     "p_yes_tb_dung": statistics.mean(s for s, y in zip(scores, labels) if y),
                     "p_yes_tb_sai": statistics.mean(s for s, y in zip(scores, labels) if not y)}
    return res, rows


def run_verify(m):
    scores, labels, rows = [], [], []
    for done, plans in VERIFY_SCENARIOS:
        for plan, y, kind in plans:
            p = noul(m.predict(verify_state(done, plan), {"q": {"type": "noul", "instructions": VERIFY_Q}})["q"])
            scores.append(p)
            labels.append(y)
            rows.append({"plan": plan, "gold": y, "kieu": kind, "p_dung": p})
    acc, thr = best_threshold_acc(scores, labels)
    by_kind = {k: statistics.mean(r["p_dung"] for r in rows if r["kieu"] == k) for k in {r["kieu"] for r in rows}}
    return {"auc": auc(scores, labels), "acc_best": acc, "nguong_best": thr, "p_dung_tb_theo_kieu": by_kind}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL, help="HF repo hoặc thư mục local của checkpoint Laya")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    ap.add_argument("--max-len", type=int, default=0, help="ghi đè max_len của predict (0 = mặc định của model)")
    ap.add_argument("--out", help="ghi kết quả chi tiết ra file JSON (vd runs/laya_smoke.json, runs/ đã gitignore)")
    a = ap.parse_args()

    try:
        import laya
    except ImportError:
        sys.exit("Chưa cài laya: pip install laya torch")
    device = a.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"laya {getattr(laya, '__version__', '?')} | model {a.model} | device {device}")
    t = time.perf_counter()
    agent = laya.load(a.model, device=device)
    load_s = time.perf_counter() - t
    n_params = count_params(agent)
    print(f"Tải model: {load_s:.1f}s | tham số: {n_params or HF_PARAMS:,}"
          + ("" if n_params else " (theo Hub API, không đếm được từ object)"))

    m = Timed(agent, a.max_len)
    m.predict("Khởi động", {"q": {"type": "noul", "instructions": "Đây là câu tiếng Việt?"}})  # warmup
    m.ms.clear()

    dom_acc, dom_rows = run_domain(m)
    yn, yn_rows = run_yesno(m)
    ver, ver_rows = run_verify(m)
    ms = sorted(m.ms)
    lat = {"n": len(ms), "p50_ms": statistics.median(ms), "p95_ms": ms[int(0.95 * (len(ms) - 1))], "max_ms": ms[-1]}

    chance = 1 / len(DOMAINS)
    print(f"\nA. Phân loại miền (5 lựa chọn, đoán bừa = {chance:.2f}, {len(DOMAIN_ITEMS)} câu mỗi biến thể)")
    for k, v in dom_acc.items():
        print(f"   {k:<11} acc = {v:.2f}")
    for r in dom_rows:
        if not r["dung"]:
            print(f"     sai [{r['bien_the']}] {r['text'][:60]!r}: {r['pred']} (gold {r['gold']})")

    print(f"\nB. Có/không về chi tiết ({len(YESNO_ITEMS)} câu mỗi ngôn ngữ, AUC đoán bừa = 0.50)")
    for lang, r in yn.items():
        print(f"   {lang}: AUC {r['auc']:.2f} | acc@ngưỡng tốt nhất {r['acc_best']:.2f} (ngưỡng {r['nguong_best']:.3f}) | "
              f"P(yes) TB khi đúng {r['p_yes_tb_dung']:.3f}, khi sai {r['p_yes_tb_sai']:.3f}")

    print(f"\nC. Zero-shot bài Verifier ({len(ver_rows)} plan, chưa fine-tune, chỉ là mốc)")
    print(f"   AUC {ver['auc']:.2f} | acc@ngưỡng tốt nhất {ver['acc_best']:.2f}")
    for k, v in sorted(ver["p_dung_tb_theo_kieu"].items()):
        print(f"   P(nhất quán) TB [{k}] = {v:.3f}")

    print(f"\nĐộ trễ predict 1 câu hỏi ({device}, {lat['n']} lượt): p50 {lat['p50_ms']:.0f} ms, "
          f"p95 {lat['p95_ms']:.0f} ms, max {lat['max_ms']:.0f} ms (ngân sách Verifier < 100 ms/mẫu)")
    if device == "cpu":
        print("   Lưu ý: đo trên CPU, không đại diện cho A30. Chạy lại với --device cuda trên Kaggle.")

    # Ngưỡng gợi ý để báo nhóm, chưa phải ngưỡng đã thống nhất.
    gap = dom_acc["tieng_anh"] - dom_acc["co_dau"]
    hieu_tv = gap <= 0.15 and dom_acc["co_dau"] >= 0.6 and yn["vi"]["auc"] >= 0.7
    print("\nKẾT LUẬN GỢI Ý (ngưỡng: acc có dấu ≥ 0.60, kém tiếng Anh ≤ 0.15, AUC có/không vi ≥ 0.70):")
    print(f"   Tiếng Việt: {'DÙNG ĐƯỢC, đi tiếp fine-tune' if hieu_tv else 'YẾU, cân nhắc phương án dự phòng 5.3d'}"
          f" (có dấu {dom_acc['co_dau']:.2f} vs Anh {dom_acc['tieng_anh']:.2f}, AUC vi {yn['vi']['auc']:.2f})")
    print(f"   Không dấu {dom_acc['khong_dau']:.2f}, phương ngữ {dom_acc['phuong_ngu']:.2f}: "
          "ASR đã trả có dấu nên chỉ để tham khảo độ bền.")

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "model": a.model, "laya_version": getattr(laya, "__version__", None), "device": device,
            "load_s": load_s, "params": n_params or HF_PARAMS, "latency": lat,
            "domain_acc": dom_acc, "yesno": yn, "verify": ver,
            "rows": {"domain": dom_rows, "yesno": yn_rows, "verify": ver_rows},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi chi tiết: {out}")


if __name__ == "__main__":
    main()
