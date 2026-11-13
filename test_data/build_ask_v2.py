"""Sinh mẫu ask_user v2 theo đúng cách BTC gán nhãn (thay cho bản template v1 bị sai).

Quy tắc rút ra từ 217 mẫu ask của train:
- ask luôn là call cuối và hỏi đúng 1 tham số;
- yêu cầu một ý, thiếu giá trị người dùng phải nói (giờ, số tiền...) -> gold chỉ có [sys.ask_missing_info];
  BTC không tra danh bạ hay số dư trước (vd "chuyển ít tiền qua ví cho Huy" -> chỉ ask amount);
- tham số x-ref (contact_id...) chỉ được hỏi sau khi tra cứu ra nhiều kết quả -> KHÔNG sinh ở đây.

Hai nguồn:
- "btc_rewrite": mẫu act một ý của train (không thuộc dev), câu được viết lại để bỏ giá trị; bối cảnh giữ nguyên mẫu gốc.
- "opus_new": câu mới viết tay, gán vào bối cảnh của một mẫu train có cài app đó.
Không dùng app chỉ có trong dev (fitness, tv, lottery, exchange_rate, ...).

Mỗi dòng được kiểm bằng ToolEnv: tool được cài trong mẫu; thiếu tham số đó thì validate báo đúng tên tham số;
điền đủ thì validate hợp lệ. Đầu ra: data/derived/ask_v2/planner_ask_v2.jsonl (không commit).
"""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "raw", "train+public_test")
OUT = os.path.join(HERE, "..", "data", "derived", "ask_v2", "planner_ask_v2.jsonl")
sys.path.insert(0, os.path.join(ROOT, "train"))
from zaic_tools import ToolEnv  # noqa: E402

DEV_APPS = {"bike_rental", "calculator", "exchange_rate", "fitness", "lottery", "package_tracking", "fuel_price", "gold_price", "tv"}

# (id mẫu gốc, tool, tham số thiếu, câu viết lại)
BTC_REWRITE = [
    ("train_0007", "alarm.create", "time", "Đặt báo thức sáng mai cho em nhé."),
    ("train_0009", "ewallet.topup_mobile", "amount", "Nạp dùm em Linh ít tiền điện thoại cho số không chín năm không năm ba sáu không ba sáu bảy nha."),
    ("train_0027", "reminders.create", "time", "Nhắc dùm chị ra sân bay tối mai nha."),
    ("train_0031", "reminders.create", "time", "Nhớ nhắc mình gọi lại cho khách nhé."),
    ("train_0060", "alarm.create", "time", "Đặt giùm tui cái báo thức nha."),
    ("train_0061", "reminders.create", "time", "Nhắc mình gọi lại cho khách tối mai nhé."),
    ("train_0066", "reminders.create", "time", "Tối mai nhắc mình gọi lại cho khách với nhé."),
    ("train_0168", "reminders.create", "time", "Nhắc mình gọi lại cho khách nha, tối mai đó."),
    ("train_0301", "reminders.create", "time", "Nhắc mình tắt bếp sáng mai nha."),
    ("train_0358", "alarm.create", "time", "Mai dậy sớm, đặt cho mình cái báo thức nhé."),
    ("train_0500", "ewallet.transfer", "amount", "Chuyển cho Hòa ít tiền qua ví giúp anh cái."),
    ("train_0567", "ewallet.transfer", "amount", "Chuyển cho Yến chút tiền qua ví giúp chị nha."),
    ("train_0874", "ewallet.transfer", "amount", "Chuyển cho Huy tiền vé bóng đá qua ví giúp em nha, nội dung ghi là tiền vé bóng đá."),
    ("train_1149", "home_energy.set_alert", "threshold_kwh", "Điện tháng này tăng quá em ạ, đặt lại mức cảnh báo điện cho chị nhé."),
]

# (tool, tham số thiếu, [câu mới])
NEW = [
    ("alarm.create", "time", [
        "Mai đặt giùm anh cái báo thức nha.",
        "Ê, đặt báo thức cho tớ với, mai phải dậy sớm.",
        "Con đặt cho mạ cái báo thức sáng mai nghe.",
        "Đặt báo thức hằng ngày cho chị nhé.",
        "Bạn ơi, cài báo thức giúp mình để mai đi làm khỏi trễ.",
    ]),
    ("reminders.create", "time", [
        "Nhắc anh uống thuốc huyết áp nha.",
        "Nhớ nhắc chị đóng tiền học cho bé nhé.",
        "Nhắc tui tưới cây ngoài ban công với.",
        "Em nhắc anh mang hồ sơ đi nộp nghe.",
        "Nhắc mình gọi điện chúc mừng sinh nhật bà ngoại nhé.",
    ]),
    ("video_meeting.create", "start", [
        "Tạo giúp chị một phòng họp online tên Tổng kết quý ba nhé.",
        "Mở cho anh cái phòng họp video đặt tên Họp phụ huynh lớp Năm A nha.",
        "Em tạo dùm anh phòng họp trực tuyến tên Chốt hợp đồng với đối tác nghe.",
        "Tạo phòng họp online tên Phỏng vấn ứng viên cho tớ với.",
        "Con mở giùm cô cái phòng họp tên Họp gia đình cuối năm nghen.",
    ]),
    ("calendar.create_event", "start", [
        "Thêm vào lịch giúp anh sự kiện Khám răng định kỳ nhé.",
        "Ghi vô lịch cho chị cái hẹn Đi xem nhà với môi giới nha.",
        "Tạo lịch tên Sinh nhật mẹ cho em với.",
        "Em lên lịch Họp lớp cấp ba giùm anh nghe.",
        "Lưu vào lịch giúp tớ buổi Học bơi cho con nhé.",
    ]),
    ("ewallet.transfer", "amount", [
        "Chuyển qua ví cho Lan ít tiền giúp chị.",
        "Gửi Tuấn chút tiền qua ví điện tử nha em.",
        "Con chuyển ví cho mẹ ít tiền đi chợ giùm ba nghe.",
        "Bắn qua ví cho Minh tiền cà phê hôm qua giúp tớ với.",
        "Chuyển tiền ví cho Hương giùm anh, nội dung ghi là tiền sinh nhật.",
    ]),
    ("ewallet.topup_mobile", "amount", [
        "Nạp tiền điện thoại cho số không chín tám bảy sáu năm bốn ba hai một giúp anh.",
        "Nạp card cho số 0912345678 giùm em nha.",
        "Con nạp tiền điện thoại cho số không ba tám tám một hai ba bốn năm sáu giúp mạ nghe.",
        "Nạp tiền vào số 0903 456 789 cho chị với.",
    ]),
    ("ewallet.topup_mobile", "phone", [
        "Nạp năm chục nghìn tiền điện thoại giúp anh nhé.",
        "Nạp giùm em một trăm ngàn tiền điện thoại nha.",
    ]),
    ("home_energy.set_alert", "threshold_kwh", [
        "Đặt cảnh báo điện tháng này cho anh nhé.",
        "Cài mức báo khi nhà xài điện nhiều quá giùm chị.",
        "Em chỉnh lại ngưỡng cảnh báo điện giúp anh nghe.",
        "Bật cảnh báo tiêu thụ điện hằng tháng cho tớ với.",
    ]),
    ("flights.search", "date", [
        "Tìm giúp chị chuyến bay từ Hà Nội vào Sài Gòn nhé.",
        "Coi giùm anh vé máy bay từ Đà Nẵng ra Hà Nội nghe.",
        "Kiếm chuyến bay Sài Gòn đi Phú Quốc cho tớ với.",
    ]),
    ("flights.search", "destination", [
        "Tìm chuyến bay từ Hà Nội ngày mai cho anh nhé.",
        "Coi giùm chị chuyến bay sáng thứ bảy từ Sài Gòn nghen.",
    ]),
    ("ride_hailing.book", "dropoff", [
        "Đặt xe bốn chỗ đón anh ở nhà nhé.",
        "Gọi xe máy từ công ty cho em với.",
        "Book xe bảy chỗ đón cả nhà ở nhà giùm chị nha.",
    ]),
    ("ride_hailing.book", "vehicle", [
        "Đặt xe từ nhà ra sân bay Tân Sơn Nhất giúp tôi.",
    ]),
    ("medication.add", "times", [
        "Thêm nhắc uống vitamin C cho chị nhé.",
        "Thêm lịch uống thuốc dạ dày giùm anh nghe.",
        "Thêm nhắc uống men vi sinh cho bé giúp em với.",
    ]),
    ("notes.create", "content", [
        "Tạo cho anh một ghi chú mới nhé.",
        "Mở ghi chú mới giùm chị với.",
    ]),
    ("tasks.add", "title", [
        "Thêm một việc vào danh sách việc cần làm giúp em nha.",
        "Thêm việc mới cho anh, hạn ngày mai nhé.",
    ]),
]

DATE, DT, TM = "2026-11-10", "2026-11-10T09:00:00+07:00", "09:00"


def value_for(prop):
    if "enum" in prop: return prop["enum"][0]
    t, fmt = prop.get("type"), prop.get("format")
    if t == "string": return {"date": DATE, "date-time": DT, "time": TM}.get(fmt, "test")
    if t in ("integer", "number"): return prop.get("minimum", 1) or 1
    if t == "boolean": return True
    if t == "array":
        it = prop.get("items", {})
        return [it["enum"][0]] if "enum" in it else ([1] if it.get("type") in ("integer", "number") else ["test"])
    if t == "object": return {}
    return "test"


def main():
    dev = set(l.strip() for l in open(os.path.join(HERE, "..", "eval", "splits", "dev_ids.txt"), encoding="utf-8") if l.strip())
    samples = [json.loads(l) for l in open(os.path.join(ROOT, "train", "train.jsonl"), encoding="utf-8")]
    by_id = {s["id"]: s for s in samples}
    cat = json.load(open(os.path.join(ROOT, "train", "train_tools_catalog.json"), encoding="utf-8"))
    tools = {t["name"]: t for a in cat["apps"] for t in a["tools"]}
    train_queries = {" ".join(s["audio_text"].lower().split()) for s in samples}

    def eligible(tool):
        app = tool.split(".")[0]
        return [s for s in samples if s["id"] not in dev and not s["history"]
                and app in s["context"]["installed_apps"] and tool in s["context"]["installed_apps"][app].get("tools", [])
                and not (set(s["context"]["installed_apps"]) & DEV_APPS)]

    def check(sample_id, tool, p):
        spec = tools[tool]["parameters"]; props = spec.get("properties", {})
        args = {k: value_for(props.get(k, {})) for k in spec.get("required", []) if k != p}
        for group in spec.get("x-one-of", []):  # chọn nhóm one-of không chứa p
            if p not in group and not any(k in args for k in group):
                for k in group: args[k] = value_for(props.get(k, {}))
                break
        env = ToolEnv("train", sample_id, root=ROOT)
        miss = env.validate(tool, args)
        full = dict(args); full[p] = value_for(props.get(p, {}))
        ok_full = env.validate(tool, full)
        return miss.get("ok") is False and p in (miss.get("error") or "") and ok_full.get("ok") is True, miss, ok_full

    rows, fails = [], []
    def add(src, sample, tool, p, query):
        assert tool.split(".")[0] not in DEV_APPS
        ok, miss, full = check(sample["id"], tool, p)
        if not ok:
            fails.append((src, sample["id"], tool, p, miss, full)); return
        if " ".join(query.lower().split()) in train_queries:
            fails.append((src, sample["id"], tool, p, "trùng câu train", None)); return
        rows.append({
            "id": f"ask_v2_{len(rows):04d}", "split": "train_synth", "source": src, "base_sample": sample["id"],
            "query": query, "label": "ask_user", "context": sample["context"], "history": [],
            "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": tool, "params": [p]}}],
            "apps": [tool.split(".")[0]], "tool": tool, "missing_param": p,
        })

    for sid, tool, p, q in BTC_REWRITE:
        assert sid not in dev
        add("btc_rewrite", by_id[sid], tool, p, q)
    for tool, p, queries in NEW:
        pool = eligible(tool)
        if not pool:
            fails.append(("opus_new", "-", tool, p, "không có mẫu train phù hợp", None)); continue
        for i, q in enumerate(queries):
            add("opus_new", pool[(i * 37) % len(pool)], tool, p, q)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("rows", len(rows), "| fails", len(fails))
    print("theo nguồn:", dict(collections.Counter(r["source"] for r in rows)))
    print("theo (tool, param):", dict(collections.Counter((r["tool"], r["missing_param"]) for r in rows)))
    for f_ in fails: print("FAIL", f_)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
