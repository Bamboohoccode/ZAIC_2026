"""Sinh bộ dữ liệu DPO preference pairs (chosen vs rejected) cho Qwen Planner từ tập dpo_train_ids.txt.

Mục tiêu (Task 2 - Tool Knowledge via Pure DPO):
- Không phình prompt hệ thống hay tool schema (giữ nguyên Champion V15 SYSTEM trong lifemate/pipeline.py).
- Nhồi tri thức phân biệt tool đúng/sai, tham số bắt buộc/tuỳ chọn, trình tự đọc-trước-ghi,
  và cổng chốt chặn mơ hồ (Ambiguity Gate) thông qua các cặp (chosen, rejected) chất lượng cao.
- Đầu ra (optimize/dpo/dataset/, được track bởi Git để cả nhóm pull về train ngay):
  - optimize/dpo/dataset/tool_specs_catalog.json
  - optimize/dpo/dataset/dpo_train.jsonl
  - optimize/dpo/dataset/dpo_val.jsonl (5% holdout nội bộ từ dpo_train_ids, 0% rò rỉ dev 300 hay test 100)
  - optimize/dpo/dataset/dpo_stats.json
  - optimize/dpo/README.md

Chạy:
  python -X utf8 scripts/build_dpo_dataset.py
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate.pipeline import (
    SYSTEM,
    assistant_msg,
    build_messages,
    clip,
    date_table,
    hints,
    tool_specs,
)

# ---------------------------------------------------------------------------
# Tri thức phân biệt tool dễ nhầm & tham số tuỳ chọn dễ bị ảo giác
# ---------------------------------------------------------------------------

CONFUSABLE_TOOLS: dict[str, list[tuple[str, str]]] = {
    # Giao tiếp (Communication)
    "sms.send": [
        ("chat_ott.send_message", "Người dùng yêu cầu gửi tin nhắn SMS qua số điện thoại/danh bạ, KHÔNG phải chat OTT."),
        ("email.send", "Người dùng yêu cầu gửi SMS ngắn, KHÔNG phải gửi email."),
        ("phone.call", "Người dùng yêu cầu nhắn tin SMS, KHÔNG phải gọi điện."),
    ],
    "chat_ott.send_message": [
        ("sms.send", "Người dùng yêu cầu nhắn qua app chat OTT (chat_id từ chat_ott.list_chats), KHÔNG phải gửi SMS thường."),
        ("chat_ott.send_location", "Người dùng gửi nội dung chữ, KHÔNG phải gửi vị trí."),
    ],
    "chat_ott.send_location": [
        ("chat_ott.send_message", "Người dùng muốn chia sẻ vị trí hiện tại qua chat_ott.send_location, KHÔNG phải gửi tin nhắn chữ."),
        ("maps.get_route", "Người dùng muốn gửi vị trí qua chat OTT, KHÔNG phải tìm đường."),
    ],
    "chat_ott.call": [
        ("phone.call", "Người dùng yêu cầu gọi qua ứng dụng chat OTT (cần chat_id), KHÔNG phải gọi điện thoại thường."),
        ("video_meeting.join", "Người dùng gọi trực tiếp qua OTT, KHÔNG phải vào phòng họp video."),
    ],
    "phone.call": [
        ("chat_ott.call", "Người dùng yêu cầu gọi điện thoại trực tiếp (cần contact_id/phone_number), KHÔNG phải gọi OTT."),
        ("sms.send", "Người dùng yêu cầu gọi điện, KHÔNG phải gửi tin nhắn SMS."),
    ],
    "contacts.search": [
        ("banking.list_beneficiaries", "Tra cứu liên hệ cá nhân (điện thoại/SMS/ví) dùng contacts.search; người thụ hưởng ngân hàng mới dùng banking.list_beneficiaries."),
        ("chat_ott.list_chats", "Tìm danh bạ cá nhân dùng contacts.search, còn tìm cuộc hội thoại OTT dùng chat_ott.list_chats."),
    ],
    "email.send": [
        ("email.create_draft", "Người dùng yêu cầu gửi email đi ngay (email.send), KHÔNG phải chỉ lưu nháp."),
        ("email.reply", "Gửi email mới theo địa chỉ to dùng email.send, còn trả lời email có sẵn mới dùng email.reply."),
        ("sms.send", "Gửi thư điện tử dùng email.send, KHÔNG phải SMS."),
    ],
    "email.create_draft": [
        ("email.send", "Người dùng chỉ yêu cầu soạn/lưu nháp email (email.create_draft), TUYỆT ĐỐI KHÔNG gửi đi bằng email.send."),
    ],
    "email.reply": [
        ("email.send", "Trả lời email hiện có cần email_id bằng email.reply, KHÔNG tạo thư rời."),
        ("email.create_draft", "Người dùng yêu cầu trả lời trực tiếp email, KHÔNG phải lưu nháp mới."),
    ],
    "email.list": [
        ("sms.read_recent", "Kiểm tra hộp thư email dùng email.list, KHÔNG phải SMS."),
        ("chat_ott.list_chats", "Kiểm tra email dùng email.list, KHÔNG phải chat OTT."),
    ],
    # Tài chính (Finance)
    "banking.list_beneficiaries": [
        ("contacts.search", "Lấy beneficiary_id để chuyển khoản ngân hàng BẮT BUỘC dùng banking.list_beneficiaries, CẤM dùng contacts.search."),
        ("banking.get_balance", "Lấy danh sách người thụ hưởng dùng banking.list_beneficiaries, xem số dư mới dùng banking.get_balance."),
    ],
    "banking.get_balance": [
        ("ewallet.get_balance", "Kiểm tra số dư tài khoản ngân hàng dùng banking.get_balance, KHÔNG nhầm với ví điện tử ewallet.get_balance."),
        ("banking.list_beneficiaries", "Lấy from_account_id và số dư dùng banking.get_balance."),
    ],
    "banking.transfer": [
        ("ewallet.transfer", "Chuyển tiền từ tài khoản ngân hàng tới beneficiary_id dùng banking.transfer, KHÔNG dùng ví điện tử ewallet.transfer."),
        ("bill_payment.pay", "Chuyển tiền cá nhân dùng banking.transfer, thanh toán hóa đơn mới dùng bill_payment.pay."),
    ],
    "banking.open_savings": [
        ("banking.transfer", "Mở sổ tiết kiệm dùng banking.open_savings, KHÔNG phải chuyển khoản."),
    ],
    "ewallet.get_balance": [
        ("banking.get_balance", "Kiểm tra số dư ví điện tử dùng ewallet.get_balance, KHÔNG nhầm với ngân hàng banking.get_balance."),
    ],
    "ewallet.transfer": [
        ("banking.transfer", "Chuyển tiền qua ví điện tử tới liên hệ/SĐT dùng ewallet.transfer, KHÔNG dùng chuyển khoản ngân hàng."),
        ("ewallet.topup_mobile", "Chuyển tiền cho người dùng ewallet.transfer, nạp tiền điện thoại dùng ewallet.topup_mobile."),
        ("bill_payment.pay", "Chuyển tiền ví cá nhân dùng ewallet.transfer, thanh toán hóa đơn dùng bill_payment.pay."),
    ],
    "ewallet.topup_mobile": [
        ("ewallet.transfer", "Nạp tiền điện thoại dùng ewallet.topup_mobile (phone, amount), KHÔNG dùng ewallet.transfer."),
        ("bill_payment.pay", "Nạp tiền trả trước trực tiếp dùng ewallet.topup_mobile, còn trả hóa đơn điện thoại/điện/nước dùng bill_payment.pay."),
    ],
    "ewallet.pay_qr": [
        ("food_delivery.place_order", "Thanh toán QR tại quán dùng ewallet.pay_qr với merchant_id, KHÔNG đặt đơn giao đồ ăn."),
        ("ewallet.transfer", "Thanh toán QR cửa hàng dùng ewallet.pay_qr (merchant_id), KHÔNG chuyển ví cá nhân."),
    ],
    "bill_payment.pay": [
        ("banking.transfer", "Thanh toán hóa đơn điện/nước/internet dùng bill_payment.pay (bill_id), KHÔNG chuyển khoản thường."),
        ("ewallet.transfer", "Thanh toán hóa đơn dùng bill_payment.pay, KHÔNG dùng ewallet.transfer."),
    ],
    "bill_payment.schedule_autopay": [
        ("bill_payment.pay", "Bật/tắt thanh toán tự động dùng bill_payment.schedule_autopay, KHÔNG phải trả hóa đơn lẻ."),
    ],
    "subscriptions.cancel": [
        ("bill_payment.schedule_autopay", "Hủy gói đăng ký định kỳ dùng subscriptions.cancel (subscription_id), KHÔNG dùng bill_payment."),
    ],
    # Kế hoạch & Lịch (Planning)
    "alarm.create": [
        ("reminders.create", "Đặt chuông báo thức theo giờ thức dậy dùng alarm.create (chỉ cần time), KHÔNG nhầm sang reminders.create."),
        ("calendar.create_event", "Đặt báo thức dùng alarm.create, tạo sự kiện/cuộc họp mới dùng calendar.create_event."),
        ("tasks.add", "Đặt báo thức dùng alarm.create, thêm đầu việc to-do dùng tasks.add."),
    ],
    "alarm.update": [
        ("alarm.create", "Sửa báo thức đã có phải đọc alarm.list lấy alarm_id rồi gọi alarm.update, KHÔNG tạo mới."),
        ("reminders.create", "Sửa báo thức dùng alarm.update, KHÔNG dùng reminders."),
    ],
    "alarm.delete": [
        ("alarm.update", "Xóa hẳn báo thức dùng alarm.delete, KHÔNG chỉ tắt tạm bằng alarm.update."),
        ("calendar.cancel_event", "Xóa báo thức dùng alarm.delete, hủy lịch họp mới dùng calendar.cancel_event."),
    ],
    "reminders.create": [
        ("alarm.create", "Tạo lời nhắc công việc/hành động cụ thể dùng reminders.create (content, time), KHÔNG dùng alarm.create."),
        ("calendar.create_event", "Tạo nhắc nhở nhanh dùng reminders.create, đặt lịch sự kiện có title/khoảng giờ mới dùng calendar.create_event."),
        ("tasks.add", "Nhắc theo thời điểm cụ thể dùng reminders.create, còn thêm đầu việc cần làm không nhất thiết dạng chuông nhắc dùng tasks.add."),
    ],
    "reminders.complete": [
        ("tasks.complete", "Hoàn thành nhắc việc của app reminders dùng reminders.complete (reminder_id), KHÔNG nhầm sang tasks.complete."),
        ("reminders.delete", "Đánh dấu đã xong nhắc việc dùng reminders.complete, xóa hẳn mới dùng reminders.delete."),
    ],
    "calendar.create_event": [
        ("reminders.create", "Tạo sự kiện/lịch hẹn trên lịch dùng calendar.create_event (title, start), KHÔNG nhầm với reminders.create."),
        ("video_meeting.create", "Tạo lịch hẹn thông thường dùng calendar.create_event, tạo phòng họp online mới dùng video_meeting.create."),
        ("tasks.add", "Đặt lịch hẹn có thời gian bắt đầu dùng calendar.create_event, KHÔNG dùng tasks.add."),
    ],
    "calendar.update_event": [
        ("calendar.create_event", "Dời/đổi giờ sự kiện đã có trên lịch PHẢI gọi calendar.list_events lấy event_id rồi dùng calendar.update_event, KHÔNG tạo sự kiện trùng."),
        ("reminders.create", "Đổi lịch sự kiện dùng calendar.update_event."),
    ],
    "calendar.cancel_event": [
        ("calendar.update_event", "Hủy sự kiện dùng calendar.cancel_event (event_id), KHÔNG sửa giờ."),
        ("video_meeting.join", "Hủy sự kiện lịch dùng calendar.cancel_event."),
    ],
    "tasks.add": [
        ("reminders.create", "Thêm việc vào danh sách to-do dùng tasks.add (title), còn đặt nhắc giờ cụ thể mới dùng reminders.create."),
        ("notes.create", "Thêm công việc cần làm có thể đánh dấu hoàn thành dùng tasks.add, ghi chú tự do mới dùng notes.create."),
        ("shopping_list.add", "Thêm công việc chung dùng tasks.add, thêm món cần mua mới dùng shopping_list.add."),
    ],
    "tasks.complete": [
        ("reminders.complete", "Đánh dấu xong công việc trong tasks dùng tasks.complete (task_id), KHÔNG dùng reminders.complete."),
    ],
    "notes.create": [
        ("notes.append", "Tạo ghi chú mới hoàn toàn dùng notes.create, thêm nối tiếp vào ghi chú cũ mới dùng notes.append."),
        ("tasks.add", "Lưu ghi chú văn bản dùng notes.create, tạo đầu việc cần làm mới dùng tasks.add."),
    ],
    "notes.append": [
        ("notes.create", "Người dùng muốn thêm nội dung vào ghi chú đã có: phải dùng notes.list lấy note_id rồi gọi notes.append, KHÔNG tạo ghi chú mới."),
    ],
    "video_meeting.create": [
        ("calendar.create_event", "Tạo cuộc họp trực tuyến dùng video_meeting.create, sự kiện lịch thường mới dùng calendar.create_event."),
    ],
    # Nhà thông minh (Smart Home)
    "smart_home.light_set": [
        ("smart_home.switch_set", "Điều khiển đèn PHẢI dùng smart_home.light_set, KHÔNG dùng smart_home.switch_set (chỉ cho ổ cắm/rèm/TV)."),
        ("smart_home.scene_run", "Chỉnh thiết bị đèn cụ thể dùng smart_home.light_set, chạy ngữ cảnh mới dùng smart_home.scene_run."),
        ("home_automation.create_rule", "Bật/tắt đèn ngay lúc này dùng smart_home.light_set, lập lịch lặp lại hằng ngày mới dùng home_automation.create_rule."),
    ],
    "smart_home.fan_set": [
        ("smart_home.switch_set", "Điều khiển quạt PHẢI dùng smart_home.fan_set, KHÔNG dùng smart_home.switch_set."),
        ("smart_home.climate_set", "Điều chỉnh quạt dùng smart_home.fan_set, điều hòa nhiệt độ mới dùng smart_home.climate_set."),
    ],
    "smart_home.climate_set": [
        ("smart_home.fan_set", "Chỉnh điều hòa/máy lạnh PHẢI dùng smart_home.climate_set (device_id, mode), KHÔNG dùng fan_set."),
        ("smart_home.switch_set", "Chỉnh điều hòa dùng smart_home.climate_set, KHÔNG dùng switch_set."),
    ],
    "smart_home.switch_set": [
        ("smart_home.light_set", "Đóng/mở rèm hoặc bật/tắt ổ cắm dùng smart_home.switch_set, KHÔNG dùng light_set."),
    ],
    "home_automation.create_rule": [
        ("smart_home.light_set", "Tạo quy tắc tự động lặp lại mỗi ngày theo giờ dùng home_automation.create_rule, KHÔNG chỉ bật tắt tức thời."),
        ("alarm.create", "Tự động điều khiển thiết bị theo giờ mỗi ngày dùng home_automation.create_rule, KHÔNG phải báo thức."),
    ],
    "door_lock.set_lock": [
        ("home_security.arm", "Khóa/mở khóa cửa cụ thể dùng door_lock.set_lock (lock_id, locked), bật chế độ an ninh tổng mới dùng home_security.arm."),
        ("door_lock.grant_guest_code", "Khóa/mở cửa dùng door_lock.set_lock, cấp mã khách mới dùng grant_guest_code."),
    ],
    "appliances.start_cycle": [
        ("robot_vacuum.start", "Chạy chu trình máy giặt/máy rửa bát/sấy dùng appliances.start_cycle (appliance_id, cycle), điều khiển robot hút bụi mới dùng robot_vacuum.start."),
        ("smart_home.switch_set", "Khởi động chu trình thiết bị gia dụng dùng appliances.start_cycle."),
    ],
    "robot_vacuum.start": [
        ("appliances.start_cycle", "Cho robot hút bụi chạy dùng robot_vacuum.start, KHÔNG dùng appliances.start_cycle."),
        ("smart_home.scene_run", "Chạy robot hút bụi dùng robot_vacuum.start."),
    ],
    # Thời tiết & Thông tin (Information)
    "weather.get_forecast": [
        ("weather.get_current", "Xem dự báo ngày mai/tương lai (có date) PHẢI dùng weather.get_forecast, KHÔNG dùng weather.get_current."),
    ],
    "weather.get_current": [
        ("weather.get_forecast", "Xem thời tiết ngay lúc này dùng weather.get_current, dự báo ngày khác mới dùng weather.get_forecast."),
    ],
    "news.search": [
        ("news.headlines", "Tìm tin theo từ khóa cụ thể dùng news.search (query), xem tin tổng hợp theo chủ đề dùng news.headlines."),
    ],
    "news.headlines": [
        ("news.search", "Xem điểm tin nổi bật tổng quát/theo chủ đề dùng news.headlines, tìm từ khóa cụ thể mới dùng news.search."),
    ],
    # Di chuyển, Ăn uống, Du lịch, Mua sắm, Media
    "maps.get_route": [
        ("maps.search_place", "Tìm đường đi giữa 2 điểm dùng maps.get_route (origin, destination), tìm thông tin địa điểm mới dùng maps.search_place."),
        ("ride_hailing.get_quotes", "Xem lộ trình/thời gian di chuyển dùng maps.get_route, xem giá cước gọi xe mới dùng ride_hailing.get_quotes."),
    ],
    "ride_hailing.book": [
        ("ride_hailing.get_quotes", "Người dùng yêu cầu đặt chuyến ngay dùng ride_hailing.book, chỉ hỏi giá trước mới dùng ride_hailing.get_quotes."),
        ("maps.get_route", "Đặt xe di chuyển dùng ride_hailing.book, KHÔNG chỉ xem đường."),
    ],
    "ride_hailing.get_quotes": [
        ("ride_hailing.book", "Người dùng chỉ hỏi giá cước ước tính (ride_hailing.get_quotes), KHÔNG tự ý đặt chuyến bằng ride_hailing.book."),
    ],
    "food_delivery.place_order": [
        ("shopping_list.add", "Đặt món ăn giao ngay dùng food_delivery.place_order, thêm vào danh sách đi chợ mới dùng shopping_list.add."),
        ("ewallet.pay_qr", "Đặt đơn giao đồ ăn dùng food_delivery.place_order, KHÔNG dùng pay_qr."),
    ],
    "shopping_list.check": [
        ("shopping_list.remove", "Đánh dấu món đã mua giữ lại trong danh sách dùng shopping_list.check, xóa hẳn mới dùng shopping_list.remove."),
    ],
    "shopping_list.remove": [
        ("shopping_list.check", "Xóa món khỏi danh sách đi chợ dùng shopping_list.remove, đánh dấu đã mua mới dùng shopping_list.check."),
    ],
    "music.play": [
        ("podcast_radio.play", "Phát nhạc/playlist dùng music.play, phát podcast/radio mới dùng podcast_radio.play."),
        ("music.control", "Phát một bài hát/playlist cụ thể (track_id/playlist_id) dùng music.play, chỉ tạm dừng/qua bài mới dùng music.control."),
    ],
    "podcast_radio.play": [
        ("music.play", "Phát podcast hoặc đài radio dùng podcast_radio.play, phát nhạc mới dùng music.play."),
        ("podcast_radio.queue", "Phát ngay lúc này dùng podcast_radio.play, chỉ thêm vào hàng đợi nghe sau mới dùng podcast_radio.queue."),
    ],
    "podcast_radio.queue": [
        ("podcast_radio.play", "Người dùng yêu cầu thêm vào danh sách nghe sau (podcast_radio.queue), KHÔNG phát đè ngay bằng podcast_radio.play."),
    ],
    # Các tool đọc hay nhầm với tool đọc khác hoặc tool cùng nhóm
    "calendar.list_events": [
        ("reminders.list", "Xem sự kiện lịch hẹn dùng calendar.list_events, xem nhắc việc mới dùng reminders.list."),
        ("tasks.list", "Xem lịch hẹn theo ngày dùng calendar.list_events, xem danh sách công việc dùng tasks.list."),
    ],
    "reminders.list": [
        ("calendar.list_events", "Xem danh sách lời nhắc dùng reminders.list, xem lịch sự kiện dùng calendar.list_events."),
        ("tasks.list", "Xem nhắc việc dùng reminders.list, xem đầu việc cần làm dùng tasks.list."),
    ],
    "learning.list_courses": [
        ("tasks.list", "Xem khóa học đăng ký/hiện có dùng learning.list_courses, KHÔNG nhầm sang tasks.list."),
    ],
    "music.list_tracks": [
        ("podcast_radio.search", "Xem thư viện bài hát/playlist nhạc dùng music.list_tracks, tìm podcast/radio mới dùng podcast_radio.search."),
    ],
    "podcast_radio.search": [
        ("music.list_tracks", "Tìm kênh/tập podcast hoặc đài radio dùng podcast_radio.search, xem nhạc dùng music.list_tracks."),
        ("news.search", "Tìm nội dung podcast/radio dùng podcast_radio.search, tìm tin tức báo chí mới dùng news.search."),
    ],
    "home_security.get_camera": [
        ("home_security.list_alerts", "Kiểm tra camera theo vị trí dùng home_security.get_camera, xem nhật ký cảnh báo dùng home_security.list_alerts."),
    ],
    "home_security.list_alerts": [
        ("home_security.get_camera", "Xem các cảnh báo an ninh dùng home_security.list_alerts, xem camera trực tiếp dùng home_security.get_camera."),
    ],
    "robot_vacuum.get_status": [
        ("appliances.list", "Kiểm tra pin/trạng thái robot hút bụi dùng robot_vacuum.get_status, xem máy giặt/thiết bị gia dụng mới dùng appliances.list."),
        ("smart_home.get_status", "Kiểm tra robot hút bụi dùng robot_vacuum.get_status, xem đèn/quạt/điều hòa dùng smart_home.get_status."),
    ],
    "loyalty.get_points": [
        ("loyalty.list_vouchers", "Kiểm tra số điểm tích lũy dùng loyalty.get_points, xem danh sách voucher hiện có dùng loyalty.list_vouchers."),
    ],
    "home_automation.list_rules": [
        ("smart_home.get_status", "Xem danh sách quy tắc tự động dùng home_automation.list_rules, xem trạng thái thiết bị tức thời dùng smart_home.get_status."),
    ],
    "email.read": [
        ("email.list", "Đã có sẵn email_id cụ thể để đọc nội dung chi tiết thì gọi thẳng email.read, KHÔNG gọi lại email.list."),
    ],
    "ride_hailing.get_ride": [
        ("ride_hailing.get_quotes", "Kiểm tra trạng thái/tài xế chuyến xe đã đặt dùng ride_hailing.get_ride, ước tính giá chuyến mới dùng ride_hailing.get_quotes."),
    ],
    "shopping_list.get": [
        ("tasks.list", "Xem danh sách món cần mua dùng shopping_list.get, xem đầu việc cần làm dùng tasks.list."),
    ],
    "flights.my_bookings": [
        ("flights.search", "Xem vé máy bay đã đặt của bản thân dùng flights.my_bookings, tìm chuyến bay mới dùng flights.search."),
    ],
}

# Các tham số tuỳ chọn mô hình hay tự điền (ảo giác) khi người dùng không hề nói
HALLUCINABLE_OPT_ARGS: dict[str, list[tuple[str, Any, str]]] = {
    "smart_home.get_status": [
        ("room", "phòng khách", "Không tự thêm bộ lọc room khi câu lệnh cần xem tổng thể hoặc khi chưa chắc tên phòng khớp state."),
        ("type", "light", "Không tự giới hạn type nếu cần đọc trạng thái chung."),
    ],
    "smart_home.light_set": [
        ("brightness", 100, "Không tự bịa brightness khi người dùng chỉ bảo bật/tắt đèn."),
        ("color", "white", "Không tự bịa color khi người dùng không yêu cầu đổi màu."),
    ],
    "smart_home.fan_set": [
        ("speed", 2, "Không tự bịa speed khi người dùng chỉ bảo bật/tắt quạt."),
    ],
    "smart_home.climate_set": [
        ("temperature", 25, "Không tự điền temperature mặc định khi người dùng không nêu nhiệt độ."),
    ],
    "weather.get_forecast": [
        ("location", "Hà Nội", "Không tự điền location nếu câu lệnh không chỉ định địa điểm cụ thể."),
    ],
    "weather.get_current": [
        ("location", "TP.HCM", "Không tự điền location nếu câu lệnh dùng vị trí mặc định hiện tại."),
    ],
    "alarm.create": [
        ("label", "Báo thức", "alarm.create KHÔNG tự bịa nhãn label khi người dùng không nói."),
        ("repeat", ["daily"], "alarm.create KHÔNG tự thêm lặp lại repeat khi người dùng chỉ đặt một lần."),
    ],
    "alarm.update": [
        ("label", "Báo thức", "alarm.update KHÔNG tự điền thêm label nếu người dùng chỉ đổi giờ."),
    ],
    "reminders.create": [
        ("repeat", "none", "reminders.create KHÔNG tự thêm tham số repeat khi người dùng không yêu cầu lặp lại."),
    ],
    "calendar.list_events": [
        ("date_from", "2026-11-10", "Tra cứu lịch chung gọi calendar.list_events với {}, KHÔNG tự thu hẹp date_from."),
        ("date_to", "2026-11-17", "Tra cứu lịch chung gọi calendar.list_events với {}, KHÔNG tự thu hẹp date_to."),
    ],
    "calendar.create_event": [
        ("location", "Phòng họp", "Không tự bịa location khi người dùng không nêu địa điểm."),
    ],
    "calendar.update_event": [
        ("title", "Sự kiện đã hẹn", "calendar.update_event KHÔNG tự điền lại title nếu người dùng chỉ dời giờ."),
        ("location", "Văn phòng", "calendar.update_event KHÔNG tự điền location nếu người dùng không yêu cầu đổi địa điểm."),
    ],
    "email.send": [
        ("cc", ["dongnghiep@congty.vn"], "email.send KHÔNG tự thêm cc khi người dùng không yêu cầu."),
    ],
    "email.create_draft": [
        ("cc", ["team@congty.vn"], "email.create_draft KHÔNG tự thêm cc khi người dùng không nói."),
    ],
    "notes.create": [
        ("title", "Ghi chú mới", "notes.create không cần tự đặt title khi người dùng chỉ đọc nội dung ghi chú."),
    ],
    "tasks.add": [
        ("priority", "medium", "tasks.add KHÔNG tự thêm priority khi người dùng không nói mức ưu tiên."),
        ("due", "2026-11-15", "tasks.add KHÔNG tự bịa hạn due khi người dùng không yêu cầu."),
    ],
    "contacts.search": [
        ("query", "Minh", "Khi cần đọc danh bạ chung hoặc tên mơ hồ không nên tự điền/đổi sai query."),
    ],
    "banking.transfer": [
        ("note", "Chuyển tiền", "banking.transfer KHÔNG tự bịa nội dung ghi chú note khi người dùng không dặn."),
    ],
    "ewallet.transfer": [
        ("note", "Chuyển tiền", "ewallet.transfer KHÔNG tự bịa note khi người dùng không nêu nội dung chuyển."),
    ],
    "bill_payment.list_bills": [
        ("type", "electricity", "Gọi bill_payment.list_bills với {} khi không cần lọc hẹp, tránh sót hóa đơn hoặc sai tham số."),
    ],
    "food_delivery.place_order": [
        ("payment", "cash", "Không tự bịa phương thức payment nếu người dùng không yêu cầu."),
    ],
    "ride_hailing.book": [
        ("payment", "cash", "Không tự bịa phương thức thanh toán payment khi người dùng không nói."),
    ],
    "flights.search": [
        ("passengers", 1, "Không tự thêm tham số tuỳ chọn passengers nếu người dùng không nêu số khách."),
    ],
    "flights.book": [
        ("seat_class", "economy", "flights.book KHÔNG tự điền seat_class khi người dùng không nói hạng ghế."),
    ],
    "shopping_list.add": [
        ("quantity", "1", "shopping_list.add KHÔNG tự thêm quantity nếu người dùng không nói số lượng."),
    ],
    "door_lock.grant_guest_code": [
        ("label", "Khách", "Không tự đặt label cho mã khách nếu người dùng không yêu cầu tên nhãn."),
    ],
    "home_automation.create_rule": [
        ("title", "Quy tắc tự động", "Không tự bịa title cho quy tắc nếu người dùng không đặt tên."),
    ],
}

OUT_DIR = ROOT / "optimize" / "dpo" / "dataset"


def norm_text(s: str) -> str:
    return re.sub(r"[^\w]+", " ", (s or "").casefold()).strip()


def call_tag(c: dict[str, Any]) -> str:
    return f'<tool_call>{json.dumps({"name": c["name"], "arguments": c.get("arguments") or {}}, ensure_ascii=False)}</tool_call>'


def format_tool_calls(calls: list[dict[str, Any]]) -> str:
    return "\n".join(call_tag(c) for c in calls)


def sample_apps(calls: list[dict[str, Any]]) -> list[str]:
    apps = []
    for c in calls:
        n = c["name"]
        if "." in n and not n.startswith("sys."):
            apps.append(n.split(".")[0])
        elif n == "sys.ask_missing_info":
            tn = (c.get("arguments") or {}).get("tool_name", "")
            if "." in tn:
                apps.append(tn.split(".")[0])
    return list(dict.fromkeys(apps))


def split_turns(calls: list[dict[str, Any]], results: list[Any], catalog: dict[str, Any]) -> list[list[int]]:
    """Nhóm chuỗi call gold thành các lượt (turn) tự nhiên, giữ nguyên tuyệt đối thứ tự gốc.
    - Meta (`sys.ask_missing_info`): luôn tách lượt riêng.
    - Read: gom các read liên tiếp không dùng x-ref từ kết quả ngay trong nhóm.
    - Write: gom các write liên tiếp không phụ thuộc x-ref lẫn nhau trong nhóm.
    Đã kiểm chứng hash_fail=0 trên toàn bộ 1,145 mẫu dpo_train.
    """
    turns: list[list[int]] = []
    i = 0
    while i < len(calls):
        kind = catalog[calls[i]["name"]]["kind"]
        if kind == "meta":
            turns.append([i])
            i += 1
        elif kind == "read":
            grp = [i]
            j = i + 1
            while j < len(calls) and catalog[calls[j]["name"]]["kind"] == "read":
                props = catalog[calls[j]["name"]]["parameters"].get("properties", {})
                has_xref = any(props.get(ak, {}).get("x-ref") for ak in (calls[j].get("arguments") or {}))
                if has_xref:
                    break
                grp.append(j)
                j += 1
            turns.append(grp)
            i = j
        else:
            grp = [i]
            j = i + 1
            while j < len(calls) and catalog[calls[j]["name"]]["kind"] == "write":
                props = catalog[calls[j]["name"]]["parameters"].get("properties", {})
                grp_blob = json.dumps([results[idx] for idx in grp], ensure_ascii=False)
                uses_grp = any(
                    isinstance(av, str) and av in grp_blob
                    for ak, av in (calls[j].get("arguments") or {}).items()
                    if props.get(ak, {}).get("x-ref")
                )
                if uses_grp:
                    break
                grp.append(j)
                j += 1
            turns.append(grp)
            i = j
    return turns


def describe_minimal_args(calls: list[dict[str, Any]], catalog: dict[str, Any]) -> str:
    parts = []
    for c in calls:
        n = c["name"]
        args = c.get("arguments") or {}
        spec = catalog.get(n, {}).get("parameters", {})
        req = spec.get("required", [])
        props = list(spec.get("properties", {}).keys())
        used_keys = list(args.keys())
        omitted_opt = [p for p in props if p not in req and p not in args]
        req_str = ", ".join(req) if req else "không có"
        used_str = ", ".join(used_keys) if used_keys else "{}"
        line = f"{n} (bắt buộc: [{req_str}], truyền: [{used_str}]"
        if omitted_opt:
            line += f", bỏ qua tuỳ chọn chưa yêu cầu: {omitted_opt[:4]}"
        line += ")"
        parts.append(line)
    return "; ".join(parts)


def make_chosen_response(
    calls: list[dict[str, Any]],
    step_idx: int,
    total_steps: int,
    intent: str,
    catalog: dict[str, Any],
    no_app_dropped: list[str] | None = None,
) -> str:
    if no_app_dropped is not None:
        return (
            f"BƯỚC 1. KIỂM TRA PHẠM VI: App xử lý yêu cầu ({', '.join(no_app_dropped)}) KHÔNG CÓ trong danh sách installed_apps. "
            "Yêu cầu ngoài phạm vi đã cài -> TUYỆT ĐỐI KHÔNG gọi tool nào (kể cả tool đọc hay sys.ask_missing_info).\n"
            "BƯỚC 2. HÀNH ĐỘNG: Dừng ngay và trả lời ngắn gọn.\n"
            "Xin lỗi, thiết bị hiện chưa cài đặt ứng dụng phù hợp để thực hiện yêu cầu này."
        )
    if not calls:
        return (
            "BƯỚC 1. KIỂM TRA PHẠM VI: Yêu cầu là trò chuyện/tự huỷ, ngoài phạm vi app đã cài, hoặc điều kiện kiểm tra trong lịch sử không thoả -> TUYỆT ĐỐI KHÔNG gọi tool nào.\n"
            "BƯỚC 2. HÀNH ĐỘNG: Không gọi tool và trả lời ngắn gọn.\n"
            "Đã rõ, tôi sẽ không thực hiện thao tác thay đổi nào thêm."
        )

    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        ask_c = next(c for c in calls if c["name"] == "sys.ask_missing_info")
        tn = (ask_c.get("arguments") or {}).get("tool_name", "")
        ps = (ask_c.get("arguments") or {}).get("params", [])
        step_note = "Sau khi đã gọi tool đọc tra cứu" if step_idx > 0 else "Dữ liệu người dùng chưa cung cấp đủ tham số bắt buộc phi-ID"
        cot = (
            f"BƯỚC 1. KIỂM TRA PHẠM VI & MƠ HỒ: {step_note}, tham số {ps} cho tool ghi `{tn}` bị thiếu hoặc mơ hồ (>=2 lựa chọn không phân biệt được). "
            "CẤM tự đoán hay tự bịa giá trị -> bắt buộc gọi `sys.ask_missing_info`.\n"
            f"BƯỚC 2. XÁC ĐỊNH ĐÍCH: Trỏ `tool_name` đúng vào tool ghi `{tn}` (không trỏ tool đọc), liệt kê đúng `params={ps}`.\n"
            f"BƯỚC 3. CÚ PHÁP: Xuất tool call hỏi lại người dùng."
        )
        return f"{cot}\n{format_tool_calls(calls)}"

    kinds = [catalog[c["name"]]["kind"] for c in calls]
    names = [c["name"] for c in calls]
    conf_notes = []
    for n in names:
        if n in CONFUSABLE_TOOLS:
            conf_notes.append(CONFUSABLE_TOOLS[n][0][1])
    conf_str = " ".join(dict.fromkeys(conf_notes)) if conf_notes else "Chọn đúng tool khớp hoàn toàn mục đích và miền ứng dụng đã cài."

    if all(k == "read" for k in kinds):
        b1 = "BƯỚC 1. KIỂM TRA PHẠM VI & MƠ HỒ: Yêu cầu hợp lệ trong installed_apps. Chưa đủ ID/trạng thái thực tế -> ĐỌC TRƯỚC, TUYỆT ĐỐI CẤM gọi tool ghi hay sys.ask_missing_info khi chưa đọc."
    else:
        b1 = "BƯỚC 1. KIỂM TRA PHẠM VI & MƠ HỒ: Yêu cầu cụ thể, rõ ràng, đã có đầy đủ ID/điều kiện xác nhận từ tool đọc hoặc hội thoại -> đủ điều kiện thực thi."

    b2 = f"BƯỚC 2. DANH SÁCH VIỆC & PHÂN BIỆT TOOL (Lượt {step_idx + 1}/{total_steps}): Chọn {', '.join(names)}. {conf_str}"
    b3 = f"BƯỚC 3. THAM SỐ TỐI THIỂU & NGUYÊN VĂN: {describe_minimal_args(calls, catalog)}. Giữ nguyên tên riêng, số liệu, chuỗi gốc của người dùng."
    return f"{b1}\n{b2}\n{b3}\n{format_tool_calls(calls)}"


def synthetic_arg_value(prop: dict[str, Any], name: str, context_now: str) -> Any:
    """Sinh giá trị hợp lệ tối thiểu theo schema khi tạo cặp negative confusable_tool."""
    if "enum" in prop:
        return prop["enum"][0]
    t = prop.get("type", "string")
    fmt = prop.get("format", "")
    if t == "integer":
        return max(1, int(prop.get("minimum", 1)))
    if t == "number":
        return float(max(1, prop.get("minimum", 1)))
    if t == "boolean":
        return True
    if t == "array":
        it = prop.get("items", {})
        return [synthetic_arg_value(it, name, context_now)]
    if t == "object":
        return {}
    if fmt == "date":
        return context_now[:10]
    if fmt == "date-time":
        return context_now if "T" in context_now else f"{context_now[:10]}T09:00:00+07:00"
    if fmt == "time":
        return "09:00"
    if "id" in name or prop.get("x-ref"):
        return f"{name.split('_')[0]}_01"
    # Giá trị tiếng Việt tự nhiên theo tên trường để negative trông giống lỗi thật của model
    natural_defaults = {
        "title": "Lịch hẹn mới",
        "content": "Nhắc việc hôm nay",
        "body": "Xin chào, gửi bạn thông tin chi tiết.",
        "subject": "Trao đổi công việc",
        "location": "TP.HCM",
        "room": "phòng khách",
        "note": "Chuyển khoản",
        "label": "Thông báo",
        "to": "lienhe@email.com",
        "query": "tin tức",
        "origin": "Nhà",
        "destination": "Sân bay",
        "pickup": "Vị trí hiện tại",
        "dropoff": "Trung tâm",
        "phone": "0901234567",
        "phone_number": "0901234567",
        "assignee": "Minh",
    }
    return natural_defaults.get(name, "thông tin")


def build_confusable_args(
    src_call: dict[str, Any],
    alt_tool: str,
    catalog: dict[str, Any],
    context_now: str,
) -> dict[str, Any]:
    """Ánh xạ tham số từ tool đúng sang tool dễ nhầm (hợp schema nhất có thể để thành hard negative)."""
    src_args = src_call.get("arguments") or {}
    alt_spec = catalog[alt_tool]["parameters"]
    alt_props = alt_spec.get("properties", {})
    alt_req = list(alt_spec.get("required", []))
    out: dict[str, Any] = {}

    # Các ánh xạ ngữ nghĩa phổ biến giữa hai tool dễ nhầm
    key_aliases = {
        "title": ["content", "title", "name"],
        "content": ["title", "content", "body", "note", "name"],
        "body": ["content", "body", "title"],
        "start": ["time", "start", "due", "date"],
        "time": ["start", "time", "due", "date"],
        "due": ["time", "start", "due"],
        "contact_id": ["chat_id", "beneficiary_id", "contact_id", "to", "phone"],
        "chat_id": ["contact_id", "chat_id"],
        "beneficiary_id": ["contact_id", "beneficiary_id"],
        "phone": ["phone_number", "phone"],
        "phone_number": ["phone", "phone_number"],
        "mode": ["power", "mode", "action"],
        "power": ["mode", "power", "action"],
        "query": ["topic", "query"],
        "topic": ["query", "topic"],
    }

    for ak in alt_req:
        if ak in src_args:
            out[ak] = src_args[ak]
        else:
            copied = None
            for cand in key_aliases.get(ak, []):
                if cand in src_args:
                    copied = src_args[cand]
                    break
            if copied is not None and (
                ("enum" not in alt_props[ak]) or (copied in alt_props[ak]["enum"])
            ):
                out[ak] = copied
            else:
                out[ak] = synthetic_arg_value(alt_props.get(ak, {}), ak, context_now)

    # Xử lý x-one-of nếu có
    for group in alt_spec.get("x-one-of", []):
        if not any(g in out for g in group):
            for g in group:
                if g in src_args:
                    out[g] = src_args[g]
                    break
            else:
                g0 = group[0]
                out[g0] = synthetic_arg_value(alt_props.get(g0, {}), g0, context_now)

    # Với các tool read có 0 required (ví dụ calendar.list_events vs tv.guide), giữ {} hoặc 1 arg trùng
    for k, v in src_args.items():
        if k in alt_props and k not in out and len(out) < 2:
            if "enum" not in alt_props[k] or v in alt_props[k]["enum"]:
                out[k] = v
    return out


def gen_negative_confusable(
    calls: list[dict[str, Any]],
    installed_tools: set[str],
    catalog: dict[str, Any],
    context_now: str,
) -> tuple[str, str] | None:
    for idx, c in enumerate(calls):
        n = c["name"]
        if n not in CONFUSABLE_TOOLS:
            continue
        # Ưu tiên tuyệt đối tool dễ nhầm đang CÓ mặt trong installed_tools của mẫu này
        alts = [alt for alt, _ in CONFUSABLE_TOOLS[n] if alt in installed_tools]
        if not alts:
            alts = [alt for alt, _ in CONFUSABLE_TOOLS[n] if alt in catalog]
        if not alts:
            continue
        alt = alts[0]
        bad_calls = copy.deepcopy(calls)
        bad_calls[idx] = {"name": alt, "arguments": build_confusable_args(c, alt, catalog, context_now)}
        resp = (
            f"BƯỚC 1. KIỂM TRA PHẠM VI: Yêu cầu hợp lệ.\n"
            f"BƯỚC 2. CHỌN TOOL: Dùng `{alt}` để xử lý yêu cầu người dùng.\n"
            f"BƯỚC 3. THỰC THI:\n{format_tool_calls(bad_calls)}"
        )
        return resp, f"swap:{n}->{alt}"
    return None


def gen_negative_hallucinated_args(
    calls: list[dict[str, Any]],
    catalog: dict[str, Any],
    rng: random.Random,
) -> tuple[str, str] | None:
    # 1. Thử thêm tham số tuỳ chọn người dùng không yêu cầu (từ schema thật của tool)
    for idx, c in enumerate(calls):
        n = c["name"]
        if n == "sys.ask_missing_info":
            continue
        args = c.get("arguments") or {}
        props = catalog[n]["parameters"].get("properties", {})
        cands = [(k, val, reason) for k, val, reason in HALLUCINABLE_OPT_ARGS.get(n, []) if k not in args and k in props]
        if not cands:
            # Fallback: lấy tham số tuỳ chọn bất kỳ trong schema chưa được dùng
            req = set(catalog[n]["parameters"].get("required", []))
            for k, pv in props.items():
                if k not in args and k not in req and not pv.get("x-ref"):
                    val = synthetic_arg_value(pv, k, "2026-11-15T09:00:00+07:00")
                    cands.append((k, val, f"Tự ý thêm tham số tuỳ chọn {k} dù người dùng không nói."))
                    break
        if cands:
            k, val, _ = cands[0]
            bad_calls = copy.deepcopy(calls)
            bad_calls[idx].setdefault("arguments", {})[k] = val
            resp = (
                f"BƯỚC 1. KIỂM TRA PHẠM VI: Đủ điều kiện thực hiện.\n"
                f"BƯỚC 2. CHỌN TOOL: Gọi `{n}`.\n"
                f"BƯỚC 3. THAM SỐ: Điền đầy đủ các trường bao gồm `{k}={json.dumps(val, ensure_ascii=False)}`.\n"
                f"{format_tool_calls(bad_calls)}"
            )
            return resp, f"extra_opt:{n}.{k}"

    # 2. Thử biến dạng chuỗi / số (ví dụ đuôi .test -> .com hoặc bỏ tham số tuỳ chọn người dùng CÓ yêu cầu)
    for idx, c in enumerate(calls):
        args = c.get("arguments") or {}
        for k, v in args.items():
            if isinstance(v, str) and v.endswith(".test"):
                bad_calls = copy.deepcopy(calls)
                bad_calls[idx]["arguments"][k] = v[:-5] + ".com"
                return (
                    f"BƯỚC 1. KIỂM TRA PHẠM VI: Hợp lệ.\nBƯỚC 2. Gọi `{c['name']}`.\n"
                    f"BƯỚC 3. Chuẩn hóa địa chỉ.\n{format_tool_calls(bad_calls)}",
                    f"literal_drift:{c['name']}.{k}",
                )
    return None


def gen_negative_premature_write_or_ask(
    turn_calls: list[dict[str, Any]],
    later_calls: list[dict[str, Any]],
    installed_tools: set[str],
    catalog: dict[str, Any],
    context_now: str,
) -> tuple[str, str] | None:
    """Trên lượt READ: (a) nhảy cóc gọi write với ID tự bịa, hoặc (b) gọi nhầm contacts.search cho banking,
    hoặc (c) gọi sys.ask_missing_info sớm trước khi đọc."""
    read_names = [c["name"] for c in turn_calls]
    # Trường hợp đặc biệt: sai nguồn ID banking.list_beneficiaries -> contacts.search
    if "banking.list_beneficiaries" in read_names and "contacts.search" in installed_tools:
        bad_calls = [
            {"name": "contacts.search", "arguments": {"query": "người nhận"}}
            if c["name"] == "banking.list_beneficiaries"
            else copy.deepcopy(c)
            for c in turn_calls
        ]
        return (
            "BƯỚC 1. KIỂM TRA PHẠM VI: Cần tìm người nhận để chuyển khoản.\n"
            "BƯỚC 2. NGUỒN ID: Dùng `contacts.search` tra tên người nhận.\n"
            f"BƯỚC 3. Gọi tool đọc.\n{format_tool_calls(bad_calls)}",
            "wrong_id_source:banking->contacts.search",
        )

    # Nhảy cóc gọi write ở bước đầu trước khi đọc x-ref ID
    for lc in later_calls:
        ln = lc["name"]
        if catalog[ln]["kind"] != "write":
            continue
        props = catalog[ln]["parameters"].get("properties", {})
        xrefs = [k for k in (lc.get("arguments") or {}) if props.get(k, {}).get("x-ref")]
        if xrefs:
            bad_call = copy.deepcopy(lc)
            for xk in xrefs:
                bad_call["arguments"][xk] = f"{xk.split('_')[0]}_01"
            return (
                "BƯỚC 1. KIỂM TRA PHẠM VI: Yêu cầu rõ ràng.\n"
                f"BƯỚC 2. THỰC THI NGAY: Bỏ qua bước đọc, gọi thẳng `{ln}` với ID mặc định.\n"
                f"BƯỚC 3. Xuất tool call.\n{call_tag(bad_call)}",
                f"premature_write:{ln}",
            )

    # Hỏi sys.ask_missing_info sớm trước khi đọc hoặc hỏi nhầm tool đọc
    if read_names and later_calls:
        target_write = next((lc for lc in later_calls if catalog[lc["name"]]["kind"] == "write"), None)
        if target_write:
            props = catalog[target_write["name"]]["parameters"].get("properties", {})
            missing_p = next(
                (k for k in (target_write.get("arguments") or {}) if props.get(k, {}).get("x-ref")),
                next(iter(catalog[target_write["name"]]["parameters"].get("required", ["id"])), "id"),
            )
            bad_ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": target_write["name"], "params": [missing_p]}}
            return (
                "BƯỚC 1. KIỂM TRA MƠ HỒ: Chưa thấy ID trong câu nói nên hỏi lại người dùng ngay trước khi tra cứu.\n"
                "BƯỚC 2. Dừng và hỏi.\n"
                f"BƯỚC 3. Gọi sys.ask_missing_info.\n{call_tag(bad_ask)}",
                f"premature_ask:{target_write['name']}",
            )
    return None


def gen_negative_ambiguity_gate(
    calls: list[dict[str, Any]],
    step_idx: int,
    installed_tools: set[str],
    catalog: dict[str, Any],
    msgs_so_far: list[dict[str, Any]],
    context_now: str,
) -> tuple[str, str] | None:
    """Lỗi Ambiguity Gate:
    - Nếu gold là sys.ask_missing_info: (a) tự bịa giá trị/chọn bừa ID đầu tiên từ tool_result để gọi tool ghi,
      HOẶC (b) trỏ sai tool_name vào tool đọc.
    - Nếu gold là write hợp lệ: gọi nhầm sys.ask_missing_info (False Ask).
    """
    ask_call = next((c for c in calls if c["name"] == "sys.ask_missing_info"), None)
    if ask_call:
        tn = (ask_call.get("arguments") or {}).get("tool_name", "")
        ps = (ask_call.get("arguments") or {}).get("params", [])
        if tn in catalog:
            spec = catalog[tn]["parameters"]
            props = spec.get("properties", {})
            guess_args: dict[str, Any] = {}
            tool_blob = "\n".join(str(m.get("content", "")) for m in msgs_so_far if m.get("role") == "tool")
            # Tìm đại 1 id xuất hiện trong tool_blob (ví dụ c_01 / dev_01 / evt_01) để mô phỏng chọn bừa ứng viên #1
            id_hits = re.findall(r'"id"\s*:\s*"([^"]+)"', tool_blob)
            for req_k in spec.get("required", []):
                if req_k in ps and props.get(req_k, {}).get("x-ref") and id_hits:
                    guess_args[req_k] = id_hits[0]
                else:
                    guess_args[req_k] = synthetic_arg_value(props.get(req_k, {}), req_k, context_now)
            for p_missing in ps:
                if p_missing not in guess_args:
                    if props.get(p_missing, {}).get("x-ref") and id_hits:
                        guess_args[p_missing] = id_hits[0]
                    elif props.get(p_missing, {}).get("type") in ("integer", "number") and "amount" in p_missing:
                        guess_args[p_missing] = 50000
                    else:
                        guess_args[p_missing] = synthetic_arg_value(props.get(p_missing, {}), p_missing, context_now)
            for g in spec.get("x-one-of", []):
                if not any(k in guess_args for k in g):
                    guess_args[g[0]] = id_hits[0] if (props.get(g[0], {}).get("x-ref") and id_hits) else synthetic_arg_value(props.get(g[0], {}), g[0], context_now)
            bad_write = {"name": tn, "arguments": guess_args}
            return (
                f"BƯỚC 1. KIỂM TRA MƠ HỒ: Có nhiều kết quả hoặc thiếu {ps}, nhưng chọn tạm giá trị mặc định/ứng viên đầu tiên để chạy luôn.\n"
                f"BƯỚC 2. THỰC THI: Gọi `{tn}`.\n"
                f"BƯỚC 3. Xuất tool call.\n{call_tag(bad_write)}",
                f"missed_ask_guessed:{tn}",
            )

    # Nếu gold là write hợp lệ -> tạo False Ask
    write_call = next((c for c in calls if catalog[c["name"]]["kind"] == "write"), None)
    if write_call:
        args = write_call.get("arguments") or {}
        target_p = next(iter(args.keys()), None)
        if not target_p:
            target_p = next(iter(catalog[write_call["name"]]["parameters"].get("required", ["id"])), "id")
        bad_ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": write_call["name"], "params": [target_p]}}
        return (
            f"BƯỚC 1. KIỂM TRA MƠ HỒ: Chưa chắc chắn tham số `{target_p}`, nên hỏi lại người dùng.\n"
            f"BƯỚC 2. Dừng quy trình.\n"
            f"BƯỚC 3. Gọi sys.ask_missing_info.\n{call_tag(bad_ask)}",
            f"false_ask:{write_call['name']}.{target_p}",
        )
    # Nếu gold là read (không có write) -> tạo False Ask trỏ nhầm vào tool đọc (lỗi kinh điển bị cấm ở V15)
    read_call = next((c for c in calls if catalog[c["name"]]["kind"] == "read"), None)
    if read_call:
        props = list(catalog[read_call["name"]]["parameters"].get("properties", {}).keys())
        target_p = props[0] if props else "query"
        bad_ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": read_call["name"], "params": [target_p]}}
        return (
            f"BƯỚC 1. KIỂM TRA MƠ HỒ: Hỏi lại tham số `{target_p}` trước khi gọi tool đọc `{read_call['name']}`.\n"
            "BƯỚC 2. Dừng và hỏi người dùng.\n"
            f"BƯỚC 3. Gọi sys.ask_missing_info.\n{call_tag(bad_ask)}",
            f"false_ask_on_read:{read_call['name']}",
        )
    return None


def gen_negative_scope_or_multi(
    calls: list[dict[str, Any]],
    later_calls: list[dict[str, Any]],
    item: dict[str, Any],
    catalog: dict[str, Any],
    context_now: str,
    no_app_dropped: list[str] | None = None,
) -> tuple[str, str] | None:
    """(a) Bỏ sót việc thứ 2 / nhánh ELSE (early stop), hoặc (b) False Act trên empty/no_app."""
    if no_app_dropped is not None:
        # Gọi bừa tool gốc vốn đã bị gỡ khỏi installed_apps hoặc sys.ask_missing_info
        orig_calls = item["label"]["tool_calls"]
        bad_call = copy.deepcopy(orig_calls[0]) if orig_calls else {"name": "sys.ask_missing_info", "arguments": {"tool_name": "app.action", "params": ["id"]}}
        if bad_call["name"] == "sys.ask_missing_info" or no_app_dropped:
            return (
                "BƯỚC 1. KIỂM TRA PHẠM VI: Thử gọi tool đáp ứng yêu cầu người dùng.\n"
                "BƯỚC 2. Thực thi mặc dù app không nằm trong installed_apps.\n"
                f"BƯỚC 3. Xuất tool call.\n{call_tag(bad_call)}",
                f"false_act_no_app:{bad_call['name']}",
            )

    if not calls:
        # Empty sample: tạo False Act gọi 1 tool hợp lệ đầu tiên trong installed_apps hoặc sys.ask_missing_info
        installed = list(item["context"]["installed_apps"].keys())
        picked = None
        for app_id in installed:
            for tname in item["context"]["installed_apps"][app_id].get("tools", []):
                if catalog[tname]["kind"] == "read" and not catalog[tname]["parameters"].get("required"):
                    picked = {"name": tname, "arguments": {}}
                    break
            if picked:
                break
        if picked is None:
            picked = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "calendar.create_event", "params": ["title"]}}
        return (
            "BƯỚC 1. KIỂM TRA PHẠM VI: Cố gắng thực hiện bằng cách gọi tool kiểm tra hoặc hỏi lại.\n"
            f"BƯỚC 2. Gọi `{picked['name']}`.\n"
            f"BƯỚC 3. Xuất tool call.\n{call_tag(picked)}",
            f"false_act_empty:{picked['name']}",
        )

    # Multi-action / multi-step early stop: bỏ qua call còn lại trong cùng lượt hoặc dừng hẳn không làm tiếp later_calls
    if len(calls) > 1:
        return (
            "BƯỚC 1. KIỂM TRA PHẠM VI: Thực hiện tác vụ đầu tiên trong yêu cầu.\n"
            f"BƯỚC 2. Chỉ gọi `{calls[0]['name']}` rồi dừng, bỏ sót các việc còn lại.\n"
            f"BƯỚC 3. Xuất tool call.\n{call_tag(calls[0])}",
            "multi_action_early_stop",
        )
    if later_calls:
        # Kết thúc sớm ngay sau lượt read mà không gọi tiếp bước thực thi
        return (
            "BƯỚC 1. KIỂM TRA PHẠM VI: Đã hoàn tất bước tra cứu ban đầu.\n"
            "BƯỚC 2. Dừng sớm, bỏ qua bước ghi/nhánh ngược lại tiếp theo.\n"
            "Tôi đã kiểm tra thông tin xong rồi ạ.",
            "early_stop_after_read",
        )
    return None


def make_no_app_sample(item: dict[str, Any], app_groups: dict[str, str]) -> tuple[dict[str, Any], list[str]] | None:
    calls = item["label"]["tool_calls"]
    if not calls:
        return None
    used = set(sample_apps(calls))
    if not used:
        return None
    used_groups = {app_groups[u] for u in used if app_groups.get(u)}
    drop = used | {a for a in item["context"]["installed_apps"] if app_groups.get(a) in used_groups}
    hist_apps = {c["name"].split(".")[0] for h in item["history"] for c in (h.get("tool_calls") or []) if "." in c["name"]}
    if drop & hist_apps:
        return None
    new_item = copy.deepcopy(item)
    new_item["context"]["installed_apps"] = {k: v for k, v in item["context"]["installed_apps"].items() if k not in drop}
    return new_item, sorted(drop)


def build_dataset(root: Path, out_dir: Path, val_ratio: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    train_dir = root / "train"
    sys.path.insert(0, str(train_dir.resolve()))
    from zaic_tools import ToolEnv

    raw_catalog = json.load(open(train_dir / "train_tools_catalog.json", encoding="utf-8"))
    catalog = {t["name"]: t for app in raw_catalog["apps"] for t in app["tools"]}
    app_groups = {app["id"]: app.get("group", "") for app in raw_catalog["apps"]}

    # Xuất catalog định dạng đúng pipeline.tool_specs() để train_dpo.py dựng tools siêu gọn
    specs_catalog = {
        n: {
            "type": "function",
            "function": {
                "name": n,
                "description": f"[{t['kind']}] {t['description']}" + (f" Trả về: {t['returns']}" if t.get("returns") else ""),
                "parameters": t["parameters"],
            },
        }
        for n, t in catalog.items()
    }

    rows = {x["id"]: x for x in map(json.loads, open(train_dir / "train.jsonl", encoding="utf-8")) if x["id"]}
    dev_ids = set(open(ROOT / "eval" / "splits" / "dev_ids.txt", encoding="utf-8").read().split())
    test_ids = set(open(ROOT / "eval" / "splits" / "test_ids.txt", encoding="utf-8").read().split())
    dpo_ids = [s.strip() for s in open(ROOT / "eval" / "splits" / "dpo_train_ids.txt", encoding="utf-8") if s.strip()]

    # Kiểm tra chéo 0% rò rỉ với dev (300) và test (100)
    assert not (set(dpo_ids) & dev_ids)
    assert not (set(dpo_ids) & test_ids)
    blocked_texts = {norm_text(rows[i]["audio_text"]) for i in (dev_ids | test_ids)}

    # Loại bản trùng nội bộ trong dpo_train_ids (giữ bản đầu)
    seen_texts: set[str] = set()
    clean_dpo_ids: list[str] = []
    skipped_dups: list[str] = []
    for sid in dpo_ids:
        nt = norm_text(rows[sid]["audio_text"])
        assert nt not in blocked_texts, f"Text leak for {sid}"
        if nt in seen_texts:
            skipped_dups.append(sid)
            continue
        seen_texts.add(nt)
        clean_dpo_ids.append(sid)

    # Tách stratified val (5%) từ chính tập dpo_train_ids để không chạm vào dev_ids (300) hay test_ids (100)
    by_intent: dict[str, list[str]] = defaultdict(list)
    for sid in clean_dpo_ids:
        calls = rows[sid]["label"]["tool_calls"]
        intent = "empty" if not calls else ("ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "act")
        by_intent[intent].append(sid)

    val_sample_ids: set[str] = set()
    for intent, sids in sorted(by_intent.items()):
        sids_copy = list(sids)
        rng.shuffle(sids_copy)
        n_val = max(1, round(len(sids_copy) * val_ratio))
        val_sample_ids.update(sids_copy[:n_val])

    pairs_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    stats = {
        "source_dpo_ids": len(dpo_ids),
        "deduped_samples": len(clean_dpo_ids),
        "skipped_internal_duplicates": skipped_dups,
        "train_samples": len(clean_dpo_ids) - len(val_sample_ids),
        "val_samples": len(val_sample_ids),
        "gold_hash_verified": 0,
        "gold_hash_failed": 0,
        "by_split": Counter(),
        "by_intent": Counter(),
        "by_negative_class": Counter(),
        "by_app": Counter(),
    }

    for sid in clean_dpo_ids:
        item = rows[sid]
        split_name = "val" if sid in val_sample_ids else "train"
        calls = item["label"]["tool_calls"]
        results = item["label"].get("tool_results") or []
        intent = "empty" if not calls else ("ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "act")
        target_apps = sample_apps(calls) or ["none"]
        now_str = item["context"]["current_time"]

        # Khởi tạo conversation gốc chuẩn pipeline.py
        turns_texts = [h["audio_text"] for h in item["history"] if h["role"] == "user"] + [item["audio_text"]]
        base_msgs = build_messages(item, turns_texts, keep=len(turns_texts) - 1)
        tools = tool_specs(item, catalog)
        tool_names = [t["function"]["name"] for t in tools]
        installed_set = set(tool_names)

        # Verify gold ToolEnv hash
        env_gold = ToolEnv("train", sid, root=str(root))
        for c in calls:
            env_gold.call(c["name"], c.get("arguments") or {})
        gold_hash = env_gold.get_state_hash()

        if not calls:
            # Mẫu empty gốc: chỉ 1 turn
            chosen = make_chosen_response([], 0, 1, "empty", catalog)
            neg = gen_negative_scope_or_multi([], [], item, catalog, now_str)
            if neg:
                rej_text, detail = neg
                pairs_by_split[split_name].append({
                    "id": f"{sid}_s0_empty_scope",
                    "sample_id": sid,
                    "step": 0,
                    "intent": "empty",
                    "negative_class": "multi_action_or_scope_error",
                    "negative_detail": detail,
                    "apps": target_apps,
                    "tool_names": tool_names,
                    "prompt": base_msgs,
                    "chosen": chosen,
                    "rejected": rej_text,
                })
                stats["by_split"][split_name] += 1
                stats["by_intent"]["empty"] += 1
                stats["by_negative_class"]["multi_action_or_scope_error"] += 1
                stats["by_app"][target_apps[0]] += 1
            stats["gold_hash_verified"] += 1
            continue

        # Mẫu act / ask: tách step và chạy replay qua ToolEnv
        turn_groups = split_turns(calls, results, catalog)
        env = ToolEnv("train", sid, root=str(root))
        msgs = copy.deepcopy(base_msgs)

        for step_idx, grp in enumerate(turn_groups):
            step_calls = [calls[idx] for idx in grp]
            later_calls = [calls[idx] for later_grp in turn_groups[step_idx + 1:] for idx in later_grp]
            chosen = make_chosen_response(step_calls, step_idx, len(turn_groups), intent, catalog)

            negs: list[tuple[str, str, str]] = []
            if res := gen_negative_confusable(step_calls, installed_set, catalog, now_str):
                negs.append(("confusable_tool", res[0], res[1]))
            if res := gen_negative_hallucinated_args(step_calls, catalog, rng):
                negs.append(("hallucinated_optional_args", res[0], res[1]))
            if step_idx == 0 and all(catalog[c["name"]]["kind"] == "read" for c in step_calls):
                if res := gen_negative_premature_write_or_ask(step_calls, later_calls, installed_set, catalog, now_str):
                    negs.append(("premature_write_or_ask", res[0], res[1]))
            if res := gen_negative_ambiguity_gate(step_calls, step_idx, installed_set, catalog, msgs, now_str):
                negs.append(("ambiguity_gate_error", res[0], res[1]))
            if res := gen_negative_scope_or_multi(step_calls, later_calls, item, catalog, now_str):
                negs.append(("multi_action_or_scope_error", res[0], res[1]))

            # Giữ tối đa 2-3 cặp đa dạng cho mỗi step để cân bằng và giữ file dưới 50MB cho GitHub
            cap = 2 if (len(msgs) > 10 or step_idx >= 2) else 3
            if len(negs) > cap:
                rng.shuffle(negs)
                negs = negs[:cap]

            for neg_cls, rej_text, detail in negs:
                pairs_by_split[split_name].append({
                    "id": f"{sid}_s{step_idx}_{neg_cls}",
                    "sample_id": sid,
                    "step": step_idx,
                    "intent": intent,
                    "negative_class": neg_cls,
                    "negative_detail": detail,
                    "apps": target_apps,
                    "tool_names": tool_names,
                    "prompt": copy.deepcopy(msgs),
                    "chosen": chosen,
                    "rejected": rej_text,
                })
                stats["by_split"][split_name] += 1
                stats["by_intent"][intent] += 1
                stats["by_negative_class"][neg_cls] += 1
                for a in target_apps:
                    stats["by_app"][a] += 1

            # Cập nhật lịch sử msgs bằng kết quả thật của ToolEnv cho step kế tiếp
            msgs.append(assistant_msg("", step_calls))
            for c in step_calls:
                r = env.call(c["name"], c.get("arguments") or {})
                msgs.append({"role": "tool", "content": clip(r)})

        if env.get_state_hash() == gold_hash:
            stats["gold_hash_verified"] += 1
        else:
            stats["gold_hash_failed"] += 1

        # Sinh thêm 1 mẫu synthetic `no_app` (cho ~18% mẫu act để dạy ranh giới installed_apps)
        if intent == "act" and rng.random() < 0.18:
            no_app_res = make_no_app_sample(item, app_groups)
            if no_app_res:
                no_app_item, dropped_apps = no_app_res
                no_app_msgs = build_messages(no_app_item, turns_texts, keep=len(turns_texts) - 1)
                no_app_tools = tool_specs(no_app_item, catalog)
                no_app_tool_names = [t["function"]["name"] for t in no_app_tools]
                chosen_na = make_chosen_response([], 0, 1, "no_app", catalog, no_app_dropped=dropped_apps)
                if neg := gen_negative_scope_or_multi([], [], no_app_item, catalog, now_str, no_app_dropped=dropped_apps):
                    rej_na, detail_na = neg
                    pairs_by_split[split_name].append({
                        "id": f"{sid}_no_app",
                        "sample_id": sid,
                        "step": 0,
                        "intent": "no_app",
                        "negative_class": "multi_action_or_scope_error",
                        "negative_detail": detail_na,
                        "apps": dropped_apps,
                        "tool_names": no_app_tool_names,
                        "prompt": no_app_msgs,
                        "chosen": chosen_na,
                        "rejected": rej_na,
                    })
                    stats["by_split"][split_name] += 1
                    stats["by_intent"]["no_app"] += 1
                    stats["by_negative_class"]["multi_action_or_scope_error"] += 1

    assert stats["gold_hash_failed"] == 0, f"Found {stats['gold_hash_failed']} ToolEnv hash mismatches!"

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "tool_specs_catalog.json"
    catalog_path.write_text(json.dumps(specs_catalog, ensure_ascii=False, indent=1), encoding="utf-8")

    file_sizes_mb = {"tool_specs_catalog.json": round(catalog_path.stat().st_size / 1e6, 3)}
    for sp in ("train", "val"):
        p = out_dir / f"dpo_{sp}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for rec in pairs_by_split[sp]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        file_sizes_mb[p.name] = round(p.stat().st_size / 1e6, 3)

    json_stats = {
        k: (dict(v) if isinstance(v, Counter) else v)
        for k, v in stats.items()
    }
    json_stats["file_sizes_mb"] = file_sizes_mb
    json_stats["total_pairs"] = sum(stats["by_split"].values())
    (out_dir / "dpo_stats.json").write_text(json.dumps(json_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DPO dataset built successfully:")
    print(f"  Samples processed: {stats['deduped_samples']} (train={stats['train_samples']}, val={stats['val_samples']})")
    print(f"  ToolEnv gold hash verified: {stats['gold_hash_verified']} / failed: {stats['gold_hash_failed']}")
    print(f"  Pairs: train={stats['by_split']['train']}, val={stats['by_split']['val']}, total={json_stats['total_pairs']}")
    print(f"  By intent: {dict(stats['by_intent'])}")
    print(f"  By negative class: {dict(stats['by_negative_class'])}")
    print(f"  File sizes (MB): {file_sizes_mb}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_dataset(args.root, args.out_dir, args.val_ratio, args.seed)
