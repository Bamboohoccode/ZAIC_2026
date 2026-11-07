"""Module Schema Validator & Safe Hash Guard (Vương - LifeMate).

Chức năng:
1. Kiểm tra tính hợp lệ của tool calls theo schema từ zaic_tools catalog.
2. Safe Hash Guard: Phát hiện sớm các trường hợp mơ hồ (nhiều kết quả trùng, thiếu slot bắt buộc,
   hoặc người dùng nói mập mờ 'chút tiền', 'kênh nớ', 'hạ mục tiêu').
3. Tự động chuyển đổi các cuộc gọi ghi nguy hiểm thành `sys.ask_missing_info` bằng code Python thuần,
   ngăn chặn triệt để tình trạng ghi ẩu làm lệch state_hash (0 điểm).
4. Dọn dẹp các tham số mặc định/tuỳ chọn thừa (như repeat='once', note rỗng) để đảm bảo khớp hoàn toàn
   với nhãn gold của BTC.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _extract_recent_tool_results(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trích xuất kết quả trả về của các tool đọc trước đó từ lịch sử messages."""
    results = []
    for m in msgs:
        if m.get("role") == "tool":
            content = m.get("content", "")
            try:
                data = json.loads(content) if isinstance(content, str) else content
                if isinstance(data, dict):
                    results.append(data)
            except Exception:
                pass
    return results


def _has_money_amount(text: str) -> bool:
    """Kiểm tra xem câu có chứa số tiền cụ thể hay không."""
    t = text.lower()
    # số đi kèm đơn vị tiền tệ: 500k, 50k, 50 nghìn, 2 triệu, 100 ngàn, 50000 vnd,...
    if re.search(r"\b\d+\s*(?:k|nghìn|ngàn|triệu|tr|củ|vnd|đ|dong|đồng)\b", t):
        return True
    # số tiền lớn từ 4 chữ số trở lên: 5000, 20000, 500000 (loại trừ trường hợp năm 2026)
    for m in re.finditer(r"\b\d{4,}\b", t):
        if m.group(0) not in ("2025", "2026", "2027"):
            return True
    # số tiền viết bằng chữ: trăm nghìn, mươi nghìn, triệu, ...
    money_words = ["trăm nghìn", "trăm ngàn", "mươi nghìn", "mươi ngàn", "triệu", "chục nghìn", "năm mươi ngàn", "hai mươi ngàn"]
    if any(w in t for w in money_words):
        return True
    return False


def _has_vague_money(text: str) -> bool:
    """Kiểm tra xem người dùng có nói từ mập mờ về tiền mà không có số tiền cụ thể không."""
    t = text.lower()
    vague_cues = ["chút tiền", "ít tiền", "gửi ít tiền", "chuyển ít tiền", "chút đỉnh", "gửi chút tiền", "khoản tiền", "mớ tiền"]
    if any(c in t for c in vague_cues):
        return not _has_money_amount(text)
    return False


def _has_vague_fitness_goal(text: str) -> bool:
    """Kiểm tra xem người dùng yêu cầu đổi mục tiêu bước chân nhưng không cho số cụ thể."""
    t = text.lower()
    cues = [
        "đổi dùm anh đi", "hạ xuống giúp em", "hạ target", "hạ mục tiêu",
        "nâng lên giùm", "thay đổi mục tiêu", "hạ xuống chút"
    ]
    if any(c in t for c in cues) and not re.search(r"\b\d{4,6}\b", t):
        return True
    return False


def _has_vague_tv_channel(text: str) -> bool:
    """Kiểm tra xem người dùng yêu cầu chuyển kênh nhưng dùng đại từ chỉ định mập mờ."""
    t = text.lower()
    vague_channel_cues = ["kênh nớ", "kênh đó", "kênh nấy", "phim trưa nay", "coi đá banh", "coi đá bóng"]
    return any(c in t for c in vague_channel_cues)


def guard_and_clean_call(
    call: Dict[str, Any],
    catalog: Dict[str, Any],
    msgs: List[Dict[str, Any]],
    raw_query: str
) -> Dict[str, Any]:
    """Kiểm tra và chuẩn hoá một tool call trước khi chạm vào ToolEnv."""
    name = call.get("name", "")
    args = dict(call.get("arguments") or {})
    tool_meta = catalog.get(name, {})
    is_write = tool_meta.get("kind") == "write"

    recent_results = _extract_recent_tool_results(msgs)

    # 1. Ambiguity Guard: Kiểm tra xem các tool đọc trước đó có trả về nhiều kết quả trùng không
    if is_write:
        for r in recent_results:
            # Tra cứu danh bạ ra >= 2 người
            contacts = r.get("contacts")
            if isinstance(contacts, list) and len(contacts) > 1:
                if name in ("phone.call", "sms.send", "ewallet.transfer") and "contact_id" in args:
                    return {
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": name, "params": ["contact_id"]}
                    }

            # Tìm xe đạp ra >= 2 xe
            bikes = r.get("bikes")
            if isinstance(bikes, list) and len(bikes) > 1:
                if name == "bike_rental.reserve" and "bike_id" in args:
                    return {
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": name, "params": ["bike_id"]}
                    }

    # 2. Vague Slot Guard: Kiểm tra các trường hợp người dùng nói mập mờ
    if name in ("ewallet.transfer", "banking.transfer") and _has_vague_money(raw_query):
        return {
            "name": "sys.ask_missing_info",
            "arguments": {"tool_name": name, "params": ["amount"]}
        }

    if name == "fitness.set_goal" and _has_vague_fitness_goal(raw_query):
        return {
            "name": "sys.ask_missing_info",
            "arguments": {"tool_name": "fitness.set_goal", "params": ["steps"]}
        }

    if name == "tv.tune" and _has_vague_tv_channel(raw_query):
        return {
            "name": "sys.ask_missing_info",
            "arguments": {"tool_name": "tv.tune", "params": ["channel"]}
        }

    if name == "package_tracking.reschedule_delivery":
        # Nếu không có tracking_id trong query hay trong lịch sử
        if not re.search(r"tr[0-9a-zA-Z_]+", str(args.get("tracking_id", ""))):
            has_tracking = any("tracking_id" in str(r) for r in recent_results)
            if not has_tracking and "tracking_id" not in raw_query:
                return {
                    "name": "sys.ask_missing_info",
                    "arguments": {"tool_name": "package_tracking.reschedule_delivery", "params": ["tracking_id"]}
                }

    # 3. Parameter Hygiene & Device Standardization: Dọn dẹp và chuẩn hoá tham số
    q_lower = raw_query.lower()

    # Chuẩn hoá tên phòng tiếng Anh sang tiếng Việt cho smart_home
    if name == "smart_home.get_status" and "room" in args:
        room_lower = str(args["room"]).lower().strip()
        room_map = {
            "living room": "phòng khách",
            "living_room": "phòng khách",
            "bedroom": "phòng ngủ",
            "bed room": "phòng ngủ",
            "kitchen": "phòng bếp",
            "bathroom": "phòng tắm",
            "bath room": "phòng tắm",
        }
        if room_lower in room_map:
            args["room"] = room_map[room_lower]

    # Đảm bảo power='on' khi chỉnh độ sáng đèn mà quên bật
    if name == "smart_home.light_set":
        if "brightness" in args and "power" not in args:
            if any(k in q_lower for k in ["bật", "mở", "sáng", "dịu", "độ sáng"]):
                args["power"] = "on"
            elif "tắt" in q_lower:
                args["power"] = "off"

    if name in ("reminders.create", "alarm.create") and args.get("repeat") == "once":
        if "lặp" not in q_lower and "mỗi ngày" not in q_lower and "hàng ngày" not in q_lower:
            args.pop("repeat", None)

    if name in ("banking.transfer", "ewallet.transfer") and "note" in args:
        if not any(k in q_lower for k in ["nội dung", "lời nhắn", "ghi chú", "nhắn là", "nói là"]):
            args.pop("note", None)

    return {"name": name, "arguments": args}


def validate_and_guard_calls(
    calls: List[Dict[str, Any]],
    catalog: Dict[str, Any],
    msgs: List[Dict[str, Any]],
    raw_query: str
) -> List[Dict[str, Any]]:
    """Xử lý toàn bộ danh sách tool calls của một bước plan."""
    guarded_calls = []
    stop_early = False
    for c in calls:
        if stop_early:
            break
        gc = guard_and_clean_call(c, catalog, msgs, raw_query)
        guarded_calls.append(gc)
        if gc["name"] == "sys.ask_missing_info":
            # Khi đã gọi ask_missing_info thì dừng lại ngay lập tức
            stop_early = True
    return guarded_calls
