"""Unit tests for lifemate.validator.validator."""
import sys
from pathlib import Path
sys.path.insert(0, ".")

from lifemate.validator.validator import validate_and_guard_calls


def test_contact_ambiguity_guard():
    catalog = {"phone.call": {"kind": "write"}}
    msgs = [
        {"role": "user", "content": "gọi cho Bình giúp chị cái"},
        {"role": "tool", "content": '{"ok": true, "contacts": [{"id": "c_01", "name": "Bình"}, {"id": "c_02", "name": "Bình"}]}'}
    ]
    calls = [{"name": "phone.call", "arguments": {"contact_id": "c_01"}}]
    guarded = validate_and_guard_calls(calls, catalog, msgs, "gọi cho Bình giúp chị cái")
    assert len(guarded) == 1
    assert guarded[0]["name"] == "sys.ask_missing_info"
    assert guarded[0]["arguments"] == {"tool_name": "phone.call", "params": ["contact_id"]}
    print("test_contact_ambiguity_guard passed!")


def test_bike_ambiguity_guard():
    catalog = {"bike_rental.reserve": {"kind": "write"}}
    msgs = [
        {"role": "user", "content": "giữ giùm cô một chiếc ở trạm Biển Đông"},
        {"role": "tool", "content": '{"ok": true, "bikes": [{"id": "b1"}, {"id": "b2"}]}'}
    ]
    calls = [{"name": "bike_rental.reserve", "arguments": {"bike_id": "b1"}}]
    guarded = validate_and_guard_calls(calls, catalog, msgs, "giữ giùm cô một chiếc ở trạm Biển Đông")
    assert len(guarded) == 1
    assert guarded[0]["name"] == "sys.ask_missing_info"
    assert guarded[0]["arguments"] == {"tool_name": "bike_rental.reserve", "params": ["bike_id"]}
    print("test_bike_ambiguity_guard passed!")


def test_vague_money_guard():
    catalog = {"ewallet.transfer": {"kind": "write"}}
    msgs = [{"role": "user", "content": "chuyển ví cho Bố chút tiền nhé"}]
    calls = [{"name": "ewallet.transfer", "arguments": {"contact_id": "c_01", "amount": 100000}}]
    guarded = validate_and_guard_calls(calls, catalog, msgs, "chuyển ví cho Bố chút tiền nhé")
    assert len(guarded) == 1
    assert guarded[0]["name"] == "sys.ask_missing_info"
    assert guarded[0]["arguments"] == {"tool_name": "ewallet.transfer", "params": ["amount"]}

    # Case có số ở câu khác (báo thức 7 giờ sáng) nhưng chuyển tiền vẫn mơ hồ
    q2 = "Đặt báo thức 7 giờ sáng mai cho em, với lại chuyển ví cho Bố chút tiền nhé."
    guarded2 = validate_and_guard_calls(calls, catalog, msgs, q2)
    assert len(guarded2) == 1
    assert guarded2[0]["name"] == "sys.ask_missing_info"
    print("test_vague_money_guard passed!")


def test_smart_home_hygiene():
    catalog = {
        "smart_home.get_status": {"kind": "read"},
        "smart_home.light_set": {"kind": "write"}
    }
    # Test English room mapping
    calls = [{"name": "smart_home.get_status", "arguments": {"room": "living room"}}]
    guarded = validate_and_guard_calls(calls, catalog, [], "Bật đèn phòng khách")
    assert guarded[0]["arguments"]["room"] == "phòng khách"

    # Test auto power on when setting brightness
    calls_light = [{"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "brightness": 30}}]
    guarded_light = validate_and_guard_calls(calls_light, catalog, [], "Bật đèn phòng khách tầm 30 phần trăm")
    assert guarded_light[0]["arguments"]["power"] == "on"
    assert guarded_light[0]["arguments"]["brightness"] == 30
    print("test_smart_home_hygiene passed!")


def test_param_hygiene():
    catalog = {"reminders.create": {"kind": "write"}}
    msgs = [{"role": "user", "content": "nhắc tôi đi mua vàng lúc 9 giờ"}]
    calls = [{"name": "reminders.create", "arguments": {"content": "đi mua vàng", "time": "2026-11-09T09:00:00+07:00", "repeat": "once"}}]
    guarded = validate_and_guard_calls(calls, catalog, msgs, "nhắc tôi đi mua vàng lúc 9 giờ")
    assert len(guarded) == 1
    assert "repeat" not in guarded[0]["arguments"]
    print("test_param_hygiene passed!")


if __name__ == "__main__":
    test_contact_ambiguity_guard()
    test_bike_ambiguity_guard()
    test_vague_money_guard()
    test_param_hygiene()
    test_smart_home_hygiene()
    print("\nALL VALIDATOR UNIT TESTS PASSED!")
