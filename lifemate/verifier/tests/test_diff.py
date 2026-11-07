"""Test diff.py. Chạy từ gốc repo: python -m pytest lifemate/verifier/tests -q
(không có pytest: python lifemate/verifier/tests/test_diff.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diff import (LOI, NGHI_NGO, Action, actions_from_item, actions_from_msgs,  # noqa: E402
                  diff, suggest, to_text, verdict)

CAT = {
    "bank.search_contact": {"kind": "read", "parameters": {"properties": {"name": {}}}},
    "bank.transfer": {"kind": "write", "parameters": {"properties": {
        "contact_id": {"x-ref": "bank.contacts"}, "amount": {}}}},
    "home.turn_off": {"kind": "write", "parameters": {"properties": {"device_id": {"x-ref": "home.devices"}}}},
    "home.turn_off_many": {"kind": "write", "parameters": {"properties": {
        "device_ids": {"type": "array", "items": {"x-ref": "home.devices"}}}}},
    "tv.volume_up": {"kind": "write", "parameters": {"properties": {}}},
    "sys.ask_missing_info": {"kind": "write", "parameters": {"properties": {"tool_name": {}, "params": {}}}},
}
SEARCH = Action("bank.search_contact", {"name": "Lan"}, {"contacts": [{"id": "C12", "name": "Lan"}]})
TRANSFER = Action("bank.transfer", {"contact_id": "C12", "amount": 200000}, {"ok": True})


def call(tool, /, **args):
    return {"name": tool, "arguments": args}


def kinds(ms):
    return [(m.kind, m.severity) for m in ms]


def test_plan_dung_khong_lech():
    ms = diff([call("bank.transfer", contact_id="C12", amount=200000)], [SEARCH], CAT)
    assert ms == [] and verdict(ms)["label"] == "CORRECT"


def test_id_bia():
    ms = diff([call("bank.transfer", contact_id="C99", amount=200000)], [SEARCH], CAT)
    assert kinds(ms) == [("UNSEEN_ID", LOI)] and ms[0].params == ["contact_id"]
    assert verdict(ms)["label"] == "WRONG"
    assert suggest(ms, [call("bank.transfer", contact_id="C99", amount=200000)])["ask"] == ("bank.transfer", ["contact_id"])


def test_id_user_tu_noi_khong_tinh_la_bia():
    ms = diff([call("home.turn_off", device_id="D7")], [], CAT, texts=["tắt thiết bị D7 giúp tôi"])
    assert ms == []


def test_id_khop_ca_tu_khong_khop_mot_phan():
    a = Action("home.list", {}, {"devices": [{"id": "D12"}]})
    assert kinds(diff([call("home.turn_off", device_id="D1")], [a], CAT)) == [("UNSEEN_ID", LOI)]


def test_mang_id():
    a = Action("home.list", {}, {"devices": [{"id": "D1"}, {"id": "D2"}]})
    assert diff([call("home.turn_off_many", device_ids=["D1", "D2"])], [a], CAT) == []
    assert kinds(diff([call("home.turn_off_many", device_ids=["D1", "D9"])], [a], CAT)) == [("UNSEEN_ID", LOI)]


def test_lap_write_cung_luot():
    ms = diff([call("bank.transfer", contact_id="C12", amount=200000)], [SEARCH, TRANSFER], CAT)
    assert kinds(ms) == [("DUP_WRITE", LOI)]


def test_lap_write_khong_tham_so_chi_nghi_ngo():
    ms = diff([call("tv.volume_up")], [Action("tv.volume_up", {}, {"ok": True})], CAT)
    assert kinds(ms) == [("DUP_WRITE", NGHI_NGO)] and verdict(ms)["label"] == "CORRECT"
    assert verdict(ms, strict=True)["label"] == "WRONG"


def test_lap_write_luot_truoc_chi_nghi_ngo():
    prev = Action("bank.transfer", {"contact_id": "C12", "amount": 200000}, {"ok": True}, current_turn=False)
    ms = diff([call("bank.transfer", contact_id="C12", amount=200000)], [SEARCH, prev], CAT)
    assert kinds(ms) == [("DUP_WRITE_PREV", NGHI_NGO)]


def test_lap_read_va_trong_cung_plan():
    ms = diff([call("bank.search_contact", name="Lan")], [SEARCH], CAT)
    assert kinds(ms) == [("DUP_READ", LOI)]
    plan = [call("bank.search_contact", name="An"), call("bank.search_contact", name="An")]
    ms = diff(plan, [], CAT)
    assert kinds(ms) == [("DUP_READ", LOI)] and ms[0].call_index == 1
    assert suggest(ms, plan) == {"drop": [1], "ask": None}


def test_tool_la():
    ms = diff([call("food.order", item="phở")], [], CAT)
    assert kinds(ms) == [("UNKNOWN_TOOL", LOI)]
    ms = diff([call("bank.transfer", contact_id="C12", amount=1)], [SEARCH], CAT, tools={"bank.search_contact"})
    assert kinds(ms) == [("UNKNOWN_TOOL", LOI)]


def test_call_sau_ask():
    plan = [call("sys.ask_missing_info", tool_name="bank.transfer", params=["amount"]),
            call("bank.transfer", contact_id="C12", amount=1)]
    ms = diff(plan, [SEARCH], CAT)
    assert kinds(ms) == [("CALL_AFTER_ASK", LOI)] and ms[0].call_index == 1
    asked = Action("sys.ask_missing_info", {"tool_name": "bank.transfer", "params": ["amount"]}, {"ok": True})
    assert kinds(diff([call("bank.search_contact", name="An")], [asked], CAT)) == [("CALL_AFTER_ASK", LOI)]


def test_hoi_lai_tham_so_da_biet():
    ms = diff([call("sys.ask_missing_info", tool_name="bank.transfer", params=["amount", "note"])], [SEARCH, TRANSFER], CAT)
    assert kinds(ms) == [("ASK_KNOWN_PARAM", NGHI_NGO)] and ms[0].params == ["amount"]


def test_actions_from_item():
    item = {"history": [
        {"role": "user", "audio": "a.wav"},
        {"role": "assistant", "tool_calls": [{"name": "bank.search_contact", "arguments": {"name": "Lan"}}],
         "tool_results": [{"contacts": [{"id": "C12"}]}], "text": "Đã tìm"},
        {"role": "assistant", "tool_calls": [{"name": "tv.volume_up", "arguments": "{}"}]},
    ]}
    acts = actions_from_item(item)
    assert [(a.name, a.arguments, a.current_turn) for a in acts] == [
        ("bank.search_contact", {"name": "Lan"}, False), ("tv.volume_up", {}, False)]
    assert acts[0].result == {"contacts": [{"id": "C12"}]} and acts[1].result is None


def test_actions_from_msgs_bo_call_bi_chan_va_chia_luot():
    def am(*calls):
        return {"role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]}
    msgs = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "tìm Lan"},
        am(("bank.search_contact", {"name": "Lan"})),
        {"role": "tool", "content": '{"contacts": [{"id": "C12"}]}'},
        {"role": "user", "content": "chuyển Lan 200 nghìn"},
        am(("bank.transfer", {"contact_id": "C99", "amount": 200000}), ("bank.transfer", {"contact_id": "C12", "amount": 200000})),
        {"role": "tool", "content": '{"ok": false, "error": "...", "note": "Chưa thực thi. Gọi tool đọc để lấy ID"}'},
        {"role": "tool", "content": '{"ok": true}'},
    ]
    acts = actions_from_msgs(msgs)
    assert [(a.name, a.arguments.get("contact_id"), a.current_turn) for a in acts] == [
        ("bank.search_contact", None, False), ("bank.transfer", "C12", True)]


def test_to_text():
    s = to_text([SEARCH], [call("bank.transfer", contact_id="C12", amount=200000)])
    assert s.splitlines()[0] == "Các action đã thực hiện:"
    assert 'bank.transfer(contact_id="C12", amount=200000)' in s and "C12" in s.splitlines()[1]
    assert "(chưa có)" in to_text([], [call("tv.volume_up")])


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
    print(f"{len(fns)} test pass")
