"""Test data.py và analyze_errors.py trên dữ liệu giả (không cần dữ liệu BTC).
Chạy: python lifemate/verifier/tests/test_data_analyze.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lifemate.verifier.analyze_errors import analyze_sample, summarize  # noqa: E402
from lifemate.verifier.data import (_extra_value, _shift_iso, examples_from_pred, make_examples)  # noqa: E402

CAT = {
    "bank.search_contact": {"kind": "read", "parameters": {"properties": {"name": {}}}},
    "bank.get_balance": {"kind": "read", "parameters": {"properties": {}}},
    "bank.transfer": {"kind": "write", "parameters": {"required": ["contact_id", "amount"], "properties": {
        "contact_id": {"x-ref": "bank.contacts"}, "amount": {"type": "integer"}, "note": {"type": "string"},
        "channel": {"type": "string", "enum": ["app", "atm"]}}}},
    "bank.pay_bill": {"kind": "write", "parameters": {"properties": {"bill_id": {"x-ref": "bank.bills"}}}},
    "sys.ask_missing_info": {"kind": "meta", "parameters": {"properties": {"tool_name": {}, "params": {}}}},
}
ITEM = {"id": "train_9", "audio_text": "chuyển cho Lan hai trăm nghìn ghi chú ăn trưa", "history": [],
        "label": {"tool_calls": [
            {"name": "bank.search_contact", "arguments": {"name": "Lan"}},
            {"name": "bank.transfer", "arguments": {"contact_id": "C12", "amount": 200000, "note": "ăn trưa"}},
        ]}}
RESULTS = [{"contacts": [{"id": "C12", "name": "Lan"}]}, {"ok": True}]


def exact(c, g, catalog):
    return c["name"] == g["name"] and (c.get("arguments") or {}) == (g.get("arguments") or {})


def test_make_examples_khong_query():
    rows = make_examples(ITEM, RESULTS, CAT, random.Random(0), with_query=False, neg_per_pos=5)
    kinds = [(r["step"], r["kind"], r["label"]) for r in rows]
    assert (0, "dung", 1) in kinds and (1, "dung", 1) in kinds
    assert (1, "dup_read", 0) in kinds and (1, "unseen_id", 0) in kinds
    assert not any(k in ("swap_tool", "change_value", "drop_arg", "ask_instead") for _, k, _ in kinds)
    bad = next(r for r in rows if r["kind"] == "unseen_id")
    assert "C12" in bad["text"] and 'contact_id="C12"' not in bad["text"].split("Kế hoạch mới:")[1]
    assert all(not r["text"].startswith("Yêu cầu") for r in rows)


def test_make_examples_co_query():
    rows = make_examples(ITEM, RESULTS, CAT, random.Random(0), with_query=True, neg_per_pos=10)
    kinds = {r["kind"] for r in rows if r["step"] == 1}
    assert {"dung", "dup_read", "unseen_id", "swap_tool", "change_value", "drop_arg", "add_arg", "ask_instead"} <= kinds
    assert all(r["text"].startswith("Yêu cầu mới nhất của người dùng: chuyển cho Lan") for r in rows)
    drop = next(r for r in rows if r["kind"] == "drop_arg")
    assert "note=" not in drop["text"].split("Kế hoạch mới:")[1]  # chỉ bỏ tham số không bắt buộc
    add = next(r for r in rows if r["kind"] == "add_arg")
    assert 'channel="app"' in add["text"] or 'channel="atm"' in add["text"]


def test_extra_value():
    rng = random.Random(0)
    assert _extra_value({"type": "string", "enum": ["a"]}, "", "", rng) == "a"
    assert _extra_value({"type": "string", "x-ref": "x"}, "q r", "", rng) is None
    v = _extra_value({"type": "string"}, "chuyển cho Lan hai trăm nghìn", "", rng)
    assert v and v in "chuyển cho Lan hai trăm nghìn" and len(v.split()) >= 2
    assert _extra_value({"type": "string", "format": "date-time"}, "", 'x "2026-11-19T09:30:00+07:00" y', rng) == "2026-11-19T09:30:00+07:00"
    assert _extra_value({"type": "string", "format": "date"}, "", "2026-11-19T09:30:00+07:00", rng) == "2026-11-19"
    assert _extra_value({"type": "string"}, "từ thứ hai tới thứ sáu", "", rng, "date_to") is None


def test_neg_per_pos_va_after_ask():
    item = {**ITEM, "label": {"tool_calls": [ITEM["label"]["tool_calls"][0],
                                             {"name": "sys.ask_missing_info", "arguments": {"tool_name": "bank.transfer", "params": ["amount"]}}]}}
    rows = make_examples(item, RESULTS, CAT, random.Random(1), with_query=False, neg_per_pos=1)
    assert sum(r["label"] == 0 and r["step"] == 1 for r in rows) <= 1
    assert rows[-1]["kind"] == "after_ask" and rows[-1]["label"] == 0


def test_shift_iso():
    assert _shift_iso("2026-11-19T09:30:00+07:00") == "2026-11-20T09:30:00+07:00"
    assert _shift_iso("2026-11-19") == "2026-11-20" and _shift_iso("09:30") == "10:30"
    assert _shift_iso("phở bò") is None


def test_examples_from_pred():
    pred = [ITEM["label"]["tool_calls"][0], {"name": "bank.transfer", "arguments": {"contact_id": "C12", "amount": 500000}}]
    rows = examples_from_pred(ITEM, pred, RESULTS, CAT, exact)
    assert [(r["kind"], r["label"]) for r in rows] == [("that_dung", 1), ("that_sai", 0)]


def test_analyze_sample_va_summarize():
    pred = [ITEM["label"]["tool_calls"][0],
            {"name": "bank.transfer", "arguments": {"contact_id": "C99", "amount": 200000, "note": "ăn trưa"}},
            {"name": "bank.transfer", "arguments": {"contact_id": "C12", "amount": 500000, "note": "ăn trưa"}}]
    s = analyze_sample(ITEM, pred, RESULTS, CAT, exact)
    assert [c["group"] for c in s["calls"]] == ["dung", "diff_bat_duoc", "can_query"]
    assert s["calls"][1]["flags"] == ["UNSEEN_ID"] and s["missing"] == [{"name": "bank.transfer", "group": "thieu_call"}]
    t = summarize([s])
    assert t["n_mau_sai"] == 1 and t["call_thua_sai"] == 2 and t["ti_le_diff_bat_trong_call_sai"] == 0.5
    assert t["ti_le_bao_nham_tren_call_dung"] == 0.0


def test_analyze_hoi_thua():
    pred = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "bank.transfer", "params": ["amount"]}}]
    s = analyze_sample(ITEM, pred, [None], CAT, exact)
    assert s["calls"][0]["group"] == "hoi_thua" and s["kind"] == "act"


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
    print(f"{len(fns)} test pass")
