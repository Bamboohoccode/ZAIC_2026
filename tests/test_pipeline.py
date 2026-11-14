"""python -m tests.test_pipeline                                             # unit asserts, no GPU
python -m tests.test_pipeline --root DATA --split train --sub submission.jsonl   # score vs train labels (RULES §7)
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # gốc repo, để import lifemate khi chạy trực tiếp

from lifemate.pipeline import date_table, time_hints, parse_calls, parse_verdict, run_agent, unseen_ids


def unit():
    assert parse_calls('ok <tool_call>{"name": "a.b", "arguments": {"x": 1}}</tool_call>'
                       '<tool_call>{"name": "c.d", "arguments": "{\\"y\\": 2}"}</tool_call><tool_call>bad</tool_call>') == \
        [{"name": "a.b", "arguments": {"x": 1}}, {"name": "c.d", "arguments": {"y": 2}}]

    # Think-block filtering (both full <think>...</think> and prompt-opened ... </think>) + Qwen3.5 XML fallback
    assert parse_calls('<think>nháp <tool_call>{"name": "wrong.tool", "arguments": {}}</tool_call></think>\n'
                       '<tool_call>{"name": "right.tool", "arguments": {"n": 2}}</tool_call>') == \
        [{"name": "right.tool", "arguments": {"n": 2}}]
    assert parse_calls('suy nghĩ với prompt mở sẵn think <tool_call>{"name": "skip.me", "arguments": {}}</tool_call>\n'
                       '</think>\n\n<tool_call>\n<function=banking.transfer>\n'
                       '<parameter=amount>\n50000\n</parameter>\n'
                       '<parameter=confirm>\ntrue\n</parameter>\n'
                       '<parameter=phone>\n0901234567\n</parameter>\n'
                       '<parameter=tags>\n["a", "b"]\n</parameter>\n'
                       '</function>\n</tool_call>') == \
        [{"name": "banking.transfer", "arguments": {"amount": 50000, "confirm": True, "phone": "0901234567", "tags": ["a", "b"]}}]

    assert parse_verdict('{"verdict": "ACCEPT"}') == (True, "")
    assert parse_verdict('```json\n{"verdict": "REJECT", "reason": "thiếu date"}\n```') == (False, "thiếu date")
    assert parse_verdict("REJECT vì sai tool")[0] is False
    assert parse_verdict("garbage")[0] is True

    cat = {"x.set": {"parameters": {"properties": {"device_id": {"x-ref": "x.devices"}, "power": {}}}}}
    msgs = [{"role": "tool", "content": '{"devices": [{"id": "light_1"}]}'}]
    assert unseen_ids({"name": "x.set", "arguments": {"device_id": "light_1", "power": "on"}}, cat, msgs) == []
    assert unseen_ids({"name": "x.set", "arguments": {"device_id": "light_2"}}, cat, msgs) == ["device_id"]
    assert unseen_ids({"name": "x.set", "arguments": {"device_id": "light"}}, cat, msgs) == ["device_id"]  # no prefix match

    t = date_table("2026-11-28T07:10:00+07:00")  # a Saturday
    assert "- 2026-11-28: thứ bảy tuần này (hôm nay, 07:10)" in t and "(hôm nay)" in date_table("2026-11-28") and "- 2026-11-29: chủ nhật tuần này (mai)" in t
    assert "- 2026-11-30: thứ hai tuần sau" in t
    assert "- 2026-12-08: thứ ba tuần sau nữa / tuần kia" in t and "- 2026-12-13: chủ nhật tuần sau nữa / tuần kia" in t
    assert "2026-12-14" not in t and "mốt / ngày mốt / ngày kia" in t
    for s, want in [("đặt báo thức 9am mai", '- "9am" = 09:00'), ("hẹn 8pm tối nay", '- "8pm" = 20:00'), ("họp 8:30pm", '- "8:30pm" = 20:30'),
                    ("12am", '- "12am" = 00:00'), ("12pm", '- "12pm" = 12:00'), ("báo thức 5am nhé à không 5:30am mai", '- "5:30am" = 05:30'),
                    ("bật đèn", ""), ("thứ sáu 9 giờ", ""), ("13pm", "")]:  # only am/pm forms; the day is the date table's job
        assert time_hints(s) == want, (s, time_hints(s))
    agent()
    print("unit ok")


def agent():
    class Env:
        def __init__(self):
            self.calls = []

        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            self.calls.append(args)
            return {"ok": True}

    class LLM:  # proposes one call, then finishes
        def __init__(self, dev):
            self.outs = [f'<tool_call>{{"name": "x.set", "arguments": {{"device_id": "{dev}"}}}}</tool_call>', "Xong."]

        def chat(self, convs, tools=None):
            return [self.outs.pop(0)]

    cat = {"x.set": {"kind": "write", "parameters": {"properties": {"device_id": {"x-ref": "x.devices"}}}}}
    seen = {"role": "tool", "content": '{"devices": [{"id": "light_1"}]}'}

    env, trace = Env(), []
    run_agent(env, LLM("ghost"), [{"role": "user", "content": "bật đèn"}], [], cat, trace, 0)
    assert env.calls == []  # invented ID is blocked, not executed

    env = Env()
    run_agent(env, LLM("light_1"), [seen, {"role": "user", "content": "bật đèn"}], [], cat, trace, 0)
    assert env.calls == [{"device_id": "light_1"}]

    env, trace = Env(), []
    run_agent(env, LLM("light_1"), [seen, {"role": "user", "content": "bật đèn"}], [], cat, trace, 0, deadline=0)
    assert env.calls == [] and trace == [{"error": "time limit"}]


def toks(s):
    return unicodedata.normalize("NFC", str(s)).lower().split()


def f1(a, b):
    a, b = toks(a), toks(b)
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    return 2 * common / (len(a) + len(b)) if common else 0.0


def match(c, g, catalog):
    if c["name"] != g["name"]:
        return False
    props = catalog.get(c["name"], {}).get("parameters", {}).get("properties", {})
    ca, ga = c.get("arguments") or {}, g.get("arguments") or {}
    if set(ca) != set(ga):
        return False
    return all(f1(ca[k], ga[k]) > 0.6 if props.get(k, {}).get("x-match") == "fuzzy" else ca[k] == ga[k] for k in ga)


def score(root, split, sub_path):
    """RULES §7. Gold hash = replay of label.tool_calls through zaic_tools (state.json is only read by the env)."""
    sys.path.insert(0, str(Path(root) / split))
    from zaic_tools import ToolEnv
    catalog = {t["name"]: t for app in json.load(open(Path(root) / split / f"{split}_tools_catalog.json"))["apps"] for t in app["tools"]}
    gold = {d["id"]: d for d in map(json.loads, open(Path(root) / split / f"{split}.jsonl", encoding="utf-8"))}
    total, by_kind = [], {}
    for s in map(json.loads, open(sub_path, encoding="utf-8")):
        G = gold[s["id"]]["label"]["tool_calls"]
        env = ToolEnv(split, s["id"], root=str(root))
        for g in G:
            env.call(g["name"], g["arguments"])
        if not G:
            sc = float(not s["tool_calls"])
        elif s["state_hash"] != env.get_state_hash():
            sc = 0.0
        else:
            left, m = list(G), 0
            for c in s["tool_calls"]:
                j = next((j for j, g in enumerate(left) if match(c, g, catalog)), None)
                if j is not None:
                    left.pop(j)
                    m += 1
            sc = 2 * m / (len(s["tool_calls"]) + len(G))
        kind = "empty" if not G else "ask" if any(g["name"] == "sys.ask_missing_info" for g in G) else "act"
        by_kind.setdefault(kind, []).append(sc)
        total.append(sc)
    print(f"score {sum(total) / len(total):.4f} over {len(total)}")
    for k, v in sorted(by_kind.items()):
        print(f"  {k}: {sum(v) / len(v):.4f} (n={len(v)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sub", default="submission.jsonl")
    a = ap.parse_args()
    unit()
    if a.root:
        score(a.root, a.split, a.sub)
