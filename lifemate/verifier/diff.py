"""Diff bằng code giữa plan của Planner và các action đã lưu (PROJECT_OVERVIEW.md mục 5.3g).

Ba việc:
  1. Trích action đã lưu: từ `history` của một mẫu dataset, hoặc từ `msgs` của pipeline.
  2. `diff(plan, actions, catalog)` liệt kê chỗ lệch, mỗi chỗ có mức `loi` (gần như chắc chắn sai)
     hoặc `nghi_ngo` (có thể hợp lệ, để Laya hoặc ngưỡng quyết).
  3. Từ chỗ lệch: nhãn CORRECT/WRONG (`verdict`), gợi ý bỏ call hay hỏi user (`suggest`),
     và chuỗi văn bản đưa vào Laya (`to_text`).

Các loại lệch:
  UNKNOWN_TOOL     tool không có trong catalog / không thuộc app đã cài              loi
  CALL_AFTER_ASK   có call sau sys.ask_missing_info (ask phải là call cuối)          loi
  DUP_READ         gọi lại tool đọc cùng tham số đã chạy trong lượt này             loi
  DUP_WRITE        gọi lại tool ghi cùng tham số đã chạy trong lượt này             loi (nghi_ngo nếu không tham số)
  DUP_WRITE_PREV   gọi lại tool ghi cùng tham số đã chạy ở lượt trước               nghi_ngo
  UNSEEN_ID        tham số x-ref (ID) chưa xuất hiện trong action, kết quả tool,
                   hay lời user: ID bịa, chạy trên ToolEnv sẽ làm lệch hash          loi
  ASK_KNOWN_PARAM  hỏi user tham số mà action đã lưu đã có giá trị                  nghi_ngo

Không kiểm schema (thiếu tham số bắt buộc, sai kiểu): việc của validator (Kiệt, Vương).
Thuần Python, không phụ thuộc thư viện ngoài, không đọc state.json.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

ASK_TOOL = "sys.ask_missing_info"
LOI, NGHI_NGO = "loi", "nghi_ngo"
# pipeline.run_agent trả các thông báo này cho call bị chặn trước ToolEnv: call đó chưa chạy
NOT_RUN_MARKERS = ("Chưa thực thi", "Đã gọi tool đọc này")
# loại lệch nên sửa bằng cách bỏ call, không cần hỏi user
DROP_KINDS = {"CALL_AFTER_ASK", "DUP_READ"}


@dataclass
class Action:
    name: str
    arguments: dict = field(default_factory=dict)
    result: object = None
    current_turn: bool = True  # False: chạy ở lượt user trước


@dataclass
class Mismatch:
    kind: str
    severity: str  # LOI | NGHI_NGO
    call_index: int  # vị trí call trong plan
    tool: str
    params: list
    message: str


# ---------------------------------------------------------------- chuẩn hoá call

def _args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return {}
    return a if isinstance(a, dict) else {}


def normalize_call(c):
    """Nhận {"name","arguments"} (pipeline, dataset), {"tool","args"} (contracts.Plan) hoặc
    {"function": {...}} (định dạng OpenAI trong msgs). Trả (name, arguments)."""
    c = c.get("function", c)
    return c.get("name") or c.get("tool") or "", _args(c.get("arguments", c.get("args")))


def call_key(name, args):
    return name + json.dumps(args, sort_keys=True, ensure_ascii=False)


def _not_run(result):
    s = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return any(m in s for m in NOT_RUN_MARKERS)


# ---------------------------------------------------------------- trích action đã lưu

def actions_from_item(item):
    """Action trong `history` của một mẫu dataset. Tất cả thuộc lượt trước (current_turn=False)."""
    out = []
    for h in item.get("history", []):
        calls = h.get("tool_calls") or []
        results = h.get("tool_results") or [None] * len(calls)
        for c, r in zip(calls, results):
            name, args = normalize_call(c)
            out.append(Action(name, args, r, current_turn=False))
    return out


def actions_from_msgs(msgs):
    """Action trong `msgs` của pipeline (assistant.tool_calls, theo sau là các message tool cùng thứ tự).
    Call sau message user cuối là lượt hiện tại. Bỏ call bị pipeline chặn (chưa chạy trên ToolEnv)."""
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
    out, i = [], 0
    while i < len(msgs):
        m = msgs[i]
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not calls:
            i += 1
            continue
        j = i + 1
        results = []
        while j < len(msgs) and msgs[j].get("role") == "tool":
            results.append(msgs[j].get("content"))
            j += 1
        results += [None] * (len(calls) - len(results))
        for c, r in zip(calls, results):
            if r is not None and _not_run(r):
                continue
            name, args = normalize_call(c)
            out.append(Action(name, args, r, current_turn=i > last_user))
        i = j
    return out


# ---------------------------------------------------------------- diff

def _blob(actions, texts):
    parts = list(texts)
    for a in actions:
        parts.append(json.dumps(a.arguments, ensure_ascii=False))
        parts.append(a.result if isinstance(a.result, str) else json.dumps(a.result, ensure_ascii=False))
    return "\n".join(parts)


def _seen(value, blob):
    return re.search(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", blob) is not None


def _unseen_ids(args, spec, blob):
    """Tham số x-ref (chuỗi hoặc mảng chuỗi) có giá trị chưa từng xuất hiện. Cùng quy tắc với
    pipeline.unseen_ids, mở rộng cho mảng ID."""
    props = (spec or {}).get("parameters", {}).get("properties", {})
    bad = []
    for k, v in args.items():
        p = props.get(k, {})
        if p.get("x-ref") and isinstance(v, str):
            vals = [v]
        elif (p.get("x-ref") or p.get("items", {}).get("x-ref")) and isinstance(v, list):
            vals = [x for x in v if isinstance(x, str)]
        else:
            continue
        if any(not _seen(x, blob) for x in vals):
            bad.append(k)
    return bad


def _asked_params(args):
    p = args.get("params")
    if isinstance(p, dict):
        return list(p)
    if isinstance(p, list):
        return [x for x in p if isinstance(x, str)]
    return [p] if isinstance(p, str) else []


def diff(plan, actions, catalog, texts=(), tools=None):
    """So plan (list call) với action đã lưu.

    plan:    list call của bước hiện tại, định dạng bất kỳ mà normalize_call nhận.
    actions: list Action (actions_from_item + actions_from_msgs).
    catalog: {tool_name: spec} như pipeline (spec có "kind", "parameters").
    texts:   lời user (đã qua ASR/mapper). ID user tự nói ra thì không tính là bịa.
    tools:   tên tool của app đã cài trong mẫu; None = không kiểm.
    """
    plan = [normalize_call(c) for c in plan]
    blob = _blob(actions, texts)
    cur = {call_key(a.name, a.arguments) for a in actions if a.current_turn}
    prev = {call_key(a.name, a.arguments) for a in actions if not a.current_turn}
    asked = any(a.name == ASK_TOOL for a in actions if a.current_turn)
    out = []

    def add(kind, sev, i, name, params, msg):
        out.append(Mismatch(kind, sev, i, name, list(params), msg))

    for i, (name, args) in enumerate(plan):
        spec = catalog.get(name)
        if name != ASK_TOOL and (spec is None or (tools is not None and name not in tools)):
            add("UNKNOWN_TOOL", LOI, i, name, [], f"Tool {name} không có trong các app đã cài.")
            continue
        if asked or any(n == ASK_TOOL for n, _ in plan[:i]):
            add("CALL_AFTER_ASK", LOI, i, name, [], f"{name} nằm sau {ASK_TOOL}, phải dừng sau khi hỏi user.")
        key = call_key(name, args)
        kind = (spec or {}).get("kind")
        in_plan = key in {call_key(n, a) for n, a in plan[:i]}
        if key in cur or in_plan:
            if kind == "read":
                add("DUP_READ", LOI, i, name, list(args), f"{name} đã đọc với cùng tham số, dùng lại kết quả trước.")
            elif name != ASK_TOOL:
                add("DUP_WRITE", LOI if args else NGHI_NGO, i, name, list(args),
                    f"{name} đã chạy với cùng tham số trong lượt này.")
        elif key in prev and kind == "write":
            add("DUP_WRITE_PREV", NGHI_NGO, i, name, list(args),
                f"{name} đã chạy với cùng tham số ở lượt trước, chỉ đúng nếu user yêu cầu làm lại.")
        bad = _unseen_ids(args, spec, blob)
        if bad:
            add("UNSEEN_ID", LOI, i, name, bad,
                f"{name}: giá trị {', '.join(f'{k}={args[k]!r}' for k in bad)} chưa xuất hiện trong kết quả tool hay lời user.")
        if name == ASK_TOOL:
            target = args.get("tool_name")
            known = [p for p in _asked_params(args)
                     if any(a.name == target and p in a.arguments for a in actions)
                     or any(n == target and p in a for n, a in plan[:i])]
            if known:
                add("ASK_KNOWN_PARAM", NGHI_NGO, i, target or name, known,
                    f"Hỏi lại {', '.join(known)} của {target} dù action đã lưu đã có giá trị.")
    return out


# ---------------------------------------------------------------- dùng kết quả diff

def verdict(mismatches, strict=False):
    """Nhãn từ diff thuần luật: WRONG nếu có lệch mức loi (strict: mọi mức). Dùng làm Verifier luật
    (mốc so sánh, phương án dự phòng 5.3d) và làm trường diff của contracts.Verdict."""
    bad = [m for m in mismatches if strict or m.severity == LOI]
    return {"label": "WRONG" if bad else "CORRECT", "diff": [asdict(m) for m in mismatches]}


def suggest(mismatches, plan):
    """Gợi ý xử lý khi WRONG.
    drop: chỉ số call nên bỏ (DUP_READ, CALL_AFTER_ASK) - bỏ là đủ, không cần làm phiền user.
    ask:  (tool_name, params) cho sys.ask_missing_info lấy từ lệch mức loi đầu tiên còn lại.
          Đây là đề xuất cho câu hỏi 9g (tool của call lệch + tham số diff chỉ ra), CHƯA chốt với nhóm.
    """
    plan = [normalize_call(c) for c in plan]
    drop = sorted({m.call_index for m in mismatches if m.kind in DROP_KINDS and m.severity == LOI})
    rest = [m for m in mismatches if m.severity == LOI and m.kind not in DROP_KINDS and m.kind != "UNKNOWN_TOOL"]
    ask = None
    if rest:
        m = rest[0]
        ask = (m.tool, m.params or list(plan[m.call_index][1]))
    return {"drop": drop, "ask": ask}


def to_text(actions, plan, limit=600):
    """Chuỗi hoá gọn action đã lưu + plan làm input cho Laya (cùng khuôn với smoke_test_laya.py)."""
    def show(name, args, result=None):
        s = f"{name}({', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in args.items())})"
        if result is not None:
            r = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            s += " -> " + (r if len(r) <= limit else r[:limit] + "…")
        return s
    lines = ["Các action đã thực hiện:"]
    lines += [f"{i}. {'' if a.current_turn else '[lượt trước] '}{show(a.name, a.arguments, a.result)}"
              for i, a in enumerate(actions, 1)] or ["(chưa có)"]
    lines.append("Kế hoạch mới:")
    lines += [f"{i}. {show(*normalize_call(c))}" for i, c in enumerate(plan, 1)]
    return "\n".join(lines)
