"""Feedback text cho reflection LM của GEPA: nói rõ sai ở đâu và đáp án đúng là gì (ngắn, đọc được bởi Qwen 4B)."""
import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

from eval.scorer import ASK, task_kind

HEADERS = {
    "FALSE_ACT": "SAI: yêu cầu không cần làm gì (người dùng tự hủy, chỉ trò chuyện, hoặc không app nào xử lý được) nhưng đã gọi tool.",
    "MISSED_ASK": f"SAI: thiếu hoặc mơ hồ thông tin nên phải gọi {ASK} rồi dừng, nhưng đã tự đoán và gọi tool thường.",
    "FALSE_ASK": f"SAI: đã đủ thông tin để làm nhưng lại gọi {ASK} (hỏi lại thừa).",
    "EXTRA_CALL": "SAI: gọi thừa tool (một call ghi thừa làm lệch trạng thái, mất cả mẫu).",
    "MISSING_CALL": "SAI: thiếu tool call so với đáp án.",
    "WRONG_TOOL_OR_ARGS": "SAI: đúng số call nhưng sai tên tool hoặc sai tham số.",
    "HASH_MISMATCH": "SAI: trạng thái cuối lệch đáp án nên mẫu bị 0 điểm.",
}


def fmt(c: Dict[str, Any], n: int = 90) -> str:
    s = f"{c['name']}({json.dumps(c.get('arguments') or {}, ensure_ascii=False)})"
    return s if len(s) <= n * 2 else s[:n * 2] + "…"


def arg_diff(c: Dict[str, Any], g: Dict[str, Any]) -> List[str]:
    """Khác biệt tham số giữa call đã gọi c và đáp án g (cùng tên tool)."""
    ca, ga = c.get("arguments") or {}, g.get("arguments") or {}
    out = [f"thiếu tham số {k}" for k in ga if k not in ca] + [f"thừa tham số {k}" for k in ca if k not in ga]
    out += [f"{k}: gọi {json.dumps(ca[k], ensure_ascii=False)}, đúng {json.dumps(ga[k], ensure_ascii=False)}"
            for k in ga if k in ca and ca[k] != ga[k]]
    return out


def text_diff(raw: str, mapped: str, gold: str, n: int = 8) -> str:
    """Feedback cho Mapper: so bản ASR, câu sau Mapper và audio_text chuẩn theo từ (không cần đúng thứ tự)."""
    w = lambda s: re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())
    r, m, g = w(raw), set(w(mapped)), w(gold)
    show = lambda xs: ", ".join(list(dict.fromkeys(xs))[:n]) or "không"
    lines = [f"ASR: {raw}", f"Mapper: {mapped}", f"Câu đúng: {gold}"]
    lines.append("Mapper sửa đúng: " + show([x for x in g if x not in r and x in m]))
    lines.append("Mapper làm mất từ vốn đúng: " + show([x for x in g if x in r and x not in m]))
    lines.append("Còn sai chưa sửa: " + show([x for x in g if x not in r and x not in m]))
    return "\n".join(lines)


def explain(G: List[Dict[str, Any]], calls: List[Dict[str, Any]], err: Optional[str], pairs, extra, missing) -> str:
    """G: đáp án, calls: đã gọi, err/pairs/extra/missing: kết quả của scorer.score_sample và match_calls."""
    if err is None:
        return "Đúng."
    lines = [HEADERS[err], "Đáp án: " + ("không gọi tool nào" if not G else "; ".join(map(fmt, G)))]
    lines.append("Đã gọi: " + ("không gọi tool nào" if not calls else "; ".join(map(fmt, calls))))
    if err != "FALSE_ACT":
        for g in missing:  # gold chưa được ghép: nếu đã gọi cùng tên tool thì chỉ ra tham số sai
            c = next((e for e in extra if e["name"] == g["name"]), None)
            if c:
                extra = [e for e in extra if e is not c]
                lines.append(f"{g['name']}: " + "; ".join(arg_diff(c, g)))
            else:
                lines.append(f"Thiếu: {fmt(g)}")
        lines += [f"Thừa: {fmt(c)}" for c in extra]
        if err == "HASH_MISMATCH" and not missing and not extra:
            lines.append("Các call khớp đáp án nhưng hash vẫn lệch: kiểm tra thứ tự call hoặc tham số cần khớp chính xác.")
    return "\n".join(lines)
