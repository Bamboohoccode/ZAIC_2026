"""Hợp đồng dữ liệu giữa các module. Chỉ sửa file này sau khi báo cả nhóm. Xem PROJECT_OVERVIEW.md mục 4."""

# Phác thảo, chưa phải code chạy được
ASROut = {"turns": [{"role": "user", "nbest": [{"text": str, "score": float}]}]}
Plan = {"action": "tool_call|ask_user|cannot_answer",
        "calls": [{"tool": str, "args": dict}],
        "missing": {"tool": str, "params": [str]}}  # khi ask_user
Verdict = {"label": "ACCEPT|REJECT|ASK", "confidence": float,
           "reason": str}  # reason do code diff sinh
