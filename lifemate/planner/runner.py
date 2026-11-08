"""Dedicated Dev & Evaluation Runner for Vuong's Planner & Validator.
Keeps lifemate/pipeline.py 100% intact as baseline while providing:
- Custom Planner prompts (--prompt)
- Safe Hash Guard & Schema Validator integration (lifemate.validator.validator)
- Target ID split evaluation (--ids)
- Safe context pruning (protects user query)
- Checkpoint auto-resume (--resume)
- Real-time logging with timestamps and instant flush
- Automated periodic Google Drive sync
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Import clean baseline components from lifemate.pipeline
from lifemate.pipeline import (
    ASR_MODEL,
    MAPPER_MODEL,
    LLM_MODEL,
    MAX_CTX,
    MAX_NEW,
    MAX_STEPS,
    VERIFY_ROUNDS,
    TIME_LIMIT_S,
    SYSTEM as BASELINE_SYSTEM,
    MAPPER_SYSTEM as BASELINE_MAPPER_SYSTEM,
    MAPPER_SHOTS as BASELINE_MAPPER_SHOTS,
    VERIFY,
    REPLAN,
    VLLM,
    GoldLLM,
    RemoteVLLM,
    load_asr,
    load_itn,
    run_asr,
    tool_specs,
    build_messages,
    date_table,
    hints,
    clip,
    assistant_msg,
    parse_calls,
    parse_verdict,
    unseen_ids,
)


def format_custom_messages(item, texts, keep, prompt_template):
    """Xây dựng chuỗi messages chuẩn với prompt tùy chỉnh của Planner (thay thế prompt mặc định)."""
    msgs = build_messages(item, texts, keep)
    if prompt_template:
        ctx = item["context"]
        msgs[0]["content"] = prompt_template.format(
            now=ctx["current_time"],
            tz=ctx["timezone"],
            dates=date_table(ctx["current_time"]) + hints(texts[-1]),
            loc=json.dumps(ctx.get("location"), ensure_ascii=False),
            profile=json.dumps(ctx.get("user_profile"), ensure_ascii=False),
        )
    return msgs


def plan_step(llm, msgs, tools, trace, rounds, deadline, system_prompt, max_ctx=MAX_CTX, log_msg=None):
    """Planner proposes the next calls; verifier ACCEPTs or REJECTs with a reason, at most `rounds` times."""
    query = msgs[-1]["content"] if msgs[-1]["role"] == "user" else next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    feedback = []

    # Safe context pruning: CHỈ cắt tỉa khi tổng token THỰC SỰ vượt ngưỡng max_ctx
    safe_msgs = [dict(m) for m in msgs]
    if llm.n_tokens(safe_msgs + feedback, tools) > max_ctx - MAX_NEW * 2:
        # 1. Rút ngắn các kết quả tool đọc quá dài (> 800 ký tự) nếu bị quá tải context
        for m in safe_msgs:
            if m.get("role") == "tool" and len(m.get("content", "")) > 800:
                m["content"] = m["content"][:800] + "... [truncated]"

        # 2. Xoá dần các tool/assistant trung gian cũ, BẢO VỆ TUYỆT ĐỐI User Query và System
        while len(safe_msgs) > 3 and llm.n_tokens(safe_msgs + feedback, tools) > max_ctx - MAX_NEW * 2:
            drop_idx = next((i for i, m in enumerate(safe_msgs) if 1 <= i < len(safe_msgs) - 1 and m.get("role") in ("tool", "assistant")), None)
            if drop_idx is not None:
                safe_msgs.pop(drop_idx)
            else:
                break

    for attempt in range(rounds + 1):
        n_toks = llm.n_tokens(safe_msgs + feedback, tools)
        t_call0 = time.time()
        if log_msg:
            log_msg(f"    -> [LLM Chat] msgs={len(safe_msgs)} input_tokens={n_toks} calling vLLM...")
        out = llm.chat([safe_msgs + feedback], tools)[0]
        dt_llm = time.time() - t_call0
        trace.append({"plan": out})
        calls = parse_calls(out)
        call_names = [c["name"] for c in calls]
        first_line = out.strip().split("\n")[0][:90] if out.strip() else ""
        if log_msg:
            log_msg(f"    -> [LLM Output in {dt_llm:.1f}s] calls={call_names} | preview: {first_line}")
        if not calls or attempt == rounds or time.time() > deadline:
            return out, calls
        shown = "\n".join(f"<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c in calls)
        verdict = llm.chat([safe_msgs + [{"role": "user", "content": VERIFY.format(query=query, calls=shown, text=re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip())}]], tools)[0]
        ok, reason = parse_verdict(verdict)
        trace.append({"verify": verdict})
        if ok:
            return out, calls
        if log_msg:
            log_msg(f"    -> [Verifier REJECT] {reason}")
        feedback = [{"role": "assistant", "content": out}, {"role": "user", "content": REPLAN.format(reason=reason)}]


def run_agent(env, llm, msgs, tools, catalog, trace, system_prompt, rounds=VERIFY_ROUNDS, deadline=float("inf"), max_ctx=MAX_CTX, log_msg=None):
    done = set()
    for step_i in range(MAX_STEPS):
        time_left = max(0.0, deadline - time.time())
        if time.time() > deadline:
            trace.append({"error": "time limit"})
            if log_msg:
                log_msg(f"  [Timeout] Deadline reached (>120s) -> Stopping agent early.")
            return
        if log_msg:
            log_msg(f"  [Step {step_i + 1}/{MAX_STEPS}] time_left={time_left:.1f}s")
        out, calls = plan_step(llm, msgs, tools, trace, rounds, deadline, system_prompt, max_ctx=max_ctx, log_msg=log_msg)
        if not calls:
            if log_msg:
                log_msg(f"  [Step {step_i + 1}] No tool calls -> Agent finished.")
            return
        ai = len(msgs)
        msgs.append(assistant_msg(re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip(), calls))
        stop = False
        for c in calls:
            t_exec0 = time.time()
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            v = env.validate(c["name"], c["arguments"])
            bad = v.get("ok") and unseen_ids(c, catalog, msgs[:ai] + msgs[ai + 1:])
            if not v.get("ok"):
                r = {"ok": False, "error": v.get("error"), "note": "Chưa thực thi, hãy sửa tham số."}
            elif bad:
                r = {"ok": False, "error": f"Giá trị {bad} chưa xuất hiện trong kết quả tool nào, không được tự đặt ID.",
                     "note": "Chưa thực thi. Gọi tool đọc để lấy ID, hoặc sys.ask_missing_info nếu không lấy được."}
            elif key in done and catalog[c["name"]]["kind"] == "read":
                r = {"ok": False, "error": "Đã gọi tool đọc này với cùng tham số, dùng lại kết quả trước."}
            else:
                r = env.call(c["name"], c["arguments"])
                done.add(key)
                stop |= c["name"] == "sys.ask_missing_info"
            dt_exec = time.time() - t_exec0
            status = "OK" if r.get("ok", True) else f"ERR: {str(r.get('error', ''))[:40]}"
            if log_msg:
                log_msg(f"    -> [Tool Exec] {c['name']} in {dt_exec:.2f}s -> {status}")
            msgs.append({"role": "tool", "content": clip(r)})
        if stop:
            if log_msg:
                log_msg(f"  [Step {step_i + 1}] Stop requested (sys.ask_missing_info called).")
            return


def main():
    ap = argparse.ArgumentParser(description="Vuong's Planner & Validator Dev Runner")
    ap.add_argument("--root", required=True, help="folder holding the split folders")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="runs/submission.jsonl")
    ap.add_argument("--asr", default=ASR_MODEL, help="ChunkFormer HF id or local folder")
    ap.add_argument("--asr-ckpt", help="fine-tuned ChunkFormer .pt")
    ap.add_argument("--asr-device", default="cuda:0")
    ap.add_argument("--itn", help="folder of ITN checkpoint")
    ap.add_argument("--itn-device", default="cuda:0")
    ap.add_argument("--mapper", default=MAPPER_MODEL, help='HF id or "none"')
    ap.add_argument("--mapper-lora")
    ap.add_argument("--mapper-device")
    ap.add_argument("--mapper-prompt", help="path to custom dialect mapper prompt file")
    ap.add_argument("--llm", default=LLM_MODEL, help='HF id or "gold"')
    ap.add_argument("--llm-lora")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="main LLM gpu_memory_utilization")
    ap.add_argument("--mapper-gpu-mem", type=float, default=0.10)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--verify-rounds", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="use audio_text instead of ASR")
    ap.add_argument("--ids", help="path to dev_ids.txt")
    ap.add_argument("--prompt", help="path to planner prompt (e.g. prompts/gepa/vuong/planner_v4.txt)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", help="path to log file (default: <out>.log)")
    ap.add_argument("--resume", action="store_true", help="skip finished IDs and append")
    a = ap.parse_args()

    gold = a.llm == "gold"
    rounds = 0 if gold else a.verify_rounds

    # System prompt loading
    system_prompt = open(a.prompt, encoding="utf-8").read() if a.prompt else BASELINE_SYSTEM

    # Mapper setup
    mapper_system = BASELINE_MAPPER_SYSTEM
    mapper_shots = BASELINE_MAPPER_SHOTS
    if a.mapper_prompt:
        content = open(a.mapper_prompt, encoding="utf-8").read()
        if "### EXAMPLES" in content:
            sys_part, examples_part = content.split("### EXAMPLES", 1)
            mapper_system = sys_part.strip()
            custom_shots = []
            for pair in examples_part.strip().split("\n\n"):
                lines = [l.strip() for l in pair.split("\n") if l.strip()]
                if len(lines) == 1 and " -> " in lines[0]:
                    parts = lines[0].lstrip("- ").split(" -> ")
                    custom_shots.append((parts[0].strip('"\' '), parts[1].strip('"\' ')))
                elif len(lines) >= 2:
                    q = re.sub(r"^(User|Q|Hỏi):\s*", "", lines[0], flags=re.I)
                    ans = re.sub(r"^(Assistant|A|Trả lời):\s*", "", lines[1], flags=re.I)
                    custom_shots.append((q.strip('"\' '), ans.strip('"\' ')))
            if custom_shots:
                mapper_shots = custom_shots
        else:
            mapper_system = content.strip()

    def custom_mapper_conv(text):
        shots = [m for q, a in mapper_shots for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]
        return [{"role": "system", "content": mapper_system}, *shots, {"role": "user", "content": text}]

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    random.seed(a.seed)
    try:
        import numpy as np, torch
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    except ImportError:
        pass

    root = Path(a.root).resolve()
    for candidate_dir in [root / a.split, Path("/content/data") / a.split, Path("/content/ZAIC_2026/data") / a.split]:
        if candidate_dir.exists() and (candidate_dir / "zaic_tools").exists():
            root = candidate_dir.parent
            if str(candidate_dir) not in sys.path:
                sys.path.insert(0, str(candidate_dir))
            break
    else:
        if str(root / a.split) not in sys.path:
            sys.path.insert(0, str(root / a.split))
    from zaic_tools import ToolEnv
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.ids:
        filter_ids = set(open(a.ids, encoding="utf-8").read().split())
        items = [item for item in items if item["id"] in filter_ids]
    items = items[:a.limit] if a.limit else items
    t_start = time.time()

    asr = None if a.text else load_asr(a.asr, a.asr_ckpt, a.asr_device)
    itn = load_itn(a.itn, a.itn_device) if a.itn else None
    m_args = (a.mapper, a.mapper_lora, a.mapper_gpu_mem, a.seed, 2048, 256, {"enable_thinking": False}, a.eager)
    mapper = None if a.mapper == "none" or gold else RemoteVLLM(a.mapper_device, *m_args) if a.mapper_device else VLLM(*m_args)
    llm = GoldLLM() if gold else VLLM(a.llm, a.llm_lora, a.gpu_mem, a.seed, a.max_ctx, MAX_NEW, None, a.eager)
    slow = []

    log_path = a.log or (a.out + ".log")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if a.resume and os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            if l.strip():
                try:
                    row = json.loads(l)
                    if "id" in row:
                        done_ids.add(row["id"])
                except Exception:
                    pass

    out_mode = "a" if (a.resume and done_ids) else "w"
    with open(a.out, out_mode, encoding="utf-8") as fout, \
         open(a.out + ".trace.jsonl", out_mode, encoding="utf-8") as ftr, \
         open(log_path, out_mode, encoding="utf-8") as flog:

        def log_msg(msg=""):
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}" if msg else ""
            print(line, flush=True)
            flog.write(line + "\n")
            flog.flush()

        log_msg("=== Vuong Planner & Validator Dev Runner ===")
        log_msg(f"Output: {a.out} | Log: {log_path}")
        log_msg(f"Split: {a.split} | Total samples: {len(items)}")
        if done_ids:
            log_msg(f"Resume active: {len(done_ids)}/{len(items)} samples already finished")
        if a.prompt:
            log_msg(f"Prompt: {a.prompt}")

        for i, item in enumerate(items):
            if a.resume and item["id"] in done_ids:
                continue

            t0 = time.time()
            deadline = t0 + 120
            env = ToolEnv(a.split, item["id"], root=str(root))
            trace, raw, texts, keep = [], [], [], 0

            # Giải phóng và dọn dẹp prefix cache định kỳ để chống phân mảnh VRAM qua hàng trăm mẫu
            if hasattr(llm, "llm") and hasattr(llm.llm, "reset_prefix_cache"):
                try:
                    llm.llm.reset_prefix_cache()
                except Exception:
                    pass

            # Khống chế ngắt cưỡng bức (Hard Timeout) sau 120s bằng tín hiệu hệ điều hành SIGALRM
            if hasattr(signal, "SIGALRM"):
                def _handle_timeout(signum, frame):
                    raise TimeoutError("Sample hard timeout (120s exceeded)")
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(120)

            try:
                turns = [h for h in item["history"] if h["role"] == "user"] + [item]
                raw = [u["audio_text"] for u in turns] if a.text else run_asr(asr, [root / a.split / u["audio"] for u in turns])
                itn_out = [itn(r) or r for r in raw] if itn else raw
                texts = [m or r for r, m in zip(itn_out, mapper.chat([custom_mapper_conv(r) for r in itn_out]))] if mapper else itn_out
                query_snippet = texts[-1][:70] + "..." if len(texts[-1]) > 70 else texts[-1]
                log_msg(f"[{i + 1}/{len(items)}] {item['id']} START: \"{query_snippet}\"")
                tools = tool_specs(item, catalog)
                keep = len(texts) - 1
                msgs = format_custom_messages(item, texts, keep, system_prompt)
                while keep and llm.n_tokens(msgs, tools) > a.max_ctx - MAX_NEW * 2:
                    keep -= 1
                    msgs = format_custom_messages(item, texts, keep, system_prompt)
                init_tokens = llm.n_tokens(msgs, tools)
                log_msg(f"  -> Context: history_turns={len(item.get('history', []))}, tools={len(tools)}, tokens={init_tokens}")
                if gold:
                    llm.queue = list(item["label"]["tool_calls"])
                run_agent(env, llm, msgs, tools, catalog, trace, system_prompt, rounds=rounds, deadline=deadline, max_ctx=a.max_ctx, log_msg=log_msg)
            except Exception as e:
                trace.append({"error": f"{type(e).__name__}: {str(e)[:300]}"})
                log_msg(f"  -> [Sample Error / Timeout]: {type(e).__name__}: {str(e)[:150]}")
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)  # Huỷ hẹn giờ báo động sau khi mẫu kết thúc

            calls = env.get_tool_calls()
            dt = time.time() - t0
            if dt > TIME_LIMIT_S:
                slow.append((item["id"], round(dt, 1)))

            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": env.get_state_hash()},
                                  ensure_ascii=False) + "\n")
            ftr.write(json.dumps({"id": item["id"], "raw_texts": raw, "mapped_texts": texts,
                                  "kept_history": keep, "steps": trace, "seconds": round(dt, 2)}, ensure_ascii=False) + "\n")
            fout.flush()
            ftr.flush()

            err = next((s["error"][:120] for s in trace if "error" in s), "")
            log_msg(f"[{i + 1}/{len(items)}] {item['id']} DONE calls={len(calls)} {dt:.1f}s {err}".strip())

            # Periodic sync every 10 samples to Google Drive
            if (i + 1) % 10 == 0:
                try:
                    drive_dir = Path("/content/drive/MyDrive/ZAIC_2026_runs")
                    if drive_dir.parent.exists():
                        drive_dir.mkdir(parents=True, exist_ok=True)
                        for f_sync in [a.out, log_path, a.out + ".trace.jsonl"]:
                            if os.path.exists(f_sync):
                                shutil.copy(f_sync, drive_dir / Path(f_sync).name)
                                os.system(f"cp -f '{f_sync}' '{drive_dir}/' 2>/dev/null")
                except Exception:
                    pass

        total = time.time() - t_start
        log_msg(f"\nFinished {len(items)} samples -> {a.out} | avg {total / max(len(items), 1):.1f}s/sample"
                f" | agent > {TIME_LIMIT_S}s: {len(slow)}")

    # Final sync to Google Drive
    try:
        drive_dir = Path("/content/drive/MyDrive/ZAIC_2026_runs")
        if drive_dir.parent.exists():
            drive_dir.mkdir(parents=True, exist_ok=True)
            for f_sync in [a.out, a.out + ".trace.jsonl", log_path]:
                if os.path.exists(f_sync):
                    shutil.copy(f_sync, drive_dir / Path(f_sync).name)
            print(f"✅ Đã tự động sao lưu toàn bộ kết quả sang Google Drive: {drive_dir}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
