"""Stage 2 DPO On-Policy Rollout Sampler for Ma7ee7/Qwen3.8_4B_Distilled.

Use this script ONLY if Stage 1 programmatic DPO (optimize/dpo/dataset/dpo_train.jsonl)
underperforms on eval/splits/dev_ids.txt. It runs the target Qwen model via vLLM on
eval/splits/dpo_train_ids.txt (1,145 training samples, zero leakage with Val/Test),
executes rollouts against zaic_tools.ToolEnv, and pairs real model failure traces as
`rejected` against ToolEnv-verified `chosen` completions.

Usage (on Colab A100/H100/L4):
  python -X utf8 scripts/sample_onpolicy_dpo.py --root data --split train \\
      --ids eval/splits/dpo_train_ids.txt --model Ma7ee7/Qwen3.8_4B_Distilled --resume

Dry-run check (no GPU / no vLLM required):
  python -X utf8 scripts/sample_onpolicy_dpo.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scorer import call_matches
from lifemate.pipeline import (
    MAX_CTX,
    MAX_NEW,
    assistant_msg,
    build_messages,
    clip,
    parse_calls,
    tool_specs,
    unseen_ids,
)
from lifemate.planner.train_dpo import wrap_reasoning_in_think
from scripts.build_dpo_dataset import (
    make_chosen_response,
    norm_text,
    sample_apps,
    split_turns,
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think_block(text: str) -> str:
    """Remove `<think>...</think>` blocks when aligned strip mode is requested."""
    return THINK_BLOCK_RE.sub("", text).strip()


def align_think_format(text: str, think_mode: str) -> str:
    """Apply consistent `<think>` formatting to both chosen and rejected strings."""
    if think_mode == "strip":
        return strip_think_block(text)
    if think_mode == "prefix":
        return wrap_reasoning_in_think(text)
    return text.strip()


def trim_msgs_to_fit(
    msgs: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    """Render prompt with Qwen chat template, dropping oldest user turns if over budget."""
    cur = list(msgs)

    def render(m_list: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            m_list, tools=tools, add_generation_prompt=True, tokenize=False
        )

    prompt_str = render(cur)
    while len(tokenizer.encode(prompt_str, add_special_tokens=False)) > max_prompt_tokens:
        user_idxs = [i for i, m in enumerate(cur) if m.get("role") == "user"]
        if len(user_idxs) <= 1:
            break
        cur = cur[: user_idxs[0]] + cur[user_idxs[1] :]
        prompt_str = render(cur)
    return cur, prompt_str


def classify_rollout_step(
    raw_out: str,
    step_calls: list[dict[str, Any]],
    intent: str,
    msgs_so_far: list[dict[str, Any]],
    env: Any,
    catalog: dict[str, Any],
) -> tuple[bool, str, str]:
    """Evaluate a single rollout completion against the step's gold ToolEnv expectations.

    Returns `(is_exact_match, negative_class, detail)`.
    """
    pred_calls = parse_calls(raw_out)

    # Empty / out-of-scope step: model must emit zero tool calls
    if not step_calls:
        if not pred_calls:
            return True, "ok", "exact_empty"
        return False, "multi_action_or_scope_error", f"onpolicy_false_act:{pred_calls[0]['name']}"

    # Ambiguity gate step: must match sys.ask_missing_info with exact target tool & params
    if intent == "ask" or any(c["name"] == "sys.ask_missing_info" for c in step_calls):
        if len(pred_calls) == len(step_calls) and all(
            any(call_matches(p, g, catalog) for g in step_calls) for p in pred_calls
        ):
            return True, "ok", "exact_ask"
        if not any(c["name"] == "sys.ask_missing_info" for c in pred_calls):
            detail = pred_calls[0]["name"] if pred_calls else "no_call"
            return False, "ambiguity_gate_error", f"onpolicy_missed_ask:{detail}"
        return False, "ambiguity_gate_error", "onpolicy_wrong_ask_args"

    # Normal read/write step: zero tool calls -> premature stop / think overflow
    if not pred_calls:
        tag = "think_overflow_no_tool" if "<think>" in raw_out and "</think>" not in raw_out else "no_tool_call"
        return False, "multi_action_or_scope_error", f"onpolicy_{tag}"

    # False Ask instead of valid action
    if any(c["name"] == "sys.ask_missing_info" for c in pred_calls):
        return False, "ambiguity_gate_error", f"onpolicy_false_ask:{step_calls[0]['name']}"

    # Schema & ID grounding validation (mirrors runner.py)
    for c in pred_calls:
        cname = c.get("name", "")
        if cname not in catalog:
            return False, "confusable_tool", f"onpolicy_unknown_tool:{cname}"
        v = env.validate(cname, c.get("arguments") or {})
        if not v.get("ok"):
            return False, "hallucinated_optional_args", f"onpolicy_schema_err:{cname}"
        bad_ids = unseen_ids(c, catalog, msgs_so_far)
        if bad_ids:
            return False, "premature_write_or_ask", f"onpolicy_unseen_id:{cname}.{bad_ids[0]}"

    # Exact 1-to-1 call match against gold calls at this turn
    left = list(step_calls)
    matched = 0
    for p in pred_calls:
        j = next((idx for idx, g in enumerate(left) if call_matches(p, g, catalog)), None)
        if j is not None:
            left.pop(j)
            matched += 1
    if matched == len(step_calls) and len(pred_calls) == len(step_calls):
        return True, "ok", "exact_act"

    # Categorize mismatch for balanced reporting
    gold_names = {g["name"] for g in step_calls}
    pred_names = {p["name"] for p in pred_calls}
    if not pred_names <= gold_names:
        wrong_t = sorted(pred_names - gold_names)[0]
        expected_t = sorted(gold_names - pred_names or gold_names)[0]
        return False, "confusable_tool", f"onpolicy_wrong_tool:{expected_t}->{wrong_t}"
    if len(pred_calls) < len(step_calls):
        return False, "multi_action_or_scope_error", "onpolicy_missing_call"
    if len(pred_calls) > len(step_calls):
        return False, "multi_action_or_scope_error", "onpolicy_extra_call"

    # Same tool name(s), wrong or extra optional args
    for p, g in zip(pred_calls, step_calls):
        extra_keys = set((p.get("arguments") or {}).keys()) - set((g.get("arguments") or {}).keys())
        if extra_keys:
            return False, "hallucinated_optional_args", f"onpolicy_extra_arg:{p['name']}.{sorted(extra_keys)[0]}"
    return False, "hallucinated_optional_args", f"onpolicy_wrong_arg_val:{pred_calls[0]['name']}"


def load_clean_samples(
    root: Path,
    split: str,
    ids_path: Path,
    limit: int,
    exclude_sids: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load catalog and deduplicated non-leaking items matching `ids_path`."""
    split_dir = root / split
    if str(split_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(split_dir.resolve()))
    raw_catalog = json.load(open(split_dir / f"{split}_tools_catalog.json", encoding="utf-8"))
    catalog = {t["name"]: t for app in raw_catalog["apps"] for t in app["tools"]}
    rows = {x["id"]: x for x in map(json.loads, open(split_dir / f"{split}.jsonl", encoding="utf-8")) if x.get("id")}
    target_ids = [s.strip() for s in open(ids_path, encoding="utf-8") if s.strip()]

    blocked = exclude_sids or set()
    seen_texts: set[str] = set()
    items: list[dict[str, Any]] = []
    for sid in target_ids:
        if sid in blocked:
            continue
        item = rows[sid]
        nt = norm_text(item["audio_text"])
        if nt in seen_texts:
            continue
        seen_texts.add(nt)
        items.append(item)
        if limit and len(items) >= limit:
            break
    return catalog, items


def load_programmatic_fallback_map(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Index Stage 1 programmatic pairs by `(sample_id, step)` for optional anti-forgetting blending."""
    by_turn: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if not path.exists():
        return by_turn
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_turn.setdefault((r["sample_id"], int(r.get("step", 0))), []).append(r)
    return by_turn


def sample_rollouts_vllm(
    llm_engine: Any,
    lora_req: Any,
    prompts: list[str],
    num_samples: int,
    temperature: float,
    top_p: float,
    max_new: int,
) -> list[list[str]]:
    """Sample 1 greedy rollout + (num_samples - 1) diverse rollouts per prompt via vLLM."""
    from vllm import SamplingParams

    sp_greedy = SamplingParams(temperature=0.0, max_tokens=max_new)
    greedy_outs = llm_engine.generate(prompts, sp_greedy, lora_request=lora_req, use_tqdm=False)
    candidates: list[list[str]] = [[o.outputs[0].text.strip()] for o in greedy_outs]

    if num_samples > 1:
        sp_sample = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            n=num_samples - 1,
            max_tokens=max_new,
        )
        samp_outs = llm_engine.generate(prompts, sp_sample, lora_request=lora_req, use_tqdm=False)
        for idx, o in enumerate(samp_outs):
            candidates[idx].extend(cand.text.strip() for cand in o.outputs)
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2 On-Policy DPO Rollout Sampler via vLLM + ToolEnv")
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", type=Path, default=Path("eval/splits/dpo_train_ids.txt"))
    ap.add_argument("--prompt", type=Path, default=Path("prompts/gepa/vuong/planner_v15.txt"))
    ap.add_argument("--model", default="Ma7ee7/Qwen3.8_4B_Distilled")
    ap.add_argument("--lora", default="", help="Optional Stage 1 LoRA adapter path to sample on-policy failures from")
    ap.add_argument("--out", type=Path, default=Path("optimize/dpo/dataset/dpo_onpolicy_train.jsonl"))
    ap.add_argument("--num-samples", type=int, default=3, help="Rollouts per turn (1 greedy + N-1 stochastic)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-ctx", type=int, default=12288)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--batch-size", type=int, default=16, help="Number of turns batched per vLLM.generate call")
    ap.add_argument("--max-pairs-per-step", type=int, default=2)
    ap.add_argument("--think-mode", choices=["prefix", "strip", "keep"], default="prefix")
    ap.add_argument(
        "--exclude-val",
        type=Path,
        default=Path("optimize/dpo/dataset/dpo_val.jsonl"),
        help="Exclude sample_ids present in Stage 1 val set to guarantee 0%% train/val overlap.",
    )
    ap.add_argument(
        "--include-programmatic",
        action="store_true",
        help="Include 1 programmatic hard negative from --programmatic-path when all rollouts pass on a turn.",
    )
    ap.add_argument("--programmatic-path", type=Path, default=Path("optimize/dpo/dataset/dpo_train.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip sample_ids already present in --out and append")
    ap.add_argument("--dry-run", action="store_true", help="Validate paths, tokenizer, and step unrolling without GPU")
    args = ap.parse_args()

    if not args.prompt.exists():
        raise FileNotFoundError(f"Missing planner prompt: {args.prompt}")
    system_template = args.prompt.read_text(encoding="utf-8")

    if not args.ids.exists():
        raise FileNotFoundError(f"Missing IDs file: {args.ids}")

    exclude_sids: set[str] = set()
    if args.exclude_val and args.exclude_val.exists():
        exclude_sids = {json.loads(l)["sample_id"] for l in open(args.exclude_val, encoding="utf-8") if l.strip()}

    has_train_data = (args.root / args.split / f"{args.split}.jsonl").exists()
    if args.dry_run:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        id_count = sum(1 for l in open(args.ids, encoding="utf-8") if l.strip() and l.strip() not in exclude_sids)
        print("[DRY-RUN] On-policy DPO sampler configuration valid.")
        print(f"  Model               : {args.model}")
        print(f"  LoRA                : {args.lora or '<none (base model)>'}")
        print(f"  IDs file            : {args.ids} ({id_count} non-val IDs; excluded {len(exclude_sids)} val IDs)")
        print(f"  Prompt              : {args.prompt} ({len(system_template)} chars)")
        print(f"  Output JSONL        : {args.out}")
        print(f"  Think mode          : {args.think_mode} | num_samples={args.num_samples} | batch_size={args.batch_size}")
        print(f"  Include programmatic: {args.include_programmatic}")
        if has_train_data:
            catalog, items = load_clean_samples(
                args.root, args.split, args.ids, limit=args.limit or 5, exclude_sids=exclude_sids
            )
            turns = [h["audio_text"] for h in items[0]["history"] if h["role"] == "user"] + [items[0]["audio_text"]]
            msgs = build_messages(items[0], turns, keep=len(turns) - 1)
            tools = tool_specs(items[0], catalog)
            _, rendered = trim_msgs_to_fit(msgs, tools, tok, args.max_ctx - args.max_new * 2)
            print(f"  Sample check        : loaded {len(items)} items, catalog={len(catalog)} tools, sample0_prompt_chars={len(rendered)}")
        else:
            print(f"  Data dir            : {args.root / args.split} not present locally (run with --root data on Colab).")
        return

    if not has_train_data:
        raise FileNotFoundError(f"Missing {args.root / args.split / f'{args.split}.jsonl'}. Extract data.zip first.")

    from lifemate.pipeline import date_table, hints
    from zaic_tools import ToolEnv
    from vllm import LLM

    catalog, items = load_clean_samples(args.root, args.split, args.ids, limit=args.limit, exclude_sids=exclude_sids)
    prog_fallback = load_programmatic_fallback_map(args.programmatic_path) if args.include_programmatic else {}

    done_samples: set[str] = set()
    ckpt_path = args.out.with_suffix(".done_ids")
    if args.resume and ckpt_path.exists():
        done_samples = {s.strip() for s in open(ckpt_path, encoding="utf-8") if s.strip()}
        items = [it for it in items if it["id"] not in done_samples]
        print(f"Resume active: skipping {len(done_samples)} finished samples; {len(items)} remaining.")

    lora_kw: dict[str, Any] = {}
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        r = json.load(open(Path(args.lora) / "adapter_config.json", encoding="utf-8"))["r"]
        lora_kw = {"enable_lora": True, "max_lora_rank": next(x for x in (8, 16, 32, 64, 128, 256, 512) if x >= r)}
        lora_req = LoRARequest("dpo_s1", 1, str(args.lora))

    print(f"Loading vLLM engine for {args.model} (max_ctx={args.max_ctx}, gpu_mem={args.gpu_mem})...")
    llm_engine = LLM(
        args.model,
        enable_prefix_caching=True,
        max_model_len=args.max_ctx,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_mem,
        **lora_kw,
    )
    tok = llm_engine.get_tokenizer()

    # Pre-unroll every gold trajectory via ToolEnv to build verified turn-level prompts
    turn_tasks: list[dict[str, Any]] = []
    for item in items:
        sid = item["id"]
        calls = item["label"]["tool_calls"]
        results = item["label"].get("tool_results") or []
        intent = "empty" if not calls else ("ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "act")
        target_apps = sample_apps(calls) or ["none"]
        turns_texts = [h["audio_text"] for h in item["history"] if h["role"] == "user"] + [item["audio_text"]]
        base_msgs = build_messages(item, turns_texts, keep=len(turns_texts) - 1)
        # Replace default SYSTEM with requested --prompt template
        ctx = item["context"]
        base_msgs[0]["content"] = system_template.format(
            now=ctx["current_time"],
            tz=ctx["timezone"],
            dates=date_table(ctx["current_time"]) + hints(turns_texts[-1]),
            loc=json.dumps(ctx.get("location"), ensure_ascii=False),
            profile=json.dumps(ctx.get("user_profile"), ensure_ascii=False),
        )
        tools = tool_specs(item, catalog)
        tool_names = [t["function"]["name"] for t in tools]

        env = ToolEnv(args.split, sid, root=str(args.root))
        msgs = copy.deepcopy(base_msgs)
        prev_calls: list[dict[str, Any]] = []
        turn_groups = split_turns(calls, results, catalog) if calls else [[]]
        for step_idx, grp in enumerate(turn_groups):
            step_calls = [calls[idx] for idx in grp]
            trimmed_msgs, prompt_str = trim_msgs_to_fit(msgs, tools, tok, args.max_ctx - args.max_new * 2)
            fallback_chosen = make_chosen_response(step_calls, step_idx, len(turn_groups), intent, catalog)
            turn_tasks.append({
                "sample_id": sid,
                "step": step_idx,
                "intent": intent,
                "apps": target_apps,
                "tool_names": tool_names,
                "msgs": trimmed_msgs,
                "prompt_str": prompt_str,
                "step_calls": step_calls,
                "prev_calls": list(prev_calls),
                "fallback_chosen": fallback_chosen,
                "is_last_step": step_idx == len(turn_groups) - 1,
            })
            if step_calls:
                msgs.append(assistant_msg("", step_calls))
                for c in step_calls:
                    r = env.call(c["name"], c.get("arguments") or {})
                    prev_calls.append(c)
                    msgs.append({"role": "tool", "content": clip(r)})

    print(f"Prepared {len(turn_tasks)} turns across {len(items)} samples. Sampling {args.num_samples} rollouts/turn...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_mode = "a" if (args.resume and args.out.exists()) else "w"
    ckpt_mode = "a" if (args.resume and ckpt_path.exists()) else "w"
    stats = Counter()

    with open(args.out, out_mode, encoding="utf-8") as fout, open(ckpt_path, ckpt_mode, encoding="utf-8") as fckpt:
        for b_start in range(0, len(turn_tasks), args.batch_size):
            batch = turn_tasks[b_start : b_start + args.batch_size]
            cand_lists = sample_rollouts_vllm(
                llm_engine,
                lora_req,
                [t["prompt_str"] for t in batch],
                args.num_samples,
                args.temperature,
                args.top_p,
                args.max_new,
            )

            for task, candidates in zip(batch, cand_lists):
                env_step = ToolEnv(args.split, task["sample_id"], root=str(args.root))
                for pc in task["prev_calls"]:
                    env_step.call(pc["name"], pc.get("arguments") or {})
                passing: list[str] = []
                failing: list[tuple[str, str, str]] = []
                seen_rej: set[str] = set()

                for cand in candidates:
                    ok, neg_cls, detail = classify_rollout_step(
                        cand, task["step_calls"], task["intent"], task["msgs"], env_step, catalog
                    )
                    if ok:
                        passing.append(cand)
                    elif cand not in seen_rej:
                        seen_rej.add(cand)
                        failing.append((cand, neg_cls, detail))

                if failing:
                    raw_chosen = passing[0] if passing else task["fallback_chosen"]
                    chosen_aligned = align_think_format(raw_chosen, args.think_mode)
                    written = 0
                    for rej_idx, (rej_raw, neg_cls, detail) in enumerate(failing):
                        if written >= args.max_pairs_per_step:
                            break
                        rej_aligned = align_think_format(rej_raw, args.think_mode)
                        if not rej_aligned or rej_aligned == chosen_aligned:
                            continue
                        rec = {
                            "id": f"{task['sample_id']}_s{task['step']}_onpolicy_{rej_idx}",
                            "sample_id": task["sample_id"],
                            "step": task["step"],
                            "intent": task["intent"],
                            "negative_class": neg_cls,
                            "negative_detail": detail,
                            "apps": task["apps"],
                            "tool_names": task["tool_names"],
                            "prompt": task["msgs"],
                            "chosen": chosen_aligned,
                            "rejected": rej_aligned,
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats[neg_cls] += 1
                        stats["total_pairs"] += 1
                        written += 1
                else:
                    stats["all_passed_skipped"] += 1
                    if args.include_programmatic:
                        fb_list = prog_fallback.get((task["sample_id"], task["step"]), [])
                        if fb_list:
                            fb = dict(fb_list[0])
                            fb["prompt"] = task["msgs"]
                            fb["chosen"] = align_think_format(fb["chosen"], args.think_mode)
                            fb["rejected"] = align_think_format(fb["rejected"], args.think_mode)
                            if fb["chosen"] and fb["rejected"] and fb["chosen"] != fb["rejected"]:
                                fout.write(json.dumps(fb, ensure_ascii=False) + "\n")
                                stats[fb.get("negative_class", "programmatic")] += 1
                                stats["programmatic_fallback"] += 1
                                stats["total_pairs"] += 1

                if task["is_last_step"]:
                    fckpt.write(task["sample_id"] + "\n")

            fout.flush()
            fckpt.flush()
            done_turns = min(b_start + args.batch_size, len(turn_tasks))
            print(f"  [{done_turns}/{len(turn_tasks)} turns] pairs={stats['total_pairs']} | all_pass={stats['all_passed_skipped']}")

    print(f"Saved on-policy DPO pairs -> {args.out}")
    print(f"Summary stats: {dict(stats)}")


if __name__ == "__main__":
    main()
