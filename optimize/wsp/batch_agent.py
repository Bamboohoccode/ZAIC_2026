"""Bộ chạy theo lô cho WSP: cùng logic agent với lifemate/pipeline.py (run_agent + plan_step), nhưng mọi mẫu chạy cùng lúc.
Mỗi bước gom yêu cầu sinh của tất cả mẫu đang chạy vào một lần llm.generate (vLLM tự xếp lịch theo bộ nhớ KV).

Vì sao: pipeline gốc chạy từng mẫu một. Trên T4, một vòng WSP (50 đến 100 mẫu) quá chậm cho ngân sách 10 đến 50 vòng.
Chỉ hỗ trợ chế độ `--text`, mapper tắt (đúng cấu hình run_round dùng).

Logic agent chép từ pipeline.run_agent/plan_step dưới dạng generator. Test `optimize/wsp/test_batch_agent.py` so từng byte
với pipeline.main bằng LLM giả. Sửa run_agent/plan_step trong pipeline thì sửa ở đây và chạy lại test.

Khác với pipeline:
a) Prompt dựng bằng tokenizer.apply_chat_template rồi llm.generate, vì vLLM chat chỉ nhận một danh sách tools cho cả lô.
b) "seconds" trong trace là thời gian thực khi chạy chung lô, KHÔNG phải độ trễ một mẫu. Thước đo chi phí: "gen_tokens"
   (số token sinh) và "llm_calls". Độ trễ thật đo riêng bằng pipeline gốc (plan bước 7.3).
Baseline và mọi vòng sửa đều chạy bằng bộ này, nên so sánh giữa các vòng vẫn cùng điều kiện.

Chạy: python optimize/wsp/batch_agent.py --llm /tmp/Qwen3-4B --root ROOT --ids ids.txt --out runs/x.jsonl --llm-dtype half
"""
import argparse, json, os, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from lifemate import pipeline as P  # noqa: E402


def agent_gen(env, msgs, tools, catalog, trace, rounds):
    """Generator: `yield (conv, tools)` nhận lại text sinh ra. Bản sao P.run_agent + P.plan_step (bỏ deadline: pipeline.main không đặt)."""
    done = set()
    for _ in range(P.MAX_STEPS):
        # ---- plan_step
        query = msgs[-1]["content"] if msgs[-1]["role"] == "user" else next(m["content"] for m in reversed(msgs) if m["role"] == "user")
        feedback = []
        for attempt in range(rounds + 1):
            out = yield (msgs + feedback, tools)
            trace.append({"plan": out})
            calls = P.parse_calls(out)
            if not calls or attempt == rounds:
                break
            shown = "\n".join(f"<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c in calls)
            text = re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip()
            verdict = yield (msgs + [{"role": "user", "content": P.VERIFY.format(query=query, calls=shown, text=text)}], tools)
            ok, reason = P.parse_verdict(verdict)
            trace.append({"verify": verdict})
            if ok:
                break
            feedback = [{"role": "assistant", "content": out}, {"role": "user", "content": P.REPLAN.format(reason=reason)}]
        # ---- run_agent (phần thực thi)
        if not calls:
            return
        ai = len(msgs)
        msgs.append(P.assistant_msg(re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip(), calls))
        stop = False
        for c in calls:
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            v = env.validate(c["name"], c["arguments"])
            bad = v.get("ok") and P.unseen_ids(c, catalog, msgs[:ai] + msgs[ai + 1:])
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
            msgs.append({"role": "tool", "content": P.clip(r)})
        if stop:
            return


class VLLMBatch:
    """vLLM, nhận một lô (conv, tools) khác nhau. Trả (texts hoặc Exception, số token sinh)."""

    def __init__(self, model, gpu_mem, seed, dtype, think, eager=False):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        self.llm = LLM(model, enable_prefix_caching=True, max_model_len=P.MAX_CTX, seed=seed,
                       gpu_memory_utilization=gpu_mem, enforce_eager=eager, dtype=dtype)
        self.tok = self.llm.get_tokenizer()
        self.kw = {} if think == "on" else {"enable_thinking": False}

    def _ids(self, msgs, tools):
        text = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **self.kw)
        return self.tok.encode(text, add_special_tokens=False)

    def n_tokens(self, msgs, tools):
        return len(self._ids(msgs, tools))

    def run(self, reqs):
        res, ntok = [None] * len(reqs), [0] * len(reqs)
        prompts, sps, idx = [], [], []
        for i, (m, t) in enumerate(reqs):
            try:
                ids = self._ids(m, t)
            except Exception as e:  # noqa: BLE001 - lỗi của một mẫu không được làm hỏng cả lô
                res[i] = e
                continue
            room = P.MAX_CTX - len(ids)
            if room <= 0:
                res[i] = ValueError(f"prompt {len(ids)} token vượt max_model_len {P.MAX_CTX}")
                continue
            prompts.append({"prompt_token_ids": ids})
            sps.append(self.SP(temperature=0, max_tokens=min(P.MAX_NEW, room)))  # greedy như pipeline
            idx.append(i)
        if prompts:
            for i, o in zip(idx, self.llm.generate(prompts, sps, use_tqdm=False)):
                res[i] = o.outputs[0].text.strip()
                ntok[i] = len(o.outputs[0].token_ids)
        return res, ntok


def run_items(engine, root, split, items, catalog, rounds, out_path, log_every=10):
    """Chạy mọi mẫu cùng lúc, ghi submission (`out_path`) và trace (`out_path`.trace.jsonl) giống định dạng pipeline."""
    from zaic_tools import ToolEnv
    t_start = time.time()
    states = []

    def finish(st, err=None):
        if err is not None:
            st["trace"].append({"error": f"{type(err).__name__}: {str(err)[:300]}"})
        st["dt"] = time.time() - st["t0"]
        st["gen"] = None

    for item in items:
        st = {"item": item, "t0": time.time(), "trace": [], "raw": [], "texts": [], "keep": 0, "gen": None,
              "pending": None, "gen_tokens": 0, "llm_calls": 0}
        st["env"] = ToolEnv(split, item["id"], root=str(root))
        states.append(st)
        try:
            turns = [h for h in item["history"] if h["role"] == "user"] + [item]
            st["raw"] = st["texts"] = [u["audio_text"] for u in turns]
            tools = P.tool_specs(item, catalog)
            keep = len(st["texts"]) - 1
            msgs = P.build_messages(item, st["texts"], keep)
            while keep and engine.n_tokens(msgs, tools) > P.MAX_CTX - P.MAX_NEW * 2:
                keep -= 1
                msgs = P.build_messages(item, st["texts"], keep)
            st["keep"] = keep
            g = agent_gen(st["env"], msgs, tools, catalog, st["trace"], rounds)
            st["pending"] = next(g)
            st["gen"] = g
        except StopIteration:
            finish(st)
        except Exception as e:  # noqa: BLE001
            finish(st, e)

    wave = 0
    while True:
        active = [st for st in states if st["gen"] is not None]
        if not active:
            break
        wave += 1
        outs, ntok = engine.run([st["pending"] for st in active])
        for st, o, n in zip(active, outs, ntok):
            st["llm_calls"] += 1
            st["gen_tokens"] += n
            try:
                st["pending"] = st["gen"].throw(o) if isinstance(o, Exception) else st["gen"].send(o)
            except StopIteration:
                finish(st)
            except Exception as e:  # noqa: BLE001
                finish(st, e)
        if log_every and wave % log_every == 0:
            print(f"  wave {wave}: còn {sum(st['gen'] is not None for st in states)}/{len(states)} mẫu, "
                  f"{time.time() - t_start:.0f}s", flush=True)

    with open(out_path, "w", encoding="utf-8") as fout, open(str(out_path) + ".trace.jsonl", "w", encoding="utf-8") as ftr:
        for st in states:
            item, env = st["item"], st["env"]
            fout.write(json.dumps({"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()},
                                  ensure_ascii=False) + "\n")
            ftr.write(json.dumps({"id": item["id"], "raw_texts": st["raw"], "mapped_texts": st["texts"],
                                  "kept_history": st["keep"], "steps": st["trace"], "seconds": round(st.get("dt", 0), 2),
                                  "gen_tokens": st["gen_tokens"], "llm_calls": st["llm_calls"]}, ensure_ascii=False) + "\n")
    total = time.time() - t_start
    print(f"{len(items)} mẫu -> {out_path} | {total:.0f}s tổng, {wave} wave, "
          f"gen_tokens TB {sum(s['gen_tokens'] for s in states) / max(len(states), 1):.0f}", flush=True)
    return total


class GoldBatch:
    """Kiểm luồng không GPU: phát lại label.tool_calls, mỗi bước một call (như pipeline.GoldLLM). Dùng với verify_rounds 0."""

    def __init__(self, items):
        self.gold = {it["audio_text"]: it["label"]["tool_calls"] for it in items}

    def n_tokens(self, msgs, tools):
        return 0

    def run(self, reqs):
        outs = []
        for msgs, _ in reqs:
            ui = max(i for i, m in enumerate(msgs) if m["role"] == "user" and not m["content"].startswith("["))
            n = sum(1 for m in msgs[ui:] if m["role"] == "assistant" and m.get("tool_calls"))
            calls = self.gold.get(msgs[ui]["content"], [])
            outs.append(f"<tool_call>{json.dumps(calls[n], ensure_ascii=False)}</tool_call>" if n < len(calls) else "Xong.")
        return outs, [1] * len(reqs)


def load_items(root, split, ids_path):
    items = [json.loads(l) for l in open(Path(root) / split / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    if ids_path:
        keep = {l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()}
        items = [it for it in items if it["id"] in keep]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="thư mục model (gốc hoặc đã sửa)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-think", choices=["on", "off"], default="on")
    ap.add_argument("--llm-dtype", default="auto")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--verify-rounds", type=int, default=P.VERIFY_ROUNDS)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = Path(a.root)
    sys.path.insert(0, str(root / a.split))
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = load_items(root, a.split, a.ids)
    if a.llm == "gold":  # như pipeline: gold không qua verifier
        run_items(GoldBatch(items), root, a.split, items, catalog, 0, a.out)
        return
    engine = VLLMBatch(a.llm, a.gpu_mem, a.seed, a.llm_dtype, a.llm_think, a.eager)
    run_items(engine, root, a.split, items, catalog, a.verify_rounds, a.out)


if __name__ == "__main__":
    main()
