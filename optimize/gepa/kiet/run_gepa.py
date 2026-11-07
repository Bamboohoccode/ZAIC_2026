"""GEPA riêng của Kiệt: tối ưu prompt Planner và prompt Mapper (cả hai chạy trên Qwen3-4B) bằng metric chung optimize/gepa/metric.

python -m optimize.gepa.kiet.run_gepa asr  --root data                              # ASR 1 lần -> runs/asr/train_asr.jsonl (cho Mapper)
python -m optimize.gepa.kiet.run_gepa run  --root data --component planner          # --text -> prompts/gepa/kiet/planner_vN.txt
python -m optimize.gepa.kiet.run_gepa run  --root data --component mapper           # ASR -> Mapper -> Planner cố định -> mapper_vN.txt
python -m optimize.gepa.kiet.run_gepa eval --root data --prompt prompts/gepa/kiet/planner_v1.txt          # dev, --text
python -m optimize.gepa.kiet.run_gepa eval --root data --asr                                             # dev, ASR, không Mapper
python -m optimize.gepa.kiet.run_gepa eval --root data --mapper old                                    # dev, ASR + Mapper cũ (0.6B)
python -m optimize.gepa.kiet.run_gepa eval --root data --mapper prompts/gepa/kiet/mapper_v1.txt          # dev, ASR + Mapper mới
python -m optimize.gepa.kiet.run_gepa eval --root data --items data/synth/train_synth.jsonl              # theo 3 nhãn synth
Cấu hình: optimize/gepa/kiet/config.yaml (đường dẫn tính từ gốc repo). Cài: pip install gepa pyyaml vllm chunkformer.

Planner: chỉ tối ưu phần hướng dẫn sau "{dates}" của system prompt, phần đầu (giờ, vị trí, hồ sơ, lịch) giữ nguyên.
Mapper: prompt riêng trên chính Qwen3-4B (không thêm model), input n-best + từ điển thực thể từ catalog. Tối ưu bằng metric
nhanh (giữ thực thể + WER so với audio_text, không chạy Planner, mapper_metric: proxy) hoặc điểm cuối (e2e); chốt bằng điểm cuối.
Train/val chia theo app và loại hẳn dev_ids.txt, nên dev là tập giữ lại chưa từng lọt vào vòng tối ưu.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import yaml
from gepa.core.adapter import EvaluationBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # gốc repo, để chạy trực tiếp như tests/test_pipeline.py

import lifemate.pipeline as P
from eval.scorer import gold_hash, load_split, task_app, task_kind
from optimize.gepa.metric.feedback import text_diff
from optimize.gepa.metric.metric import evaluate_mapper, evaluate_sample

MARK = "{dates}\n"
read = lambda p: Path(p).read_text(encoding="utf-8")
words = lambda s: re.findall(r"\w+", unicodedata.normalize("NFC", s).lower())


def split_prompt(tpl):
    """Template pipeline -> (phần đầu có placeholder, phần hướng dẫn dạng text thuần: '{{' -> '{')."""
    head, body = tpl.split(MARK, 1)
    return head + MARK, body.replace("{{", "{").replace("}}", "}")


def join_prompt(head, body):
    """Ngược lại split_prompt: escape ngoặc để SYSTEM.format() không vỡ khi LLM viết JSON ví dụ."""
    return head + body.replace("{", "{{").replace("}", "}}")


def pick(items, n, mix, rng):
    """n mẫu theo tỉ lệ nhóm act/ask/empty, để ask và empty (nhóm yếu) không bị nhóm act lấn át."""
    out = []
    for kind, share in mix.items():
        pool = [i for i in items if task_kind(i["label"]["tool_calls"]) == kind]
        out += rng.sample(pool, min(round(n * share), len(pool)))
    return out


def split_data(items, dev_ids, cfg, synth=()):
    """(train, val) loại dev; val gồm các app lạ (val_app_frac số app) nên đo được khả năng khái quát sang app mới.
    synth: mẫu sinh từ synth/, chỉ vào train, bỏ mẫu có nguồn thuộc dev hoặc app của val để không rò."""
    rng = random.Random(cfg["seed"])
    app = lambda i: task_app(i["label"]["tool_calls"])
    pool = [i for i in items if i["id"] not in dev_ids]
    apps = sorted({app(i) for i in pool} - {"(none)"})
    held = set(rng.sample(apps, max(1, round(len(apps) * cfg["val_app_frac"]))))
    val, train = [], []
    for i in pool:
        to_val = app(i) in held or (app(i) == "(none)" and rng.random() < cfg["val_app_frac"])
        (val if to_val else train).append(i)
    train += [s for s in synth if s["synth"] != "full" and s["id"] not in dev_ids and s["synth_app"] not in held]
    return pick(train, cfg["n_train"], cfg["mix"], rng), pick(val, cfg["n_val"], cfg["mix"], rng)

# ---------------------------------------------------------------- ASR + Mapper


def load_asr(path):
    """File dạng ASROut (lifemate/contracts.py), mỗi dòng {"id", "turns": [{"role", "nbest": [{"text", "score"}]}]}
    -> {id: [[hypothesis, ...] cho từng lượt user]}. n-best của Khánh cùng định dạng nên thay file là dùng được."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {r["id"]: [[h["text"] for h in t["nbest"]] for t in r["turns"]] for r in rows}


def build_asr_cache(cfg):
    """ASR 1-best bằng ChunkFormer của pipeline cho mọi lượt user (history rồi query); chạy tiếp nếu file đã có."""
    _, items, _ = load_split(cfg["root"], cfg["split"])
    out = Path(cfg["asr_cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(load_asr(out)) if out.exists() else set()
    todo = [i for i in items if i["id"] not in done]
    model = P.load_asr(P.ASR_MODEL, None, cfg["asr_device"])
    with open(out, "a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            turns = [h for h in it["history"] if h["role"] == "user"] + [it]
            texts = P.run_asr(model, [Path(cfg["root"]) / cfg["split"] / u["audio"] for u in turns])
            f.write(json.dumps({"id": it["id"], "turns": [{"role": "user", "nbest": [{"text": t, "score": 0.0}]} for t in texts]},
                               ensure_ascii=False) + "\n")
            print(f"[{k}/{len(todo)}] {it['id']}", flush=True)


def clean_mapped(out, fallback):
    """Dòng đầu của câu trả lời, bỏ tiền tố/ngoặc kép; rỗng thì giữ bản ASR."""
    line = next((l.strip() for l in out.splitlines() if l.strip()), "")
    line = re.sub(r"^(câu (đã )?(sửa|chuẩn( hoá)?)|kết quả|output)\s*:\s*", "", line, flags=re.I).strip("\"'“” ")
    return line or fallback


def map_turns(llm, prompt, items, asr, vocab, last_only=False):
    """Mapper trên Qwen3-4B cho các lượt user (hoặc chỉ lượt cuối), một lần gọi batch. -> [[câu đã sửa mỗi lượt] mỗi mẫu].
    vocab(item): từ điển thực thể từ catalog (Planner.vocab)."""
    jobs = [(k, nb, it) for k, it in enumerate(items) for nb in (asr[it["id"]][-1:] if last_only else asr[it["id"]])]
    show = lambda nb: nb[0] if len(nb) == 1 else "\n".join(f"{j}. {h}" for j, h in enumerate(nb, 1))
    convs = [[{"role": "system", "content": prompt}, {"role": "user", "content": f"Bản nhận dạng:\n{show(nb)}\n{vocab(it)}"}]
             for _, nb, it in jobs]
    res = [[] for _ in items]
    for (k, nb, _), o in zip(jobs, llm.chat(convs) if convs else []):
        res[k].append(clean_mapped(o, nb[0]))
    return res


def mapper_pool(items, asr, cfg):
    """Mẫu có ASR; ưu tiên mẫu ASR lệch audio_text (nơi Mapper có tác dụng), giữ thêm mapper_same_frac mẫu ASR đã đúng
    để Mapper học chép y nguyên."""
    rng = random.Random(cfg["seed"])
    have = [i for i in items if i["id"] in asr]
    diff = [i for i in have if words(asr[i["id"]][-1][0]) != words(i["audio_text"])]
    same = [i for i in have if words(asr[i["id"]][-1][0]) == words(i["audio_text"])]
    return diff + rng.sample(same, min(len(same), round(len(diff) * cfg["mapper_same_frac"])))

# ---------------------------------------------------------------- planner + GEPA


class Planner:
    """Qwen3-4B + ToolEnv của split: chạy một mẫu với một system prompt (dùng lại run_agent của pipeline)."""

    def __init__(self, cfg, llm=None):
        self.cfg = cfg
        self.catalog, self.items, self.ToolEnv = load_split(cfg["root"], cfg["split"])
        apps = json.load(open(Path(cfg["root"]) / cfg["split"] / f"{cfg['split']}_tools_catalog.json", encoding="utf-8"))["apps"]
        self.app_vi = {a.get("id"): a.get("name_vi") for a in apps}
        if llm is None:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            llm = P.VLLM(cfg["model"], None, cfg["gpu_mem"], cfg["seed"], cfg["max_ctx"], P.MAX_NEW, None, cfg["eager"])
        self.llm, self._gold = llm, {}

    def vocab(self, item):
        """Từ điển thực thể cho Mapper, lấy từ catalog: app đã cài (kèm tên tiếng Việt) và giá trị enum của tool."""
        apps = item["context"]["installed_apps"]
        names = [f"{a} ({self.app_vi[a]})" if self.app_vi.get(a) else a for a in apps]
        enums = [str(e) for v in apps.values() for t in v.get("tools", [])
                 for p in self.catalog.get(t, {}).get("parameters", {}).get("properties", {}).values() for e in p.get("enum", [])]
        enums = list(dict.fromkeys(enums))[:self.cfg["max_vocab"]]
        return f"App đã cài: {', '.join(names)}" + (f"\nGiá trị hợp lệ: {', '.join(enums)}" if enums else "")

    def gold_hash(self, item):
        key = item.get("synth_id", item["id"])  # mẫu synth dùng chung môi trường (id) với mẫu gốc nhưng đáp án khác
        if key not in self._gold:
            c = self.cfg
            self._gold[key] = gold_hash(self.ToolEnv, c["root"], c["split"], item["id"], item["label"]["tool_calls"])
        return self._gold[key]

    def run(self, item, system, texts=None):
        """-> (tool_calls, state_hash, trace, giây). Giống pipeline.main; texts None = audio_text (--text)."""
        c, t0 = self.cfg, time.time()
        P.SYSTEM = system
        env = self.ToolEnv(c["split"], item["id"], root=c["root"])
        texts = texts or [h["audio_text"] for h in item["history"] if h["role"] == "user"] + [item["audio_text"]]
        tools, keep, trace = P.tool_specs(item, self.catalog), len(texts) - 1, []
        msgs = P.build_messages(item, texts, keep)
        while keep and self.llm.n_tokens(msgs, tools) > c["max_ctx"] - 2 * P.MAX_NEW:
            keep -= 1
            msgs = P.build_messages(item, texts, keep)
        try:
            deadline = t0 + c["time_limit_s"] if c["time_limit_s"] else float("inf")  # 0 = không cắt, như pipeline
            P.run_agent(env, self.llm, msgs, tools, self.catalog, trace, c["verify_rounds"], deadline)
        except Exception as e:  # giữ các call đã chạy, như pipeline
            trace.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        return env.get_tool_calls(), env.get_state_hash(), trace, time.time() - t0


class Adapter:
    """Cầu nối GEPA. component planner: prompt Planner, input audio_text. component mapper: prompt Mapper, input ASR,
    Planner cố định (system). Cả hai chấm bằng điểm cuối (metric chung); reflective dataset = lỗi kèm feedback."""
    propose_new_texts = None  # dùng proposer mặc định của GEPA

    def __init__(self, planner, cfg, component, head=None, system=None, asr=None):
        self.pl, self.cfg, self.comp, self.head, self.system, self.asr = planner, cfg, component, head, system, asr

    def evaluate(self, batch, candidate, capture_traces=False):
        text, outs, scores, trajs = candidate[self.comp], [], [], []
        if not text.strip() or len(text) > self.cfg["max_prompt_chars"]:
            t = [{"item": i, "ok": False, "out": "", "feedback": "Prompt rỗng hoặc quá dài."} for i in batch]
            return EvaluationBatch(outputs=[{"id": i["id"], "score": 0.0} for i in batch], scores=[0.0] * len(batch),
                                   trajectories=t if capture_traces else None)
        if self.comp == "mapper" and self.cfg["mapper_metric"] == "proxy":  # rẻ: chỉ Mapper lượt cuối, không chạy Planner
            for it, (mapped,) in zip(batch, map_turns(self.pl.llm, text, batch, self.asr, self.pl.vocab, last_only=True)):
                m = evaluate_mapper(self.asr[it["id"]][-1][0], mapped, it["audio_text"], it["label"]["tool_calls"])
                outs.append({"id": it["id"], "score": m["score"], "wer": round(m["wer"], 3)})
                scores.append(m["score"])
                trajs.append({"item": it, "ok": m["score"] >= 0.999, "out": mapped, "feedback": m["feedback"]})
            return EvaluationBatch(outputs=outs, scores=scores, trajectories=trajs if capture_traces else None)
        if self.comp == "planner":
            system, mapped = join_prompt(self.head, text), [None] * len(batch)
        else:  # mapper e2e: chấm bằng điểm cuối của Planner cố định
            system, mapped = self.system, map_turns(self.pl.llm, text, batch, self.asr, self.pl.vocab)
        for item, texts in zip(batch, mapped):
            G = item["label"]["tool_calls"]
            calls, h, trace, dt = self.pl.run(item, system, texts)
            m = evaluate_sample(G, calls, h, self.pl.gold_hash(item), self.pl.catalog, self.cfg["hash_credit"])
            fb = f"[nhóm {m['kind']}] {m['feedback']}" + "".join(f"\nLỗi chạy: {s['error']}" for s in trace if "error" in s)
            if texts:
                fb = text_diff(self.asr[item["id"]][-1][0], texts[-1], item["audio_text"]) + "\nKết quả Planner với câu này: " + fb
            outs.append({"id": item["id"], "score": m["score"], "kind": m["kind"], "error": m["error"], "seconds": round(dt, 1)})
            scores.append(m["shaped"])
            out = texts[-1] if texts else P.clip(next((s["plan"] for s in trace if "plan" in s), ""), 450)
            trajs.append({"item": item, "ok": m["error"] is None, "out": out, "feedback": fb})
        return EvaluationBatch(outputs=outs, scores=scores, trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        ts = eval_batch.trajectories
        rows = [t for t in ts if not t["ok"]][:self.cfg["max_records"]] + [t for t in ts if t["ok"]][:1]
        recs = []
        for t in rows:
            it = t["item"]
            inputs = ({"Bản nhận dạng": " | ".join(self.asr[it["id"]][-1]), "Từ điển": self.pl.vocab(it)} if self.comp == "mapper" else
                      {"Yêu cầu": it["audio_text"], "Số lượt trước đó": len(it["history"]),
                       "App đã cài": ", ".join(it["context"]["installed_apps"])})
            recs.append({"Inputs": inputs, "Generated Outputs": t["out"], "Feedback": t["feedback"]})
        return {c: recs for c in components_to_update}


def make_reflector(cfg, planner):
    """reflection LM: mặc định chính Qwen3-4B đang chạy (offline); đặt reflection.api_url để dùng API OpenAI-compatible."""
    r = cfg["reflection"]
    msgs_of = lambda p: p if isinstance(p, list) else [{"role": "user", "content": p}]
    if r.get("api_url"):
        def lm(prompt):
            req = urllib.request.Request(
                r["api_url"], json.dumps({"model": r["api_model"], "messages": msgs_of(prompt), "temperature": r["temperature"]}).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ[r["api_key_env"]]})
            return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"]
        return lm
    from vllm import SamplingParams

    def lm(prompt):
        msgs = msgs_of(prompt)
        n = planner.llm.n_tokens(msgs, None)
        room = cfg["max_ctx"] - n - 32
        if room < 512:
            raise ValueError(f"reflection prompt {n} token, không đủ chỗ trong max_ctx={cfg['max_ctx']}")
        sp = SamplingParams(temperature=r["temperature"], max_tokens=min(r["max_tokens"], room), seed=cfg["seed"])
        return planner.llm.llm.chat([msgs], sp, use_tqdm=False)[0].outputs[0].text.strip()
    return lm


def update_manifest(cfg, key, info):
    d = Path(cfg["prompts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    f = d / "manifest.json"
    m = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    m[key] = {**m.get(key, {}), **info}
    f.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def optimize(cfg, planner, reflect, component="planner"):
    """Chạy GEPA cho một component, ghi prompts/gepa/kiet/<component>_vN.txt nếu tốt hơn seed. -> đường dẫn hoặc None."""
    import gepa
    dev = set(read(cfg["dev_ids"]).split())
    if component == "planner":
        head, seed = split_prompt(read(cfg["seed_prompt"]))
        sp = cfg.get("synth_path")
        synth = [json.loads(l) for l in open(sp, encoding="utf-8") if l.strip()] if sp and Path(sp).exists() else []
        adapter, (train, val) = Adapter(planner, cfg, "planner", head=head), split_data(planner.items, dev, cfg, synth)
    else:
        cfg = {**cfg, **{k: cfg[f"mapper_{k}"] for k in ("budget", "minibatch", "n_train", "n_val")}}
        head, seed, asr = None, read(cfg["mapper_seed"]), load_asr(cfg["asr_cache"])
        adapter = Adapter(planner, cfg, "mapper", system=read(cfg["mapper_planner_prompt"]), asr=asr)
        train, val = split_data(mapper_pool(planner.items, asr, cfg), dev, cfg)
    print(f"GEPA {component}: train {len(train)}, val {len(val)} (app lạ), budget {cfg['budget']} rollout"
          + (f", metric {cfg['mapper_metric']}" if component == "mapper" else ""))
    res = gepa.optimize(
        seed_candidate={component: seed}, trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflect, max_metric_calls=cfg["budget"], reflection_minibatch_size=cfg["minibatch"],
        candidate_selection_strategy="pareto", cache_evaluation=True, seed=cfg["seed"],
        run_dir=f"{cfg['run_dir']}/{component}", raise_on_exception=False)  # thư mục riêng: resume không lẫn component
    best = res.best_candidate[component]
    seed_val, best_val = res.val_aggregate_scores[0], res.val_aggregate_scores[res.best_idx]
    print(f"val (shaped): seed {seed_val:.4f} -> best {best_val:.4f}")
    if best == seed:
        print("Không có prompt nào hơn seed, không ghi file.")
        return None
    d = Path(cfg["prompts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(m.group(1)) for f in d.glob(f"{component}_v*.txt") if (m := re.fullmatch(rf"{component}_v(\d+)\.txt", f.name))), default=0)
    path = d / f"{component}_v{n}.txt"
    path.write_text(join_prompt(head, best) if head else best, encoding="utf-8")
    update_manifest(cfg, path.stem, {"file": path.name, "val_shaped_seed": round(seed_val, 4), "val_shaped_best": round(best_val, 4),
                                     "notes": f"GEPA budget={cfg['budget']}, seed={cfg['seed']}, chưa chấm trên dev: chạy mode eval"})
    return path


def eval_dev(cfg, planner, prompt_path, limit=None, items_path=None, mapper_path=None, use_asr=False):
    """Chấm trên dev (tập giữ lại) theo công thức BTC, ghi điểm theo nhóm + độ trễ vào manifest.
    items_path: file synth -> chỉ mẫu có nguồn thuộc dev, điểm theo nhãn full/missing_slot/no_app.
    use_asr / mapper_path: input là ASR 1-best (không Mapper) / ASR + Mapper, thay cho audio_text. mapper_path = "old":
    Mapper cũ của pipeline (Qwen3-0.6B + MAPPER_SYSTEM). Đủ 3 cấu hình ablation: không Mapper, Mapper cũ, Mapper mới.
    File kết quả lưu raw (ASR) và texts (sau Mapper) từng lượt: Khánh dùng để test entity guard trên đầu ra thật."""
    tpl, dev = read(prompt_path), set(read(cfg["dev_ids"]).split())
    asr = load_asr(cfg["asr_cache"]) if (use_asr or mapper_path) else None
    old = P.VLLM(P.MAPPER_MODEL, None, cfg["old_mapper_gpu_mem"], cfg["seed"], 2048, 256, {"enable_thinking": False},
                 cfg["eager"]) if mapper_path == "old" else None
    mapper = read(mapper_path) if mapper_path and not old else None
    src = [json.loads(l) for l in open(items_path, encoding="utf-8") if l.strip()] if items_path else planner.items
    items = [i for i in src if i["id"] in dev and (asr is None or i["id"] in asr)][:limit]
    name = Path(prompt_path).stem + ("+mapper_old" if old else f"+{Path(mapper_path).stem}" if mapper_path else "+asr" if use_asr else "") \
        + (f"@{Path(items_path).stem}" if items_path else "") + (f"#{limit}" if limit else "")
    sub, groups, secs = Path(cfg["run_dir"]) / f"eval_{name}.jsonl", {}, []
    sub.parent.mkdir(parents=True, exist_ok=True)
    with open(sub, "w", encoding="utf-8") as f:
        for k, it in enumerate(items, 1):
            t0 = time.time()
            raw = [nb[0] for nb in asr[it["id"]]] if asr else None
            if old:
                texts = [m or r for r, m in zip(raw, old.chat([P.mapper_conv(r) for r in raw]))]  # như pipeline.main
            else:
                texts = map_turns(planner.llm, mapper, [it], asr, planner.vocab)[0] if mapper else raw
            calls, h, _, _ = planner.run(it, tpl, texts)
            dt = time.time() - t0  # gồm cả Mapper
            m = evaluate_sample(it["label"]["tool_calls"], calls, h, planner.gold_hash(it), planner.catalog)
            groups.setdefault(it.get("synth", m["kind"]), []).append(m["score"])
            secs.append(dt)
            f.write(json.dumps({"id": it.get("synth_id", it["id"]), "tool_calls": calls, "state_hash": h, "raw": raw, "texts": texts,
                                "score": m["score"], "error": m["error"]}, ensure_ascii=False) + "\n")
            print(f"[{k}/{len(items)}] {it['id']} {m['score']:.2f} {m['error'] or ''} {dt:.1f}s", flush=True)
    mean = lambda v: round(sum(v) / max(len(v), 1), 4)
    info = {"file": Path(prompt_path).name, "mapper": "old (Qwen3-0.6B)" if old else Path(mapper_path).name if mapper_path else ("asr" if use_asr else "text"),
            "mean_score": mean([s for v in groups.values() for s in v]), **{k: mean(v) for k, v in sorted(groups.items())},
            "avg_latency_s": round(sum(secs) / max(len(secs), 1), 1),
            "notes": f"{len(items)} mẫu {'synth' if items_path else 'dev'}, verify_rounds={cfg['verify_rounds']}, "
                     f"max_ctx={cfg['max_ctx']}, đo trên GPU Kaggle (chậm hơn A30)"}
    print(json.dumps(info, ensure_ascii=False))
    update_manifest(cfg, name, info)  # --limit N ghi khoá riêng "...#N", không đè bản chấm đủ
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["asr", "run", "eval"])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--root", help="ghi đè root trong config")
    ap.add_argument("--budget", type=int, help="ghi đè budget (số rollout)")
    ap.add_argument("--component", choices=["planner", "mapper"], default="planner", help="mode run: prompt cần tối ưu")
    ap.add_argument("--prompt", help="mode eval: prompt Planner (mặc định seed_prompt)")
    ap.add_argument("--mapper", help='mode eval: prompt Mapper (input = ASR + Mapper), "old" = Mapper Qwen3-0.6B của pipeline')
    ap.add_argument("--asr", action="store_true", help="mode eval: input = ASR 1-best, không Mapper")
    ap.add_argument("--items", help="mode eval: file synth thay cho dev gốc")
    ap.add_argument("--limit", type=int, help="mode eval: chỉ chạy N mẫu đầu (không ghi manifest)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg.update({k: v for k, v in (("root", a.root), ("budget", a.budget), ("mapper_budget", a.budget)) if v})
    if a.mode == "asr":
        return build_asr_cache(cfg)
    if a.mapper == "old":  # chừa VRAM cho engine Qwen3-0.6B
        cfg["gpu_mem"] = round(cfg["gpu_mem"] - cfg["old_mapper_gpu_mem"] - 0.03, 2)
    pl = Planner(cfg)
    if a.mode == "run":
        optimize(cfg, pl, make_reflector(cfg, pl), a.component)
    else:
        default = cfg["mapper_planner_prompt"] if (a.mapper or a.asr) else cfg["seed_prompt"]
        eval_dev(cfg, pl, a.prompt or default, a.limit, a.items, a.mapper, a.asr)


if __name__ == "__main__":
    main()
