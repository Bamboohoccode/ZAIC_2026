"""Tự tìm bản sửa trọng số WSP trên Kaggle, không gọi API ngoài. Chính sách do Opus viết sẵn (trên máy), Kaggle chỉ chạy.

Thay cho vai "người sửa" trong blog Sparks of Weight-Space Programming (Thí nghiệm 8): blog trừ một hướng gây vòng lặp khỏi
64 cột đầu ra của MLP ở tầng 15 và 19 (Qwen3.5-0.8B, 24 tầng); chỉ giữ thay đổi khi kết quả tốt hơn.
Ở đây mỗi vòng = một forward pass trên tập chọn (`--select-ids`, mặc định practice100: train, không thuộc dev):

  Pha A (từng tầng):  project_out 64 cột, alpha 1, ở từng tầng trong `--top-layers` tầng có Cohen's d cao nhất (d >= --min-d).
  Pha B (ghép tầng):  bắt đầu từ tầng tốt nhất pha A; thêm lần lượt tầng kế tiếp (theo điểm pha A), giữ nếu tốt hơn.
                      Nếu pha A không có tầng nào tốt hơn baseline: thử một lần cặp 2 tầng d cao nhất (như blog), rồi sang pha C.
  Pha C (độ mạnh):    trên bản tốt nhất, thử top_k 128, top_k 32, alpha 0.5.
Dừng khi hết `--budget` vòng (không tính vòng 0).

Luật giữ (cố định trước khi chạy, không chỉnh theo kết quả):
a) mean_score >= điểm tốt nhất hiện tại + `--min-gain`;
b) gen_tokens TB mỗi mẫu <= baseline x `--max-token-ratio` (thay cho "thời gian không tăng");
c) số mẫu pipeline_error không tăng so với baseline.
Hoà điểm: không đổi (ưu tiên ít phép sửa).

Đầu ra trong `--work`: wsp_log.jsonl (mỗi vòng một dòng, do run_round ghi), search_log.jsonl (quyết định từng vòng),
best_spec.json (bản tốt nhất, [] nếu không có bản nào qua luật giữ), r*.summary.json.
Kiểm cuối trên 300 mẫu dev làm riêng (notebook), không dùng dev trong vòng tìm.

Chạy: python optimize/wsp/auto_search.py --base-model /tmp/Qwen3-4B --root ROOT --dirs dirs.pt --stats dirs.stats.json \
        --work /kaggle/working/search --model-out /tmp/wsp_model --think on --llm-dtype half --gpus 0,1 --budget 12
Kiểm luồng không GPU: thêm --dry-run (điểm giả, không chạy model).
"""
import argparse, json, random, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


class Search:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work); self.work.mkdir(parents=True, exist_ok=True)
        self.log = self.work / "wsp_log.jsonl"
        self.slog = self.work / "search_log.jsonl"
        self.rounds_used = 0
        self.round_sec = a.round_sec_init  # thời gian vòng lâu nhất đã đo (ước lượng cho vòng kế)
        self.cache = {}  # khoá spec -> dòng log (không chạy lại spec đã đo)
        if self.log.exists():  # chạy tiếp sau khi bị ngắt
            for l in open(self.log, encoding="utf-8"):
                r = json.loads(l)
                self.cache[self.key(r["spec"])] = r
                self.rounds_used = max(self.rounds_used, r["round"])

    @staticmethod
    def key(spec):
        return json.dumps(sorted(spec, key=lambda o: json.dumps(o, sort_keys=True)), sort_keys=True)

    def evaluate(self, spec, why):
        k = self.key(spec)
        if k in self.cache:
            return self.cache[k]
        if spec and self.rounds_used >= self.a.budget:
            return None
        if spec and self.a.deadline_ts and time.time() + self.round_sec > self.a.deadline_ts:
            print(f"\n=== dừng: vòng kế ước {self.round_sec:.0f}s sẽ vượt hạn thời gian; giữ kết quả hiện có", flush=True)
            return None
        rnd = 0 if not spec else self.rounds_used + 1
        if self.a.dry_run:
            row = self.fake(spec, rnd)
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            sp = self.work / f"r{rnd:02d}.spec.json"
            json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            cmd = [sys.executable, str(HERE / "run_round.py"), "--base-model", self.a.base_model, "--round", str(rnd),
                   "--spec", str(sp) if spec else "none", "--ids", self.a.select_ids, "--root", self.a.root,
                   "--work", str(self.work), "--think", self.a.think, "--llm-dtype", self.a.llm_dtype,
                   "--gpu-mem", self.a.gpu_mem, "--gpus", self.a.gpus, "--verify-rounds", self.a.verify_rounds]
            if spec:
                cmd += ["--dirs", self.a.dirs]
            if self.a.model_out:
                cmd += ["--model-out", self.a.model_out]
            if self.a.llm_gold:
                cmd += ["--llm-gold"]
            print(f"\n=== vòng {rnd}: {why}", flush=True)
            t0 = time.time()
            subprocess.run(cmd, cwd=REPO, check=True)
            self.round_sec = max(self.round_sec, time.time() - t0)
            row = [json.loads(l) for l in open(self.log, encoding="utf-8")][-1]
        if spec:
            self.rounds_used = rnd
        self.cache[k] = row
        return row

    def fake(self, spec, rnd):  # --dry-run: điểm giả có cấu trúc, để kiểm luồng chính sách
        rng = random.Random(self.key(spec))
        gain = sum(0.03 if o["layer"] in (self.layers[:2] if hasattr(self, "layers") else []) else -0.01 for o in spec)
        return {"round": rnd, "spec": spec, "mean_score": round(0.5 + gain + rng.uniform(-0.01, 0.01), 4),
                "gen_tokens_mean": 900 - 50 * len(spec), "flags": {"pipeline_error": 0}, "n_changed": 64 * 2560 * len(spec)}

    def ok(self, row, best, base):
        if row is None:
            return False
        if row["mean_score"] < best["mean_score"] + self.a.min_gain - 1e-9:
            return False
        bt, rt = base.get("gen_tokens_mean"), row.get("gen_tokens_mean")
        if bt and rt and rt > bt * self.a.max_token_ratio:
            return False
        return row["flags"].get("pipeline_error", 0) <= base["flags"].get("pipeline_error", 0)

    def note(self, phase, spec, row, kept):
        rec = {"phase": phase, "round": row["round"] if row else None, "spec": spec,
               "mean_score": row["mean_score"] if row else None, "gen_tokens_mean": row.get("gen_tokens_mean") if row else None,
               "kept": kept}
        with open(self.slog, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{phase}] r{rec['round']} mean={rec['mean_score']} tokens={rec['gen_tokens_mean']} "
              f"{'GIỮ' if kept else 'bỏ'} spec={[(o['layer'], o.get('top_k_cols'), o.get('alpha')) for o in spec]}", flush=True)

    def run(self):
        a = self.a
        stats = json.load(open(a.stats, encoding="utf-8"))
        layers = [s["layer"] for s in stats["layers"] if s["cohen_d"] >= a.min_d][: a.top_layers]
        self.layers = layers
        print("tầng ứng viên (theo Cohen's d trên practice):", layers)
        base = self.evaluate([], "baseline")
        best, best_spec = base, []
        self.note("baseline", [], base, True)
        if not layers:
            print("Không có tầng nào có d >= min-d: không sửa.")
            return self.finish(best_spec, base, best)

        def op(l, k=64, alpha=1.0):
            return {"op": "project_out", "layer": l, "direction_layer": l, "top_k_cols": k, "alpha": alpha}

        # Pha A
        single = {}
        for l in layers:
            row = self.evaluate([op(l)], f"pha A, tầng {l}")
            if row is None:
                break
            single[l] = row
            kept = self.ok(row, best, base)
            self.note("A", [op(l)], row, kept)
            if kept:
                best, best_spec = row, [op(l)]
        # Pha B
        order = sorted(single, key=lambda l: -single[l]["mean_score"])
        if best_spec:
            for l in order:
                if any(o["layer"] == l for o in best_spec):
                    continue
                cand = best_spec + [op(l)]
                row = self.evaluate(cand, f"pha B, thêm tầng {l}")
                if row is None:
                    break
                kept = self.ok(row, best, base)
                self.note("B", cand, row, kept)
                if kept:
                    best, best_spec = row, cand
        elif len(layers) >= 2:
            cand = [op(layers[0]), op(layers[1])]
            row = self.evaluate(cand, "pha B, cặp 2 tầng d cao nhất (như blog)")
            if row is not None:
                kept = self.ok(row, best, base)
                self.note("B", cand, row, kept)
                if kept:
                    best, best_spec = row, cand
        # Pha C
        if best_spec:
            for k, alpha, why in [(128, 1.0, "128 cột"), (32, 1.0, "32 cột"), (64, 0.5, "alpha 0.5")]:
                cand = [op(o["layer"], k, alpha) for o in best_spec]
                row = self.evaluate(cand, f"pha C, {why}")
                if row is None:
                    break
                kept = self.ok(row, best, base)
                self.note("C", cand, row, kept)
                if kept:
                    best, best_spec = row, cand
        return self.finish(best_spec, base, best)

    def finish(self, best_spec, base, best):
        json.dump(best_spec, open(self.work / "best_spec.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        res = {"rounds_used": self.rounds_used, "budget": self.a.budget, "best_spec": best_spec,
               "baseline_mean": base["mean_score"], "best_mean": best["mean_score"],
               "baseline_gen_tokens": base.get("gen_tokens_mean"), "best_gen_tokens": best.get("gen_tokens_mean"),
               "select_ids": Path(self.a.select_ids).name, "think": self.a.think}
        json.dump(res, open(self.work / "search_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\nKẾT QUẢ TÌM:", json.dumps(res, ensure_ascii=False))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-out")
    ap.add_argument("--select-ids", default=str(HERE / "practice100_20261107.txt"))
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--llm-dtype", default="auto")
    ap.add_argument("--gpu-mem", default="0.9")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--verify-rounds", default="2")
    ap.add_argument("--budget", type=int, default=12, help="số vòng có sửa tối đa (không tính vòng 0)")
    ap.add_argument("--deadline-ts", type=float, default=0.0, help="mốc unix time: không bắt đầu vòng mới nếu vòng đó (ước lượng) sẽ vượt mốc này. 0 = không giới hạn")
    ap.add_argument("--round-sec-init", type=float, default=3600.0, help="ước lượng thời gian một vòng trước khi đo được vòng nào")
    ap.add_argument("--top-layers", type=int, default=6)
    ap.add_argument("--min-d", type=float, default=0.5)
    ap.add_argument("--min-gain", type=float, default=0.01, help="mức tăng mean_score tối thiểu để giữ (0,01 = 1 mẫu trên 100)")
    ap.add_argument("--max-token-ratio", type=float, default=1.05)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--llm-gold", action="store_true", help="kiểm luồng thật (áp sửa, lưu, chạy, chấm) với replay nhãn, không GPU")
    a = ap.parse_args()
    Search(a).run()


if __name__ == "__main__":
    main()
