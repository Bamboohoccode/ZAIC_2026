"""Sinh 1 notebook Kaggle cho WSP, chạy trong 1 lần Commit (tối đa 12 giờ):

  wsp_single: bước 1 baseline 300 mẫu dev (think on), bước 2 baseline practice + tìm hướng (dirs.pt),
              bước 3-4 tự tìm bản sửa (có hạn giờ), bước 5 kiểm bản tốt nhất trên 300 mẫu dev, ô tay (tuỳ chọn).

Thay cho bản 2 notebook A/B (2026-11-09). Bỏ baseline think off: kết quả Kaggle 2026-11-09 cho thấy
cờ think off không đổi đầu ra (r00.jsonl của on và off giống hệt nhau).
Không gọi API ngoài trên Kaggle: chính sách sửa nằm trong optimize/wsp/auto_search.py (Opus viết sẵn).
Chạy: python optimize/wsp/make_kaggle_notebook.py --out-dir ../kaggle_upload
"""
import argparse, json
from pathlib import Path

CODE_ZIP = "zaic_code_20261111_v6.zip"
DATA_ZIP = "zaic_data_noaudio_20261107.zip"
MODEL_ID = "Ma7ee7/Qwen3.8_4B_Distilled"
MODEL_REV = "f33da883e0dadd9a4026556274e87489cd544d77"
NB_NAME = "wsp_single_20261111.ipynb"
NB_FAST = "wsp_fast_20261111.ipynb"


def sub(s):
    return (s.replace("%%CODE_ZIP%%", CODE_ZIP).replace("%%MODEL_ID%%", MODEL_ID)
             .replace("%%MODEL_REV%%", MODEL_REV).replace("%%DATA_ZIP%%", DATA_ZIP))


# ---------------------------------------------------------------- ô dùng chung
CELL_PATHS = r'''# ĐƯỜNG DẪN INPUT (chỉ sửa nếu cần):
#  - CODE_ZIP: phải TRÙNG tên file zip mã đã upload vào dataset (hiện: %%CODE_ZIP%%).
#  - Dữ liệu (%%DATA_ZIP%%) và train/train.jsonl: không cần sửa, ô tự tìm dưới /kaggle/input/**.
#  - Nếu assert báo lỗi: kiểm tra dataset có đúng tên zip ở trên và là bản v5 (có --label-rule).
import glob, os, shutil, subprocess, sys, zipfile, json, time
NB_T0 = time.time()  # mốc bắt đầu notebook, để tính hạn giờ
W = "/kaggle/working"; T = "/tmp"
CODE_ZIP = "%%CODE_ZIP%%"
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):   # trường hợp Kaggle không tự giải nén
    n = os.path.basename(z)
    if "zaic_code" in n:
        zipfile.ZipFile(z).extractall(f"{T}/code_src/{n[:-4]}")
    elif "zaic_data" in n:
        zipfile.ZipFile(z).extractall(f"{T}/data_root")
cands = sorted(glob.glob(f"{T}/code_src/**/optimize/wsp/auto_search.py", recursive=True)
               + glob.glob("/kaggle/input/**/optimize/wsp/auto_search.py", recursive=True))
cands = [p for p in cands if CODE_ZIP[:-4] in p] or cands   # ưu tiên đúng bản zip, tránh lấy nhầm bản cũ
assert cands, f"không thấy mã nguồn (optimize/wsp/auto_search.py): kiểm tra dataset có {CODE_ZIP}"
assert "--deadline-ts" in open(cands[0], encoding="utf-8").read(), f"mã nguồn cũ ({cands[0]}): cần {CODE_ZIP}"
REPO = W + "/ZAIC_2026"
shutil.copytree(os.path.dirname(os.path.dirname(os.path.dirname(cands[0]))), REPO, dirs_exist_ok=True)
tj = glob.glob(f"{T}/data_root/**/train/train.jsonl", recursive=True) or glob.glob("/kaggle/input/**/train/train.jsonl", recursive=True)
assert tj, "không thấy dữ liệu (train/train.jsonl)"
ROOT = os.path.dirname(os.path.dirname(tj[0]))
print("REPO =", REPO, "(từ", cands[0], ")"); print("ROOT =", ROOT, os.listdir(ROOT))'''

CELL_INSTALL = r'''!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q vllm==0.33.0   # bản đã chạy được trên Kaggle T4 ngày 09/11
import torch, importlib.metadata as md
DTYPE = "auto" if torch.cuda.get_device_capability(0) >= (8, 0) else "half"   # T4 không có bf16
GPUS = ",".join(str(i) for i in range(torch.cuda.device_count()))
print("torch", torch.__version__, "| vllm", md.version("vllm"), "| dtype:", DTYPE, "| GPU:", GPUS)'''

CELL_MODEL = r'''from huggingface_hub import snapshot_download
# cùng bản với máy cá nhân (docs/sources.md, DECISIONS 09/11); để ở /tmp cho khỏi bị lưu vào Output
MODEL_ID, MODEL_REV = "%%MODEL_ID%%", "%%MODEL_REV%%"
M = snapshot_download(MODEL_ID, revision=MODEL_REV, local_dir=T + "/model")
print(M, sorted(os.listdir(M)))'''

CELL_HELPERS = r'''def run(args):
    cmd = [sys.executable] + args
    print(">>", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO, check=True)
    print(f"   xong sau {time.time() - t0:.0f}s | đã chạy {(time.time() - NB_T0) / 3600:.2f} giờ", flush=True)

def head(src, n, dst):
    ids = [l.strip() for l in open(f"{REPO}/{src}", encoding="utf-8") if l.strip()][:n]
    open(dst, "w", encoding="utf-8").write("\n".join(ids) + "\n"); return dst

def summary(p):
    s = json.load(open(p, encoding="utf-8")); s.pop("per_sample", None); return s

def pack(name="wsp_results.zip"):  # gọi sau mỗi bước: Commit bị dừng giữa chừng vẫn còn kết quả các bước đã xong
    out = f"{W}/{name}"
    files = glob.glob(f"{W}/dev_*/*") + glob.glob(f"{W}/search/*") + \
            [f"{W}/{x}" for x in ("dirs.pt", "dirs.stats.json", "best_spec.json", "run_config.json")]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.isfile(f):
                z.write(f, os.path.relpath(f, W))
    print("->", out, round(os.path.getsize(out) / 1e6, 2), "MB", flush=True)

DEV = head("eval/splits/dev_ids.txt", 10, W + "/dev_quick.txt") if QUICK else f"{REPO}/eval/splits/dev_ids.txt"
PRACTICE = head("optimize/wsp/practice100_20261107.txt", 20, W + "/practice_quick.txt") if QUICK else f"{REPO}/optimize/wsp/practice100_20261107.txt"
K = ["--base-model", M, "--root", ROOT, "--llm-dtype", DTYPE, "--gpu-mem", "0.9", "--gpus", GPUS, "--model-out", T + "/wsp_model"]
S = f"{W}/search"
print("QUICK:", QUICK, "| MODE:", MODE, "| DEV:", DEV, "| PRACTICE:", PRACTICE)'''

CELL_CONFIG = r'''# ===== CẤU HÌNH =====
QUICK = True          # True: chạy thử (10 mẫu dev, 20 mẫu practice) để đo tốc độ. False: chạy đầy đủ (Save & Run All)
MODE = "on"           # WSP nhắm vào thinking bị cắt/lặp nên dùng "on". Cờ "off" đã kiểm: không đổi đầu ra, không đo lại.
BUDGET = 12           # số vòng sửa tối đa (blog: 10, 25, 50). Hạn giờ bên dưới thường dừng trước
HOURS = 11.0          # tổng giờ cho phép dùng (Commit tối đa 12 giờ; chừa 1 giờ dự phòng)
FINAL_RESERVE_H = 3.5 # giờ chừa cho bước 5 (300 mẫu dev, ước 35 giây/mẫu, CHƯA XÁC MINH)'''

CELL_STEP1 = r'''# Bước 1: baseline trên dev, model gốc
json.dump({"quick": QUICK, "mode": MODE, "model": MODEL_ID, "revision": MODEL_REV, "code_zip": CODE_ZIP,
           "dev": os.path.basename(DEV), "practice": os.path.basename(PRACTICE)},
          open(W + "/run_config.json", "w"), indent=1)
run(["optimize/wsp/run_round.py", "--round", "0", "--spec", "none", "--think", MODE, "--ids", DEV, "--work", f"{W}/dev_{MODE}"] + K)
pack()
s = summary(f"{W}/dev_{MODE}/r00.summary.json")
print(f"== baseline dev think {MODE}: mean={s['mean_score']} by_kind={s['by_kind']} flags={s['flags']}")'''

CELL_STEP2 = r'''# Bước 2: baseline trên practice (= vòng 0 của bước tìm), rồi tìm hướng gây lỗi
run(["optimize/wsp/run_round.py", "--round", "0", "--spec", "none", "--think", MODE, "--ids", PRACTICE, "--work", S] + K)
exp = S + "/r00.export.jsonl"
labels = [json.loads(l)["label"] for l in open(exp, encoding="utf-8")]
NB, NG = labels.count("bad"), labels.count("good")
N = summary(S + "/r00.summary.json").get("n") or len([l for l in open(PRACTICE, encoding="utf-8") if l.strip()])
print(f"practice: bad {NB} | good {NG} | không dùng (sai vì lý do khác) {N - NB - NG} | flags {summary(S + '/r00.summary.json')['flags']}")
HAVE_DIRS = NB >= 2 and NG >= 2
if HAVE_DIRS:
    run(["optimize/wsp/directions.py", "--model", M, "--root", ROOT, "--export", exp, "--think", MODE,
         "--out", W + "/dirs.pt", "--stats", W + "/dirs.stats.json"])
    print(json.dumps(json.load(open(W + "/dirs.stats.json"))["layers"][:10], indent=0))
elif NB < 2:
    print(f"THIẾU MẪU BAD ({NB} < 2): ít thinking bị cắt/lặp ở chế độ '{MODE}'. WSP ở đây chỉ nhắm vào lỗi cắt/lặp, "
          "nên không có gì để gỡ. Bỏ qua bước sửa; quyết định go/no-go dựa trên bảng baseline.")
else:
    print(f"THIẾU MẪU GOOD ({NG} < 2): lỗi cắt/lặp nhiều ({NB} mẫu) nhưng quá ít mẫu đúng hoàn toàn (điểm 1,0) để so hướng. "
          "Lỗi có thật, chỉ thiếu mẫu đối chứng. Nếu đang QUICK: chạy bản đầy đủ (100 mẫu practice). Nếu bản đầy đủ vẫn thiếu: gửi kết quả cho Opus.")
pack()'''

CELL_STEP34 = r'''# Bước 3-4: tự tìm bản sửa trên practice (vòng 0 đã chạy ở bước 2, không chạy lại)
BUDGET_RUN = 2 if QUICK else BUDGET
DEADLINE = NB_T0 + (HOURS - FINAL_RESERVE_H) * 3600
if HAVE_DIRS:
    print(f"hạn bắt đầu vòng mới: {(DEADLINE - NB_T0) / 3600:.1f} giờ sau khi mở notebook | budget {BUDGET_RUN}")
    run(["optimize/wsp/auto_search.py", "--base-model", M, "--root", ROOT, "--dirs", W + "/dirs.pt", "--stats", W + "/dirs.stats.json",
         "--work", S, "--model-out", T + "/wsp_model", "--select-ids", PRACTICE, "--think", MODE, "--llm-dtype", DTYPE,
         "--gpus", GPUS, "--budget", str(BUDGET_RUN), "--deadline-ts", str(DEADLINE), "--round-sec-init", "3600"])
BEST = json.load(open(S + "/best_spec.json")) if os.path.exists(S + "/best_spec.json") else []
print("best_spec:", BEST)
pack()'''

CELL_STEP5 = r'''# Bước 5: kiểm bản tốt nhất trên dev (cùng tập, cùng chế độ với baseline). Không có bản sửa nào qua luật giữ thì bỏ qua.
if BEST:
    json.dump(BEST, open(W + "/best_spec.json", "w"), indent=1)
    run(["optimize/wsp/run_round.py", "--round", "1", "--spec", W + "/best_spec.json", "--dirs", W + "/dirs.pt",
         "--think", MODE, "--ids", DEV, "--work", f"{W}/dev_{MODE}"] + K)
rows = [json.loads(l) for l in open(f"{W}/dev_{MODE}/wsp_log.jsonl", encoding="utf-8")]
print(f"{'cấu hình':28s} {'n':>4s} {'mean':>7s} {'gen_tokens':>10s}  flags")
for r in rows:
    name = f"think {r['think']} " + ("WSP " + str([(o['layer'], o['top_k_cols'], o['alpha']) for o in r['spec']]) if r["spec"] else "gốc")
    print(f"{name[:28]:28s} {r['n']:4d} {r['mean_score']:7.4f} {str(r['gen_tokens_mean']):>10s}  {r['flags']}")
print("Go/no-go (plan bước 5): WSP phải cao hơn baseline gốc trên dev, gen_tokens không tăng. QUICK chỉ để đo tốc độ, không kết luận.")
print(f"Tổng thời gian: {(time.time() - NB_T0) / 3600:.2f} giờ")
pack()'''

CELL_ROUND = r'''spec = []  # THAY bằng spec thật do Opus viết. Spec rỗng thì ô này bỏ qua (không báo lỗi, để Save & Run All vẫn hoàn tất)
if not spec:
    print("Bỏ qua vòng tay: chưa có spec. Dán spec vào `spec` rồi chạy lại ô này.")
elif not os.path.exists(W + "/dirs.pt"):
    print("Bỏ qua vòng tay: không có dirs.pt (không tìm được hướng).")
else:
    ROUND = 1 + max(json.loads(l)["round"] for l in open(S + "/wsp_log.jsonl", encoding="utf-8"))
    sp = f"{S}/r{ROUND:02d}.spec.json"; json.dump(spec, open(sp, "w"), indent=1)
    run(["optimize/wsp/run_round.py", "--round", str(ROUND), "--spec", sp, "--dirs", W + "/dirs.pt",
         "--think", MODE, "--ids", PRACTICE, "--work", S] + K)
    print(open(S + "/wsp_log.jsonl", encoding="utf-8").read().splitlines()[-1])
    pack()'''

MD_HEAD = """# WSP: baseline, tìm hướng gây lỗi, tự tìm bản sửa, kiểm trên dev (một notebook)

Model: `%%MODEL_ID%%` (revision `%%MODEL_REV%%`). Cài đặt: **GPU T4 x2**, **Internet On**.
Input: dataset riêng tư chứa `%%CODE_ZIP%%` và `%%DATA_ZIP%%`. Không cần output notebook nào khác.

Các bước:
1. Baseline trên dev (300 mẫu; QUICK: 10), think on.
2. Baseline trên practice (100 mẫu train, không thuộc dev; QUICK: 20), rồi tìm hướng gây lỗi (`dirs.pt`).
3-4. `auto_search.py` thử sửa trên practice, chỉ giữ khi qua luật cố định (DECISIONS l). Không bắt đầu vòng mới
   nếu vòng đó sẽ vượt `HOURS - FINAL_RESERVE_H` giờ.
5. Kiểm bản tốt nhất trên dev (cùng tập, cùng chế độ), in bảng so với baseline.

Nhãn: `bad` = thinking bị cắt hoặc lặp; `good` = điểm đúng 1,0; mẫu sai vì lý do khác không dùng để tìm hướng.
Chạy thử trước với `QUICK = True`, rồi `QUICK = False` và Save & Run All (Commit).
Kết quả: `/kaggle/working/wsp_results.zip` (ghi lại sau mỗi bước). Không gọi API ngoài; không chạy trên test BTC."""

MD_TAY = """## Vòng sửa tay (tuỳ chọn, sau khi Opus đọc kết quả trên máy)

Opus (chạy trên máy, không phải trên Kaggle) đọc `wsp_results.zip` và viết spec mới. Dán spec vào ô dưới, chạy.
Vòng tay đo trên practice (cùng log `search/`), không đo trên dev. Gửi lại dòng cuối `search/wsp_log.jsonl`."""


# ---------------------------------------------------------------- notebook nhanh (khoảng 1 giờ, chạy tương tác)
MD_FAST = """# WSP bản nhanh: 10 mẫu, nhãn theo điểm, 3 vòng sửa (khoảng 1 giờ)

Model: `%%MODEL_ID%%` (revision `%%MODEL_REV%%`). Cài đặt: **GPU T4 x2**, **Internet On**. Chạy **Run All** (tương tác).
Input: dataset riêng tư chứa `%%CODE_ZIP%%` và `%%DATA_ZIP%%`.
Đường dẫn cần kiểm: ô "Thiết lập đường dẫn" (dòng CODE_ZIP). Các đường dẫn khác tự tìm dưới `/kaggle/input`.

Vì sao có bản này: lần chạy 09/11 không ra bản sửa vì practice có `bad 0` (model không bị cắt/lặp thinking),
không phải vì hết giờ. Bản này đổi luật nhãn (`--label-rule score`): **bad = mẫu điểm 0, good = mẫu điểm 1,0**.
Tập chạy: 10 mẫu practice (`fast10_20261111.txt`, 5 mẫu mỗi GPU), chọn từ 20 mẫu QUICK ngày 09/11 gồm 2 mẫu đúng và 8 mẫu sai.

1. Vòng 0 (model gốc) trên 10 mẫu, gán nhãn, tìm hướng (`dirs.pt`).
2. `auto_search.py`: 3 vòng sửa, mỗi vòng trừ hướng khỏi 64 cột MLP ở 1 tầng (3 tầng có Cohen's d cao nhất).
3. Bảng: từng bản sửa, điểm so với vòng 0, GIỮ hay bỏ theo luật cố định (tăng >= 0,01, gen_tokens <= 105%).
4. Nếu có bản được giữ và còn giờ: kiểm trên 10 mẫu dev.

Giới hạn: 10 mẫu là quá ít để kết luận go/no-go; 10 mẫu này được chọn theo kết quả cũ nên điểm vòng 0 không đại diện.
Kết quả: `/kaggle/working/wsp_results.zip`. Không gọi API ngoài; không chạy trên test BTC."""

CELL_CONFIG_FAST = r"""# ===== CẤU HÌNH =====
QUICK = True          # bản nhanh luôn dùng tập nhỏ
MODE = "on"
BUDGET = 3            # số vòng sửa (mỗi vòng ước 8-10 phút, CHƯA XÁC MINH)
TOP_LAYERS = 3        # số tầng ứng viên (pha A thử từng tầng)
HOURS = 1.3           # không bắt đầu vòng sửa mới sau mốc này (giờ, tính từ lúc chạy ô 2)
DEV_CHECK_BEFORE_H = 1.0  # chỉ kiểm dev (2 lượt x 10 mẫu) nếu lúc đó chưa quá mốc này"""

CELL_FAST_A = r"""# Bước 1: vòng 0 trên 10 mẫu, nhãn theo điểm, tìm hướng
assert "hidden_mean" in open(f"{REPO}/optimize/wsp/directions.py", encoding="utf-8").read(), f"mã nguồn cũ: cần {CODE_ZIP}"
SEL = f"{REPO}/optimize/wsp/fast10_20261111.txt"
json.dump({"notebook": "fast", "mode": MODE, "model": MODEL_ID, "revision": MODEL_REV, "code_zip": CODE_ZIP,
           "label_rule": "score", "select": os.path.basename(SEL), "budget": BUDGET, "top_layers": TOP_LAYERS},
          open(W + "/run_config.json", "w"), indent=1)
exp = S + "/r00.export.jsonl"
if os.path.exists(exp) and os.path.exists(S + "/wsp_log.jsonl"):   # chạy lại trong cùng phiên: dùng lại vòng 0
    print("Dùng lại vòng 0 đã có trong", S)
else:
    run(["optimize/wsp/run_round.py", "--round", "0", "--spec", "none", "--think", MODE, "--ids", SEL, "--work", S,
         "--label-rule", "score"] + K)
recs = [json.loads(l) for l in open(exp, encoding="utf-8")]
NB, NG = sum(r["label"] == "bad" for r in recs), sum(r["label"] == "good" for r in recs)
print(f"nhãn: bad {NB} | good {NG} |", [(r["id"], r["label"], r["score"]) for r in recs])
HAVE_DIRS = NB >= 2 and NG >= 2
if HAVE_DIRS:
    run(["optimize/wsp/directions.py", "--model", M, "--root", ROOT, "--export", exp, "--think", MODE,
         "--out", W + "/dirs.pt", "--stats", W + "/dirs.stats.json"])
    print(json.dumps(json.load(open(W + "/dirs.stats.json"))["layers"][:6], indent=0))
else:
    print(f"KHÔNG ĐỦ NHÃN (bad {NB}, good {NG}): không tìm được hướng. Gửi lại wsp_results.zip.")
pack()"""

CELL_FAST_B = r"""# Bước 2: 3 vòng sửa (pha A: từng tầng một)
DEADLINE = NB_T0 + HOURS * 3600
if HAVE_DIRS:
    run(["optimize/wsp/auto_search.py", "--base-model", M, "--root", ROOT, "--dirs", W + "/dirs.pt", "--stats", W + "/dirs.stats.json",
         "--work", S, "--model-out", T + "/wsp_model", "--select-ids", SEL, "--think", MODE, "--llm-dtype", DTYPE,
         "--gpus", GPUS, "--budget", str(BUDGET), "--top-layers", str(TOP_LAYERS), "--min-d", "0",
         "--deadline-ts", str(DEADLINE), "--round-sec-init", "600"])
BEST = json.load(open(S + "/best_spec.json")) if os.path.exists(S + "/best_spec.json") else []
pack()"""

CELL_FAST_C = r"""# Bước 3: bảng các bản sửa đã thử (trên 10 mẫu chọn)
rows = [json.loads(l) for l in open(S + "/wsp_log.jsonl", encoding="utf-8")]
base = rows[0]
print(f"{'vòng':>4s} {'bản sửa (tầng, cột, alpha)':32s} {'mean':>7s} {'so vòng 0':>9s} {'gen_tokens':>10s} {'phần tử đổi':>11s}  flags")
for r in rows:
    name = str([(o['layer'], o['top_k_cols'], o['alpha']) for o in r['spec']]) if r["spec"] else "gốc (vòng 0)"
    print(f"{r['round']:4d} {name[:32]:32s} {r['mean_score']:7.4f} {r['mean_score'] - base['mean_score']:+9.4f} "
          f"{str(r['gen_tokens_mean']):>10s} {r['n_changed']:11d}  {r['flags']}")
tried = [r for r in rows if r["spec"]]
if tried:
    top = max(tried, key=lambda r: r["mean_score"])
    json.dump(top["spec"], open(W + "/best_tried_spec.json", "w"), indent=1)
    print(f"\nĐã chạy {len(tried)} bản sửa. Bản điểm cao nhất: vòng {top['round']}, mean {top['mean_score']} (vòng 0: {base['mean_score']}).")
print("best_spec (qua luật giữ):", BEST if BEST else "[] = chưa có bản nào tăng >= 0,01 mà không tăng gen_tokens quá 5%")
print(f"Đã chạy {(time.time() - NB_T0) / 3600:.2f} giờ")"""

CELL_FAST_D = r"""# Bước 4 (tuỳ chọn): kiểm bản được giữ trên 10 mẫu dev. Bỏ qua nếu không có bản được giữ hoặc đã muộn.
el = (time.time() - NB_T0) / 3600
if BEST and el < DEV_CHECK_BEFORE_H:
    json.dump(BEST, open(W + "/best_spec.json", "w"), indent=1)
    run(["optimize/wsp/run_round.py", "--round", "0", "--spec", "none", "--think", MODE, "--ids", DEV, "--work", f"{W}/dev_{MODE}"] + K)
    run(["optimize/wsp/run_round.py", "--round", "1", "--spec", W + "/best_spec.json", "--dirs", W + "/dirs.pt",
         "--think", MODE, "--ids", DEV, "--work", f"{W}/dev_{MODE}"] + K)
    for r in [json.loads(l) for l in open(f"{W}/dev_{MODE}/wsp_log.jsonl", encoding="utf-8")]:
        print("dev 10 mẫu |", "WSP" if r["spec"] else "gốc", "| mean", r["mean_score"], "| gen_tokens", r["gen_tokens_mean"], "|", r["flags"])
else:
    print(f"Bỏ qua kiểm dev: best_spec={'có' if BEST else 'không'}, đã chạy {el:.2f} giờ (mốc {DEV_CHECK_BEFORE_H}).")
pack()"""


def build_fast():
    return notebook([cell("markdown", MD_FAST)] + [cell("code", c) for c in
                    (CELL_CONFIG_FAST, CELL_PATHS, CELL_INSTALL, CELL_MODEL, CELL_HELPERS,
                     CELL_FAST_A, CELL_FAST_B, CELL_FAST_C, CELL_FAST_D)])


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": sub(src).splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "cells": cells}


def build():
    cells = [cell("markdown", MD_HEAD), cell("code", CELL_CONFIG), cell("code", CELL_PATHS),
             cell("code", CELL_INSTALL), cell("code", CELL_MODEL), cell("code", CELL_HELPERS),
             cell("code", CELL_STEP1), cell("code", CELL_STEP2), cell("code", CELL_STEP34), cell("code", CELL_STEP5),
             cell("markdown", MD_TAY), cell("code", CELL_ROUND)]
    return notebook(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, nb in ((NB_NAME, build()), (NB_FAST, build_fast())):
        json.dump(nb, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("wrote", out / name, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
