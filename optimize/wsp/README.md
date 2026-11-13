# WSP: weight-space programming trên Qwen3-4B

Thí nghiệm weight-space programming. [Tiên]
Phương pháp: Sparks of Weight-Space Programming (Thí nghiệm 8). Nguồn và giới hạn: `docs/sources.md`.
Chỉ làm WSP, không làm LoRA (người dùng quyết định 2026-11-07).

## 1. Thành phần

a) `weights.py`: đọc tensor qua mmap, phép `project_out` (trừ hướng d khỏi cột của `down_proj`), `scale_cols`, lưu model đầy đủ cho vLLM (shard không sửa thì hardlink).
b) `make_subset.py`: tập đo `dev_subset50_20261107.txt` (50 mẫu dev) và tập practice `practice100_20261107.txt` (100 mẫu train không thuộc dev). Seed 42.
c) `analyze.py`: điểm theo `eval/scorer.py` và cờ lỗi từ trace (thinking bị cắt, lặp, sai định dạng `<tool_call>`, không gọi tool khi gold có call).
d) `directions.py`: hướng gây lỗi theo từng tầng = trung bình trạng thái ẩn mẫu lỗi trừ mẫu đúng, kèm Cohen's d. Cần GPU.
e) `run_round.py`: một vòng = áp sửa, lưu, chạy agent trên tập id, chấm, ghi `wsp_log.jsonl`. Mặc định `--runner batch`.
f) `batch_agent.py`: agent chạy theo lô (cùng logic `pipeline.run_agent`, mọi mẫu song song, chia cho nhiều GPU). Kiểm bằng `test_batch_agent.py`. Sửa `run_agent`/`plan_step` trong pipeline thì sửa theo và chạy lại test.
g) `auto_search.py`: tự tìm bản sửa trên Kaggle (không gọi API), luật giữ cố định. Xem docstring.
h) `make_kaggle_notebook.py`: sinh notebook chạy trọn bước 1 đến 5. Hướng dẫn: `kaggle_upload/HUONG_DAN.md`.

Sửa 2026-11-07: phần chạy Kaggle dùng notebook mới (mục 3 và 4 bên dưới là cách chạy tay từng bước, vẫn dùng được). Vòng tìm đo trên practice, không đo trên dev 50 (`docs/DECISIONS.md` mục k).

## 2. Nguyên tắc chống rò rỉ

a) Tìm hướng chỉ trên tập practice (train, không thuộc dev).
b) Đo mỗi vòng trên tập 50 mẫu dev. Bản sửa cuối kiểm lại trên toàn bộ 300 mẫu dev.
c) Không chạy WSP trên test BTC, không đưa log test cho bước sửa.

## 3. Chạy trên Kaggle

Cần: GPU T4 (vLLM không chạy trên P100), Internet bật để cài vLLM và tải model.
Dữ liệu BTC và mã nguồn đưa lên dưới dạng dataset **riêng tư** (người dùng tự tải lên, vì đây là đưa dữ liệu BTC lên dịch vụ ngoài).

```bash
pip install -q vllm
cp -r /kaggle/input/<dataset-ma-nguon>/ZAIC_2026 /kaggle/working/ZAIC_2026 && cd /kaggle/working/ZAIC_2026
mkdir -p data/raw && unzip -q /kaggle/input/<dataset-du-lieu>/train+public_test.zip -d data/raw -x "__MACOSX/*" "*/audio/*"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', local_dir='/kaggle/working/Qwen3-4B')"
M=/kaggle/working/Qwen3-4B; K="--llm-dtype half --gpu-mem 0.9"
```

Bước 1. Baseline trên tập đo, cả hai chế độ thinking (để không quy nhầm lợi ích của việc tắt thinking cho WSP):
```bash
python optimize/wsp/run_round.py --base-model $M --round 0 --spec none --think on  $K --work /kaggle/working/wsp_on
python optimize/wsp/run_round.py --base-model $M --round 0 --spec none --think off $K --work /kaggle/working/wsp_off
```

Bước 2. Baseline trên tập practice để lấy mẫu lỗi và mẫu đúng, rồi tìm hướng:
```bash
python optimize/wsp/run_round.py --base-model $M --round 0 --spec none --think on $K \
  --ids optimize/wsp/practice100_20261107.txt --work /kaggle/working/practice
python optimize/wsp/directions.py --model $M --export /kaggle/working/practice/r00.export.jsonl \
  --out /kaggle/working/wsp_on/dirs.pt --stats /kaggle/working/wsp_on/dirs.stats.json --think on
```

Bước 3. Vòng sửa (mỗi vòng một spec trong `edits/`):
```bash
python optimize/wsp/run_round.py --base-model $M --round 1 --spec optimize/wsp/edits/r01.json \
  --dirs /kaggle/working/wsp_on/dirs.pt --think on $K --work /kaggle/working/wsp_on
```

Mẫu spec (`edits/r01.json`):
```json
[{"op": "project_out", "layer": 18, "direction_layer": 18, "top_k_cols": 64, "alpha": 1.0, "note": "tầng có Cohen's d cao nhất"}]
```

## 4. Vòng lặp với người sửa (Opus)

a) Sau bước 1 và 2: gửi lại cho Opus các tệp `r00.summary.json` (cả hai chế độ) và `dirs.stats.json`.
b) Opus đọc phân bố lỗi và các tầng có độ tách cao, viết spec vòng kế (tầng, số cột, alpha).
c) Chạy `run_round.py` với spec đó, gửi lại `rNN.summary.json` hoặc dòng cuối `wsp_log.jsonl`.
d) Giữ bản sửa khi điểm tập đo tăng và thời gian không tăng; ngược lại quay về bản trước.
e) Ngân sách theo blog: 10, 25, 50 vòng; chỉ 50 vòng mới có hiệu quả rõ. Thời gian mỗi vòng: CHƯA XÁC MINH (đo ở bước 1).

## 5. Go/no-go (2026-11-13)

a) Giữ WSP khi điểm 300 mẫu dev tăng rõ so với baseline cùng chế độ thinking, và thời gian mỗi mẫu không tăng.
b) Nếu lỗi chính ở baseline không phải thinking bị cắt hay lặp: blog cho thấy WSP khó giúp, nên dừng sớm.
c) Nếu tắt thinking đã gỡ phần lớn lỗi: so WSP với baseline `--think off`, không so với `--think on`.

## 6. Đã kiểm trên máy cá nhân (2026-11-07, không GPU)

a) Kernel trên tensor thật: Qwen3-4B có 4.022.468.096 tham số; `project_out` 64 cột tầng 35 đổi 158.541 phần tử (0,0039%), thành phần theo hướng giảm từ 0,073 xuống 0,00003; cột khác giữ nguyên.
b) Lưu và đọc lại: chỉ shard có tensor sửa được ghi mới, tensor khác giữ nguyên, model gốc không đổi.
c) `run_round.py --llm-gold` (pipeline replay nhãn): vòng 0 và vòng 1 có sửa chạy hết luồng, mean 1,0.
d) `analyze.py` bắt đúng 4 lỗi giả (thinking bị cắt và lặp, sai định dạng, lỗi pipeline, call thừa).
e) `directions.py --prompt-only` dựng prompt 3.135 đến 8.005 token bằng tokenizer thật, đúng chat template cả hai chế độ.
f) CHƯA XÁC MINH: chạy vLLM với `--llm-dtype half` trên T4, forward của `directions.py` trên GPU.
