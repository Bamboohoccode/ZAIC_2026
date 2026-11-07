# GEPA của Kiệt (`optimize/gepa/kiet/`)

Kế hoạch 07/11: Kiệt làm **Mapper trước** (Vương làm Planner), vì GPU Kaggle chỉ có 30 giờ mỗi tuần.

**Mapper** (bắt lỗi, sửa lỗi ASR và phương ngữ):
- Prompt chạy trên chính Qwen3-4B, không thêm model. Seed: `prompts/gepa/kiet/mapper_v0.txt`.
- Input: n-best ASR (`ASROut` của `lifemate/contracts.py`) cùng từ điển thực thể lấy từ catalog (app đã cài kèm tên tiếng Việt, giá trị enum của tool).
- Tối ưu bằng metric nhanh `evaluate_mapper`: 0,7 × tỉ lệ giữ đúng thực thể + 0,3 × (1 − WER) so với `audio_text`, không chạy Planner.
- Chốt bằng điểm cuối với Planner cố định (`planner_v4` của Vương, champion hiện tại: mean F1 0.3597).
- Ablation 3 cấu hình: không Mapper (`--asr`), Mapper cũ Qwen3-0.6B (`--mapper old`), Mapper mới (`--mapper mapper_vN.txt`).

**Planner:** runner vẫn có (`--component planner`), nhưng hiện Vương phụ trách.

**Ra:**
- `prompts/gepa/kiet/mapper_vN.txt` và `manifest.json` (điểm, độ trễ).
- `runs/gepa_kiet/eval_*.jsonl` có `raw` (ASR) và `texts` (sau Mapper) từng lượt, để Khánh test entity guard.

Trên Kaggle: mở `notebooks/run_gepa_kiet.ipynb`.

```
pip install gepa pyyaml vllm chunkformer
python -m optimize.gepa.kiet.run_gepa asr  --root data                         # một lần
python -m optimize.gepa.kiet.run_gepa run  --root data --component mapper      # metric proxy, rẻ
python -m optimize.gepa.kiet.run_gepa eval --root data --asr --limit 100       # không Mapper
python -m optimize.gepa.kiet.run_gepa eval --root data --mapper old --limit 100
python -m optimize.gepa.kiet.run_gepa eval --root data --mapper prompts/gepa/kiet/mapper_v1.txt --limit 100
```

`--limit N` ghi vào manifest dưới khoá `...#N`, không đè bản chấm đủ. Train/val chia theo app và loại hẳn `dev_ids.txt`.
Điểm hiện tại: chưa chạy trên GPU.
