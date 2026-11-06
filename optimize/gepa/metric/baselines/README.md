# GEPA Metric Baselines

Thư mục lưu trữ dự đoán baseline trên tập `dev_ids.txt` (300 mẫu) dùng chung cho Kiệt và Vương để chạy GEPA, tính metric và trích xuất failure cases cho reflection mà không cần chạy lại GPU.

## 1. File `dev_baseline.jsonl`

- **Số mẫu:** 300 mẫu (khớp với `eval/splits/dev_ids.txt`, 9 app hold-out).
- **Model:** `Qwen/Qwen3-4B-Instruct-2507` (chạy qua vLLM trên Kaggle GPU Tesla T4).
- **Prompt:** `prompts/gepa/vuong/planner_v0.txt` (prompt gốc trong `lifemate/pipeline.py`).
- **Cấu hình pipeline:**
  ```bash
  python -m lifemate.pipeline \
      --root data \
      --split train \
      --text \
      --llm Qwen/Qwen3-4B-Instruct-2507 \
      --mapper none \
      --gpu-mem 0.90 \
      --max-ctx 8192 \
      --verify-rounds 2 \
      --ids eval/splits/dev_ids.txt \
      --out runs/dev_baseline.jsonl
  ```

## 2. Kết quả chấm điểm (`eval/scorer.py`)

- **Mean Score:** `0.3117`
- **Act F1 (n=236):** `0.3268`
- **Ask F1 (n=42):** `0.0810`
- **Empty F1 (n=22):** `0.5909`

### Phân bố lỗi chính (Dùng cho GEPA Reflection):
- `HASH_MISMATCH`: 139 mẫu (46.3%) — Lệch state hash (do thiếu call write phụ thuộc hoặc sai thứ tự).
- `WRONG_TOOL_OR_ARGS`: 52 mẫu (17.3%) — Sai tên tool hoặc sai tham số.
- `MISSED_ASK`: 16 mẫu (5.3%) — Đáng lẽ phải gọi `sys.ask_missing_info` nhưng lại cố gọi tool thường.
- `MISSING_CALL`: 12 mẫu (4.0%) — Bỏ sót tool call.
- `FALSE_ACT`: 9 mẫu (3.0%) — Người dùng tự huỷ hoặc không yêu cầu nhưng vẫn gọi tool.
- `EXTRA_CALL`: 4 mẫu (1.3%) — Gọi thừa tool.

## 3. Cách dùng trong GEPA
Kiệt và Vương khi viết script GEPA (`optimize/gepa/`) có thể load trực tiếp `dev_baseline.jsonl` để:
1. Đọc điểm mốc chuẩn `0.3117` nhằm đo delta cải thiện ($\Delta F1$).
2. Trích xuất danh sách các `sample_id` thất bại để đưa vào prompt phản tư (reflection prompt) cho LLM đề xuất sửa đổi prompt mới.
