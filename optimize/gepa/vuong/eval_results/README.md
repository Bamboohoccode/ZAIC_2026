# Evaluation Results & Traces (Vương)

Lưu trữ các kết quả dự đoán và trace log của các vòng thử nghiệm trên tập `dev_ids.txt` (300 mẫu).

## 1. `dev_no_verify.jsonl` (Ablation verify_rounds=0)
- **Prompt:** `prompts/gepa/vuong/planner_v0.txt`
- **Cấu hình:** `--verify-rounds 0`
- **Mean F1:** `0.3109` (Act: 0.3304, Ask: 0.0786, Empty: 0.5455)
- **Độ trễ trung bình:** 24.34s / mẫu (giảm 6.6s so với baseline có self-verify)

## 2. `dev_v1.jsonl` (GEPA Iteration 1)
- **Prompt:** `prompts/gepa/vuong/planner_v1.txt`
- **Cấu hình:** `--verify-rounds 0`
- **Mean F1:** `0.3523` (Act: 0.3659, Ask: 0.1989, Empty: 0.5000)
- **Độ trễ trung bình:** 27.05s / mẫu
- **Điểm nhấn:** Ask F1 tăng từ 0.0786 lên 0.1989 (+145%), Mean F1 tăng +0.0406.

## 3. `dev_v2.jsonl` (GEPA Iteration 2)
- **Prompt:** `prompts/gepa/vuong/planner_v2.txt`
- **Cấu hình:** `--verify-rounds 0`
- **Mean F1:** `0.3436` (Act: 0.3665, Ask: 0.0854, Empty: 0.5909)
- **Độ trễ trung bình:** 29.62s / mẫu
- **Điểm nhấn:** Lỗi `WRONG_TOOL_OR_ARGS` giảm mạnh từ 53 xuống 41 (-22.6%).

## 4. `dev_v3.jsonl` (GEPA Iteration 3)
- **Prompt:** `prompts/gepa/vuong/planner_v3.txt`
- **Cấu hình:** `--verify-rounds 0`
- **Mean F1:** `0.3294` (Act: 0.3507, Ask: 0.0492, Empty: 0.6364)
- **Độ trễ trung bình:** 27.04s / mẫu
- **Điểm nhấn:** Empty F1 đạt kỷ lục `0.6364`.

## 5. `dev_v4.jsonl` (GEPA Iteration 4 - Champion Prompt)
- **Prompt:** `prompts/gepa/vuong/planner_v4.txt`
- **Cấu hình:** `--verify-rounds 0`
- **Mean F1:** `0.3597` (Act: 0.3831, Ask: 0.1314, Empty: 0.5455)
- **Độ trễ trung bình:** 28.01s / mẫu
- **Điểm nhấn:** Mean F1 đạt đỉnh 0.3597, Act F1 kỷ lục 0.3831. Tăng điểm net +22 mẫu vs baseline.

## 6. `dev_v5.jsonl` (GEPA Iteration 5)
- **Prompt:** `prompts/gepa/vuong/planner_v5.txt`
- **Cấu hình:** `--verify-rounds 0`
- **Mean F1:** `0.3454` (Act: 0.3592, Ask: 0.1632, Empty: 0.5455)
- **Độ trễ trung bình:** 26.80s / mẫu
- **Điểm nhấn:** Ask F1 hồi phục lên 0.1632.

## 7. `dev_v4_guarded.jsonl` (Full Cloud GPU Run: Planner V4 + Safe Hash Guard)
- **File:** `dev_v4_guarded.jsonl`, `dev_v4_guarded.jsonl.trace.jsonl`, `dev_v4_guarded.jsonl.log`
- **Prompt:** `prompts/gepa/vuong/planner_v4.txt`
- **Guard runtime:** `lifemate/validator/validator.py` (Safe Hash Guard & Schema Validator)
- **Môi trường:** Google Colab Cloud GPU (Tesla T4, vLLM, `--gpu-mem 0.90`, `--max-ctx 8192`)
- **Số mẫu:** Đủ 300/300 mẫu dev (`eval/splits/dev_ids.txt`)
- **Điểm số đánh giá chính thức (`eval/scorer.py`):**
  - **Mean Score:** `0.2961`
  - **Act F1:** `0.2687` (n=236)
  - **Ask F1:** `0.3200` (n=42) — **KỶ LỤC MỚI: Tăng +295% so với baseline 0.0810!**
  - **Empty F1:** `0.5455` (n=22)
- **Độ trễ trung bình:** **17.1s / mẫu** (Tốc độ suy luận kỷ lục, giảm mạnh từ 28.01s xuống 17.1s!)
- **Số mẫu > 30s:** Chỉ còn 36/300 mẫu (12.0%, giảm từ 130 mẫu của baseline và 89 mẫu của V4 thuần).
- **Phân loại lỗi:**
  - `HASH_MISMATCH`: 147 (49.0%)
  - `WRONG_TOOL_OR_ARGS`: 46 (15.3%)
  - `MISSED_ASK`: 13 (4.3%)
  - `MISSING_CALL`: 12 (4.0%)
  - `FALSE_ACT`: 10 (3.3%)
  - `EXTRA_CALL`: 6 (2.0%)
  - `FALSE_ASK`: 1 (0.3%)
