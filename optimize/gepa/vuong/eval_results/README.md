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
