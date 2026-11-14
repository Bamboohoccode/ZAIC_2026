# Chia tập Train / Val (Dev) / Test theo App (eval/splits/)

Tài liệu phân chia cố định 1,545 mẫu từ `data/train/train.jsonl` thành 3 tập không giao nhau (0% rò rỉ mẫu hay app lạ):

| Tập | File | Số mẫu | App giữ riêng hoàn toàn | Phân bố `act` / `ask` / `empty` | Mục đích |
|---|---|---|---|---|---|
| **Val (Dev)** | `dev_ids.txt` | **300** (19.4%) | **9 app**: `bike_rental`, `calculator`, `exchange_rate`, `fitness`, `lottery`, `package_tracking`, `fuel_price`, `gold_price`, `tv` (228 mẫu) + 72 mẫu stratified | 236 / 42 / 22 (78.7% / 14.0% / 7.3%) | Kiểm chứng cố định trong quá trình GEPA & chọn checkpoint DPO |
| **Test** | `test_ids.txt` | **100** (6.5%) | **3 app**: `library` (24), `home_energy` (27), `medication` (27) (tổng 78 mẫu) + 22 mẫu stratified | 79 / 14 / 7 (79.0% / 14.0% / 7.0%) | Đo tổng quát hóa sang app lạ sau khi chốt model DPO (mô phỏng public/private test) |
| **DPO Train** | `dpo_train_ids.txt` | **1,145** (74.1%) | 34 app còn lại (0 mẫu chứa 9 app Dev hay 3 app Test) | 902 / 161 / 82 (78.8% / 14.1% / 7.2%) | Nguồn sinh dữ liệu cặp preference `(chosen, rejected)` cho DPO |

## Tính tái lập (Reproducibility)
- Sinh lại tập Val (`dev_ids.txt`, seed=42):
  ```bash
  python eval/splits/make_dev_split.py
  ```
- Sinh lại tập Test 100 mẫu (`test_ids.txt`) & DPO Train (`dpo_train_ids.txt`, seed=42):
  ```bash
  python eval/splits/make_test_split.py
  ```
