# Chia tập Train / Dev theo App (eval/splits/)

Tệp `dev_ids.txt` chứa danh sách 300 mẫu (19.4% tập train) được tách ra làm tập kiểm chứng cố định (held-out validation set) cho quá trình phát triển (A/B testing, GEPA prompt optimization).

## 1. Nguyên tắc thiết kế (Khái quát hoá sang App lạ)
1. **Giữ lại 20% app lạ (9 / 46 app)**:
   - Các app chỉ xuất hiện trong dev: `bike_rental`, `calculator`, `exchange_rate`, `fitness`, `lottery`, `package_tracking`, `fuel_price`, `gold_price`, `tv`.
   - 100% mẫu gọi các app này được đưa vào `dev_ids.txt` (tổng 228 mẫu).
   - Mô phỏng thực tế bảng `public_test` (có 11 app hoàn toàn mới so với train) để đánh giá khả năng zero-shot tool selection & parameter reasoning của Agent dựa vào schema trong `catalog.json`.
2. **Cân bằng nhãn mục tiêu (Stratified ratio)**:
   - Bổ sung 72 mẫu từ các app còn lại (và các mẫu không gọi app) để tỉ lệ tương đương toàn bộ tập train:
     - `act`: 236 mẫu (78.67% vs train 78.77%)
     - `ask` (`sys.ask_missing_info`): 42 mẫu (14.00% vs train 14.05%)
     - `empty` (`[]` out-of-scope / hủy): 22 mẫu (7.33% vs train 7.18%)
3. **Tính tái lập (Reproducibility)**:
   - Được sinh tự động bằng script `make_dev_split.py` với `seed=42`.
   - Chạy lại: `python eval/splits/make_dev_split.py`.
