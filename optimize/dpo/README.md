# DPO Tool Knowledge Dataset & Fine-Tuning (`optimize/dpo/`)

Bộ dữ liệu preference pairs `(chosen, rejected)` cho bài toán **Task 2: DPO — Cải thiện Tool Knowledge giúp Qwen 4B phân biệt tool đúng/sai, tham số bắt buộc/tuỳ chọn và cổng chốt chặn mơ hồ** mà **không làm phình prompt/schema** (giữ nguyên Champion V15 prompt trong `lifemate/pipeline.py`).

## 1. Cấu trúc tệp
- `optimize/dpo/dataset/dpo_train.jsonl`: Bộ cặp preference huấn luyện, sinh từ **tối đa 1,087 mẫu** thuộc tập `eval/splits/dpo_train_ids.txt` (0% giao cắt với `dev_ids.txt` 300 mẫu hay `test_ids.txt` 100 mẫu).
- `optimize/dpo/dataset/dpo_val.jsonl`: 5% mẫu giữ riêng nội bộ từ `dpo_train_ids.txt` để theo dõi loss / reward accuracy khi train DPO (không đụng `dev_ids.txt` hay `test_ids.txt`).
- `optimize/dpo/dataset/tool_specs_catalog.json`: Bảng schema chuẩn 147 tool (đồng nhất byte với `lifemate.pipeline.tool_specs()`), giúp tệp JSONL gọn nhẹ dưới ngưỡng cảnh báo GitHub.
- `optimize/dpo/dataset/dpo_stats.json`: Thống kê chi tiết phân bố theo nhãn, 5 nhóm negative và app.

## 2. 5 nhóm Hard Negatives trong bộ DPO
1. **`confusable_tool`**: Đánh tráo giữa các tool hay nhầm cùng nhóm hoặc liên app (`sms.send` vs `chat_ott.send_message`, `banking.transfer` vs `ewallet.transfer`, `alarm.create` vs `reminders.create` vs `calendar.create_event` vs `tasks.add`, `smart_home.light_set` vs `smart_home.switch_set`, `weather.get_forecast` vs `weather.get_current`...).
2. **`hallucinated_optional_args`**: Tự ý điền tham số tuỳ chọn người dùng không yêu cầu (`note`, `label`, `repeat`, `seat_class`, `location`, `brightness`, `date_from`...) hoặc làm biến dạng literal (`.test` $\rightarrow$ `.com`).
3. **`premature_write_or_ask`**: Nhảy cóc gọi tool ghi ở lượt đầu với ID tự bịa, gọi sai nguồn ID (`contacts.search` cho `banking`), hoặc gọi `sys.ask_missing_info` sớm trước khi tra cứu ID.
4. **`ambiguity_gate_error`**: Missed Ask (chọn đại ứng viên #1 hoặc tự bịa số tiền) và False Ask (hỏi lại dù đã có bộ lọc duy nhất).
5. **`multi_action_or_scope_error`**: Dừng sớm bỏ sót hành động thứ 2 / nhánh `ELSE`, hoặc False Act trên mẫu `empty` / app chưa cài (`no_app`).

## 3. Tách tập đánh giá (0% rò rỉ app lạ)
- **Val (`eval/splits/dev_ids.txt`)**: 300 mẫu (**9 app lạ**: `bike_rental`, `calculator`, `exchange_rate`, `fitness`, `lottery`, `package_tracking`, `fuel_price`, `gold_price`, `tv`).
- **Test (`eval/splits/test_ids.txt`)**: 100 mẫu (**3 app lạ**: `library`, `home_energy`, `medication`, cộng mẫu stratified đạt đúng tỉ lệ 79% act / 14% ask / 7% empty).
- **Train DPO (`eval/splits/dpo_train_ids.txt`)**: 1,145 mẫu (34 app còn lại).

## 4. Lệnh chạy cho đồng đội
1. Kiểm tra dữ liệu cục bộ (không cần GPU hay model):
   ```bash
   python -m lifemate.planner.train_dpo --dry-run
   ```
2. Chạy DPO fine-tuning (điền phiên bản Qwen vào `--model` hoặc biến môi trường `QWEN_MODEL_PATH`):
   ```bash
   python -m lifemate.planner.train_dpo --model <TEN_HOAC_DUONG_DAN_QWEN> --load-in-4bit
   ```
3. Chạy inference & chấm điểm theo từng app (trước vs sau):
   ```bash
   python -m lifemate.pipeline --root data --split train --ids eval/splits/dev_ids.txt \
       --llm <TEN_HOAC_DUONG_DAN_QWEN> --llm-lora checkpoints/planner_dpo_lora --mapper none --text --verify-rounds 0 --out runs/dev_dpo.jsonl
   python eval/scorer.py --sub runs/dev_baseline.jsonl --compare runs/dev_dpo.jsonl --dev-ids eval/splits/dev_ids.txt --by-app
   ```
