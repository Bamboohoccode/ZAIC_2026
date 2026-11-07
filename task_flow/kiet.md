# Task flow: Kiệt (nhánh gepa/kiet)

## Mục tiêu
Cùng Vương cải tiến Planner và Mapper bằng GEPA: mỗi người một bản prompt riêng, chạy độc lập trên cùng metric và cùng dev set, rồi ensemble. Phần nền của Kiệt: eval harness, metric GEPA, dữ liệu synthetic.

## Luồng dự định
Kế hoạch 07/11 (anh Vương): Kiệt làm Mapper trước, Vương làm Planner, vì GPU Kaggle 30 giờ/tuần không đủ để chạy song song. Thứ tự ưu tiên: 1, 2, 3, 7, rồi 4, 6, 5, 8.

1. Rà baseline của Vương, tạo nhánh gepa/kiet từ planner/vuong [ ] ClickUp subtask: Rà baseline, tạo nhánh
2. eval/: scorer dùng chung (score_sample), điểm theo app, thống kê từng tầng, pass^k [ ] ClickUp subtask: Eval harness (hạn 07/11)
3. optimize/gepa/metric/: metric chấm đúng công thức BTC kèm feedback text [ ] ClickUp subtask: Metric GEPA
4. optimize/gepa/kiet/: runner GEPA cho prompt Planner và prompt Mapper, train/val chia theo app, loại dev [ ] ClickUp subtask: Runner GEPA
5. Chạy GEPA trên Kaggle T4, chấm dev với cùng cấu hình seed v0, ghi docs/DECISIONS.md [ ] ClickUp subtask: Chạy GEPA Planner (hạn 14/11)
6. synth/: bộ test 3 nhãn (đủ, thiếu slot, không có app) từ catalog [ ] ClickUp subtask: Synth 3 nhãn
7. ASR 1 lần + GEPA cho prompt Mapper (Qwen3-4B, metric nhanh không chạy Planner), ablation điểm cuối với planner_v4 (champion đã đo): không Mapper / Mapper cũ Qwen3-0.6B / Mapper mới; đổi sang n-best của Khánh khi có [ ] ClickUp subtask: GEPA Mapper (hạn ablation 12/11)
8. Ensemble với Vương tại optimize/gepa/ensemble/, một PR chung [ ] ClickUp subtask: Ensemble (13 đến 14/11)

## Note tiến độ (mới nhất ở trên)
- 07/11: đổi kế hoạch, làm Mapper trước. Thêm metric nhanh cho Mapper (giữ thực thể + WER, không chạy Planner), từ điển thực thể từ catalog, eval --mapper old (Qwen3-0.6B), file eval lưu raw và texts cho entity guard; Planner cố định cập nhật thành planner_v4 (0.3597; v5 chưa có điểm). 9 unit test qua; gold replay 100 mẫu đạt mean 1.0. GEPA Mapper chưa chạy trên GPU.
- 06/11 (khuya): thêm Mapper vào runner: mode asr (cache dạng ASROut), GEPA component mapper, eval 3 cấu hình ablation; seed prompts/gepa/kiet/mapper_v0.txt. Test giả lập pass, chưa chạy GPU.
- 06/11 (tối): viết xong bước 6 (synth 3 nhãn theo schema thật) và notebooks/run_gepa_kiet.ipynb (một-click Kaggle, cùng mẫu notebook của Vương). Chưa chạy trên GPU.
- 06/11: viết xong bước 2, 3, 4 (scorer refactor giữ nguyên kết quả, run_eval, metric, run_gepa) và test bằng dữ liệu giả lập; chưa chạy trên GPU và dữ liệu thật.

## Quyết định và điểm đo
- Reflection LM mặc định là Qwen3-4B local, có thể đổi sang API trong config (chưa được BTC xác nhận, PROJECT_OVERVIEW mục 9b).
- Cùng cấu hình chuẩn với Vương: --text, mapper none, verify_rounds=0, max_ctx=8192, không cắt giờ. Mốc so sánh: seed v0 = 0.3109 (Act 0.3304, Ask 0.0786, Empty 0.5455), ablation của Vương 06/11, nên không chạy lại.
- Khác hướng với Vương để hai bản độc lập: Vương làm GEPA thủ công (phản tư trong chat), Kiệt chạy GEPA tự động với train/val chia theo app và giữ dev ngoài vòng tối ưu.
- Điểm GEPA có hash_credit=0.2 cho mẫu lệch hash (46% baseline lệch hash nên điểm 0/1 quá thưa); báo cáo luôn dùng điểm BTC.
