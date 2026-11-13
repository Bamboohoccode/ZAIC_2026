# Nhật ký phiên làm việc (memory)

Tệp này gom nhật ký các phiên làm việc để chuyển sang phiên mới mà không mất ngữ cảnh.
Nguồn: `docs/PROGRESS.md`, `docs/DECISIONS.md`, `own_flow/plan_data_tool.md`, và phiên chat ngày 11/11/2026 (mục 4).
Lưu ý: tôi không có bản ghi đầy đủ của các phiên trước. Phần 2 chỉ tổng hợp những gì đã được ghi trong tài liệu dự án. Quy tắc làm việc nằm ở `CLAUDE.md`, không lặp lại ở đây.

---

## 1. Trạng thái dự án (tính đến 11/11/2026)

a) Dữ liệu BTC: `data/raw/train+public_test/` (release v3.2.0). Train 1545 mẫu có nhãn, public_test 634 mẫu chỉ có input.
b) Catalog: train 46 app, 147 tool (62 read, 84 write, 1 meta). public_test 36 app, 122 tool (52 read, 69 write, 1 meta).
c) Việc đã xong: khảo sát tool, planner và verifier train/dev, nhãn ask v2, chuỗi chấm điểm của nhóm, mã WSP, gói Kaggle, bộ data nhiều lượt có tool result.
d) Việc còn: chạy WSP trên Kaggle (chưa có số đo GPU), C7 go/no-go ngày 13/11, B5, B7, B8, chốt cách fine-tune.

---

## 2. Nhật ký các phiên trước (tổng hợp từ PROGRESS và DECISIONS)

### 07/11/2026
a) Giải nén dữ liệu BTC, kiểm ToolEnv (147/147 và 122/122 tool trả `ok`), kiểm chuỗi chấm điểm (gold dev = 1,0000).
b) Khảo sát tool (`TOOL_SURVEY.md`), planner và verifier train/dev, nhãn ask v2 (65 mẫu).
c) Tải Qwen3-4B. Quyết định: chỉ làm WSP, bỏ LoRA. Không lượng tử hoá.
d) Viết mã WSP (`optimize/wsp/`), đóng gói Kaggle (`kaggle_upload/`), kiểm tràn fp16 trên trọng số (0 tràn).
e) Dùng Kaggle để chạy model vì GPU local 6 GB. Không gọi API Claude từ Kaggle.

### 08/11/2026
a) Người dùng duyệt DECISIONS k và l (vòng tìm trên practice 100 mẫu, luật giữ bản sửa).
b) Đổi `LLM_MODEL` trong `lifemate/pipeline.py` sang `Qwen/Qwen3-4B`.
c) Lần chạy QUICK trên Kaggle lỗi ô 12 vì không đủ mẫu bad/good. Đã xoá zip trung gian theo đồng ý của người dùng.

### 09/11/2026 đến 10/11/2026
a) Chuyển sang model `Ma7ee7/Qwen3.8_4B_Distilled` (revision `f33da883`), quyết định của người dùng (DECISIONS 09/11).
b) Đóng gói `zaic_code_20261109_v4.zip`, notebook A/B.
c) Kết quả đợt đầu (`reports/wsp_run1_20261109.md`): dev think on và off đều mean 0,4694 (n=10). Practice mean 0,1792 (n=20), 14 mẫu HASH_MISMATCH.

### 11/11/2026 (đợt Kaggle và gộp notebook)
a) Gộp A/B thành một notebook `kaggle_upload/wsp_single_20261111.ipynb` (12 ô), theo yêu cầu người dùng.
b) Sửa OOM `directions.py` (forward theo đoạn có KV cache), gói `zaic_code_20261111_v6.zip` (55 tệp).
c) Chạy notebook nhanh v6 trên Kaggle: có `best_spec` (tầng 0, 64 cột, alpha 1,0), practice 10 mẫu 0,3583 -> 0,5250. Chưa phải bằng chứng cải thiện vì đo và chọn trên cùng 10 mẫu.

---

## 3. Quy ước làm việc người dùng đã nêu

a) Trả lời tiếng Việt, ngắn, đi thẳng vào kết quả. Gạch đầu dòng dạng a) b) c). Không dùng em dash. Dùng hệ mét.
b) Điều chưa kiểm ghi "CHƯA XÁC MINH". Số liệu phải truy được về file hoặc đầu ra script.
c) Thống kê và trình bày dữ liệu: người dùng muốn biểu đồ (HTML tự chứa), không chỉ bảng md.
d) Khi người dùng hỏi, trả lời câu hỏi trước, chưa sửa file. Chỉ sửa sau khi người dùng nói rõ.
e) Dữ liệu BTC không đăng công khai, không gửi ra dịch vụ ngoài.
f) Khi người dùng chỉ gửi ảnh mà không có lời nhắn, hỏi lại họ muốn gì thay vì đoán.

---

## 4. Phiên chat 11/11/2026 (phiên hiện tại)

### 4.1 Việc đã làm
a) Đọc `CLAUDE.md`, `check_data.md`, `TOOL_SURVEY.md`, `PROGRESS.md`, `DECISIONS.md`, `plan_data_tool.md`, `data/raw/.../README.md`, và `zaic_tools` (ToolEnv).
b) Quyết định với người dùng: bộ data là hội thoại nhiều lượt có tool result, chỉ lấy từ train BTC, không sinh thêm.
c) Viết `scripts/build_sft_multiturn.py`. Kết quả:
   - `data/processed/sft_multiturn_train_20261111.jsonl`: 1244 mẫu (act 980, ask 175, empty 89).
   - `data/processed/sft_multiturn_dev_20261111.jsonl`: 300 mẫu (act 236, ask 42, empty 22).
   - Loại `train_0066` vì trùng `audio_text` trong train. Không có mẫu train nào trùng dev.
   - Replay ToolEnv trên 1544 mẫu: 0 lệch `tool_results`, 0 lệch `final_state`, 0 call bị từ chối.
   - Độ dài token: trung vị 4891, max 9384, 23 mẫu > 8192.
   - Manifest: `data/manifest/sft_multiturn_sha256_20261111.txt`. Báo cáo: `reports/sft_multiturn_20261111.md`.
d) Cập nhật `docs/PROGRESS.md` (A4 xong, B6 xong), `docs/DECISIONS.md` (mục n), `docs/data_dictionary.md` (mục 4).
e) Viết `data/processed/sft_multiturn_FIELDS.md` (mô tả trường, ví dụ thật từ `train_0454`). Đã kiểm 5 quan hệ nêu trong file: 0 lệch.
f) Viết `scripts/make_stats_html.py`, sinh `data/derived/stats_btc_20261111.html` và `data/derived/stats_sft_multiturn_20261111.html`. Đã sửa: giải thích read/write/meta, thêm danh mục app với mô tả tool, so sánh train và public_test theo app và tool.

### 4.2 Số liệu đã xác nhận (chạy thật, 11/11/2026)
a) Tool train 147 = 62 read + 84 write + 1 meta. Tool public_test 122 = 52 + 69 + 1.
b) App trùng 25 (cả 25 có cùng tập tool). Chỉ train 21 app, chỉ public_test 11 app.
c) Tool trùng 88. Chỉ train 59. Chỉ public_test 34.
d) Dev có 9 app lạ theo app được gọi trong nhãn: `bike_rental`, `calculator`, `exchange_rate`, `fitness`, `fuel_price`, `gold_price`, `lottery`, `package_tracking`, `tv`. Khớp `eval/splits/README.md`.
e) Ghi chú sai của tôi trong phiên này: ban đầu tôi nói dev không có app lạ, vì tôi chỉ so theo `installed_apps`. Đã sửa lại theo app được gọi.

### 4.3 Việc còn dở
a) Chưa cập nhật `docs/PROGRESS.md` cho việc thống kê HTML. Người dùng chưa đồng ý.
b) Chưa xem bản giao diện sáng của hai trang HTML (ảnh chụp là giao diện tối).
c) Người dùng gửi ảnh thứ ba nhưng không có lời nhắn. Chưa rõ họ muốn xem phần nào hay muốn sửa gì. Đã hỏi lại, chưa có trả lời.
d) Chưa chốt cách fine-tune. Người dùng nói "cách finetune không quan trọng" khi xây data. Hướng sau này là LoRA hoặc WSP tuỳ quyết định.
e) Chưa có câu trả lời bằng lời sau tool result cuối trong nhãn BTC. Nếu cần, phải quyết định thêm.

### 4.4 Tệp tạo hoặc sửa trong phiên này
- `scripts/build_sft_multiturn.py` (mới)
- `scripts/make_stats_html.py` (mới, đã sửa nhiều lần)
- `data/processed/sft_multiturn_train_20261111.jsonl`, `sft_multiturn_dev_20261111.jsonl`, `sft_multiturn_FIELDS.md`
- `data/derived/stats_btc_20261111.html`, `stats_sft_multiturn_20261111.html`
- `reports/sft_multiturn_20261111.md`
- `data/manifest/sft_multiturn_sha256_20261111.txt`
- `logs/build_sft_multiturn_20261111.log`
- `docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/data_dictionary.md` (sửa thêm)
- `docs/memory.md` (tệp này)

### 4.5 Cách tiếp tục ở phiên mới
a) Đọc `docs/memory.md` mục 1, mục 3 và mục 4.3.
b) Hỏi người dùng ảnh thứ ba cần gì, và có muốn cập nhật `PROGRESS.md` cho thống kê HTML không.
c) Chạy lại `scripts/make_stats_html.py` nếu cần sinh lại trang. Chạy lại `scripts/build_sft_multiturn.py` chỉ khi người dùng yêu cầu.
