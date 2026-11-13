# Luật đã đối chiếu

Ghi luật đã xác nhận bằng văn bản chính thức và mục "chưa xác nhận" (xem `PROJECT_OVERVIEW.md` mục 9).

## 1. Đã xác nhận từ README trong gói dữ liệu BTC (v3.2.0, đọc 2026-11-07)

a) Mỗi mẫu dùng một `ToolEnv` riêng, chỉ tạo một lần. Mọi call đã gọi, kể cả call bị từ chối, đều tính vào bài nộp.
b) Call sai schema (tool không có hoặc chưa cài, thiếu tham số, sai kiểu) bị từ chối và không đổi state.
c) Agent không đọc trực tiếp `state.json`, chỉ xem dữ liệu qua tool.
d) Mỗi mẫu chỉ được gọi tool trong `context.installed_apps`.
e) File nộp: UTF-8 JSONL, một dòng mỗi mẫu, gồm `id`, `tool_calls`, `state_hash`.
f) Bài nộp được chấm bằng chính thư viện `zaic_tools`, nên kết quả chạy local giống khi chấm.
g) Dữ liệu chỉ dành cho đội đã đăng ký, không chia sẻ hay phát tán.
h) Private test phát hành sau khi nộp Docker.

## 2. Đã xác nhận từ mã nguồn `zaic_tools` (đọc 2026-11-07)

a) Lý do duy nhất để từ chối call là lỗi schema: tool không có trong catalog hoặc chưa cài, thiếu `required` (None, chuỗi rỗng, list rỗng, dict rỗng đều tính là thiếu), sai `type`, `enum`, `minimum`/`maximum`, `format`, `items`, hoặc thiếu nhóm `x-one-of`.
b) ID không tồn tại, tìm kiếm rỗng, không đủ số dư không phải lỗi: tool chạy, không đổi gì, trả `{"ok": true, "done": false, ...}`.

## 3. CHƯA XÁC MINH (chưa đọc thể lệ PDF chính thức)

a) Trần tham số 6B toàn hệ thống và 5B riêng LLM; ASR có tính không.
b) Thời gian 30 giây mỗi mẫu trên A30.
c) Dùng model ngoài (Opus) để sửa trọng số và để sinh dữ liệu khi phát triển. Người dùng cho biết đã hỏi BTC (2026-11-07), chưa có văn bản trả lời trong repo.
d) Model đã sửa trọng số có còn được coi là "public trước 26/10/2026" không.
e) Số lần nộp Docker.
