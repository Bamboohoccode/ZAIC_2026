# Module Schema Validator & Safe Hash Guard (Vương)

Module kiểm tra tính hợp lệ của tool calls và chốt chặn an toàn (Safe Guard) bảo vệ `state_hash` trước khi lệnh ghi được thực thi vào `ToolEnv`.

## 1. Trách nhiệm & Chức năng
- **Schema Validation**: Đối chiếu tham số của Planner với JSON Schema từ catalog `zaic_tools`.
- **Ambiguity Guard (Chống chọn bừa khi trùng tên/nhiều đối tượng)**: Khi tool đọc trả về $\ge 2$ kết quả (ví dụ tìm ra 2 người cùng tên "Bình", nhiều xe đạp), nếu Planner cố gọi tool ghi thay vì hỏi, Validator sẽ tự động can thiệp, chuyển đổi thành `sys.ask_missing_info`.
- **Vague Slot Guard**: Phát hiện các yêu cầu mập mờ thiếu số liệu cụ thể (như "chút tiền", "ít tiền", "hạ target", "kênh nớ") và chặn không cho model tự bịa giá trị làm lệch state hash.
- **Parameter Hygiene**: Tự động lược bỏ các tham số mặc định thừa (như `repeat: "once"`, `note` không được yêu cầu) để khớp 100% với nhãn gold của BTC.

## 2. Đầu vào / Đầu ra
- **Đầu vào**:
  - `calls`: Danh sách `tool_calls` do Planner đề xuất (`[{"name": "...", "arguments": {...}}]`).
  - `catalog`: Từ điển schema của toàn bộ tool từ `zaic_tools`.
  - `msgs`: Lịch sử hội thoại và kết quả trả về của các tool đọc trước đó.
  - `raw_query`: Câu nói gốc của người dùng.
- **Đầu ra**:
  - `guarded_calls`: Danh sách tool calls đã được kiểm duyệt, khử mơ hồ và làm sạch tham số an toàn.

## 3. Cách chạy Unit Test
```bash
python tests/test_validator.py
```
