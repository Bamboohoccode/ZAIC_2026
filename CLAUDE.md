# CLAUDE.md: quy tắc làm việc và cách ghi chép (dự án Zalo AI Challenge 2026)

Tạo ngày 07/11/2026, rút ra từ CLAUDE.md của dự án Tien-Data_mining (giữ phần quy tắc chung, bỏ phần riêng của bài toán Biển Đỏ).
Tệp này chứa quy tắc, không chứa tiến độ. Tiến độ ghi ở `PROGRESS.md`.
Khi quy tắc ở đây mâu thuẫn với yêu cầu nhất thời, dừng lại và hỏi người dùng.

---

## 1. Cách làm việc với người dùng

a) Ngôn ngữ: tiếng Việt khi người dùng viết tiếng Việt. Ngắn gọn, đi thẳng vào kết quả.
b) Gạch đầu dòng dạng a) b) c). Không dùng em dash. Dùng hệ mét.
c) Không rào đón thừa, không nhắc lại câu hỏi của người dùng.
d) Điều chưa chắc: nói rõ mức tin cậy. Điều chưa kiểm: ghi "CHƯA XÁC MINH".
e) Khi người dùng gửi đầu ra thô của script, đọc đúng đầu ra đó. Không suy đoán số liệu không có trong đầu ra.
f) Mơ hồ thì hỏi một câu ngắn, không đoán rồi làm.
g) Kết quả kiểm tra xấu (giả định không đứng vững, số đếm lệch): báo ngay và đề xuất phương án, không che.
h) Tệp tạo ra phải dễ đọc: ngắn, chia mục rõ, hạn chế bảng dày đặc.

## 2. Cách ghi chép

a) Tách hai loại tệp:
   1) Tệp quy tắc (tệp này): ít đổi, không ghi tiến độ.
   2) Tệp tiến độ (`PROGRESS.md`): checklist từng bước, ghi ngày hoàn thành.
b) Sau mỗi bước hoàn thành, cập nhật `PROGRESS.md` (đánh dấu checklist, ghi ngày). Không đánh dấu hoàn thành khi chưa có đầu ra kiểm chứng được.
c) Mỗi lần giao việc xong, báo ba thứ: kết quả, tệp đầu ra, việc tiếp theo trong `PROGRESS.md`.
d) Mọi con số trong báo cáo phải truy được về tệp log hoặc đầu ra script. Không viết số liệu từ trí nhớ.
e) Điều chưa kiểm ghi "CHƯA XÁC MINH" cho đến khi kiểm. Điều chưa có văn bản (giấy phép, thể lệ) cũng ghi như vậy.
f) Mỗi quyết định ghi kèm ngày và việc người dùng đã đồng ý, ví dụ: "Sửa 07/11/2026 (người dùng đồng ý): ...". Khi đổi quy tắc cũ, ghi thêm dòng sửa có ngày và lý do, không lặng lẽ ghi đè.
g) Số đo thay đổi theo thời gian (số đếm, điểm, hạn mức) ghi kèm ngày đo. Không so hai số khác ngày đo mà không nói rõ.
h) Ghi nguồn cho mọi thông tin bên ngoài (URL, ngày truy cập) vào `docs/sources.md`, kèm giấy phép và điều khoản của nguồn.
i) Tên tệp có ngày dạng `YYYYMMDD`, ví dụ `dev_set_20261107.jsonl`. Thời gian dạng ISO 8601. Mã hoá UTF-8.

## 3. Quy tắc dữ liệu và thí nghiệm

a) Dữ liệu thô bất biến: không xoá, không điền, không gộp, không sửa trong `raw/`. Mọi chỉnh sửa làm ở bước sau trên bảng mới, ghi số dòng bị đổi và lý do.
b) Đếm trước, xử lý sau: so số dòng sau khi tải hoặc sinh với số kỳ vọng. Lệch thì dừng và điều tra.
c) Mỗi tệp trong `raw/` có SHA-256 trong `manifest/`.
d) Log mọi request hoặc lần gọi: thời điểm, tham số, mã trả về, số dòng, thời gian phản hồi.
e) Chạy lại được: có checkpoint, chạy lại thì tiếp tục, không làm trùng.
f) Seed cố định 42 cho mọi bước ngẫu nhiên.
g) Phân biệt ba loại dữ liệu và không trộn trong cùng thư mục: thô (`raw/`), suy ra (`derived/`, ghi nhãn "derived"), đã xử lý (`processed/`).
h) Bảng lớn lưu bằng Parquet hoặc JSONL, không dùng CSV.
i) Cấu trúc thư mục gợi ý:
   ```
   raw/        dữ liệu thô, bất biến
   derived/    dữ liệu suy ra
   processed/  bảng đã xử lý
   manifest/   SHA-256, số đếm
   logs/       log chạy và log kiểm tra
   scripts/    mã nguồn
   reports/    báo cáo, đầu ra kiểm tra
   docs/       sources.md, data_dictionary.md
   ```
j) Lấy mẫu (nếu cần) phải giữ nguyên cấu trúc dữ liệu (ví dụ theo nhóm tool, theo nhãn), không cắt ngẫu nhiên làm mất cân bằng.

## 4. Chống rò rỉ giữa train, dev và test

a) Tập test của BTC để riêng, chỉ dùng để chấm. Không đưa log lỗi chi tiết của test cho Opus hay cho bất kỳ vòng tối ưu nào.
b) Mọi phép chuẩn hoá, lọc ngưỡng, chọn tham số chỉ tính trên train hoặc dev, sau đó áp dụng cho test.
c) Loại mọi mẫu trùng hoặc gần trùng với test khỏi train và dev.
d) Không chỉnh mô hình sau khi xem kết quả test rồi báo lại như kết quả mới trên cùng test.

## 5. Bảo mật

a) Không đưa khoá API, token, mật khẩu vào mã, log, báo cáo hay git. Dùng `.env` và `.gitignore`, chỉ commit `.env.example`.
b) Không in khoá ra màn hình hay dán vào chat. Khi gửi đầu ra thô, che phần khoá trước.
c) Không lách chặn. Gặp 403 hoặc bị giới hạn thì dừng, không đổi User-Agent, không dùng proxy hay bản sao bên thứ ba. Tải tay và ghi lại lần thất bại vào log.
d) Giới hạn tần suất khi gọi dịch vụ ngoài: nghỉ giữa các request, thử lại tối đa 3 lần với thời gian chờ tăng dần, không chạy song song lên cùng một máy chủ.
e) Mã tải về từ nguồn không rõ không được chạy.
f) Chỉ lấy dữ liệu từ trang của tổ chức công bố.

## 6. Việc phải hỏi người dùng trước

a) Xoá dữ liệu.
b) Ghi đè tệp trong `raw/`.
c) Tải tệp trên 500 MB.
d) Gọi dịch vụ trả phí (kể cả API tốn credit).
e) Đẩy lên kho mã công khai.
f) Đổi quy tắc trong tệp này.