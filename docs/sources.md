# Nguồn bên ngoài

Theo `CLAUDE.md` mục 2h: mỗi nguồn ghi URL, ngày truy cập, giấy phép và điều khoản.

## 1. Dữ liệu cuộc thi

a) Gói dữ liệu Zalo AI Challenge 2026, bảng LifeMate, release v3.2.0 (built 2026-10-24).
   1) Nhận qua cổng thi, lưu `data/train+public_test.zip`, SHA-256 `57d37151...47320` (đầy đủ trong `data/manifest/raw_sha256_20261107.txt`).
   2) Điều khoản (README trong gói): chỉ dành cho đội đã đăng ký, không chia sẻ, phát tán, phân phối. Vì vậy `data/` nằm trong `.gitignore`.
   3) Giấy phép chính thức: CHƯA XÁC MINH (chưa đọc thể lệ PDF).

## 2. Model

a) Qwen3-4B, https://huggingface.co/Qwen/Qwen3-4B, truy cập 2026-11-07.
   1) Giấy phép theo model card: Apache 2.0 (`license: apache-2.0` trong README đã tải).
   2) Bản tải về: snapshot `1cfa9a7208912126459214e8b04321603b3df60c` (đọc khi thử config, 2026-11-07).
   3) Ngày phát hành so với mốc "public trước 26/10/2026": CHƯA XÁC MINH bằng văn bản.
b) Qwen3-4B-Instruct-2507, https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507, truy cập 2026-11-07. Chỉ đọc config để kiểm truy cập, chưa tải. `pipeline.py` đã đổi sang Qwen3-4B ngày 08/11 (xem DECISIONS).
c) Qwen3-1.7B, https://huggingface.co/Qwen/Qwen3-1.7B, tải 2026-11-07 rồi xoá theo yêu cầu người dùng. Không còn dùng.

## 3. Phương pháp

a) Takehi, Teoh, Zhang, Khattab. Sparks of Weight-Space Programming, 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/, truy cập 2026-11-07.
   1) Số liệu đã đọc: Thí nghiệm 8 trên Qwen3.5-0.8B, sửa 131.072 trọng số (0,016%) ở MLP tầng 15 và 19; sau 50 forward pass MATH-500 từ 7,9% lên 48,1%. Với 10 hoặc 25 forward pass hiệu quả hạn chế.
   2) Giới hạn tác giả nêu: chỉ gỡ trở ngại (vòng lặp), không thêm năng lực; Gemma 3 270M gần như không cải thiện.
   3) Giấy phép nội dung blog: CHƯA XÁC MINH.
b) Log thí nghiệm: https://github.com/RikiyaT/lm_understand-logs. Chưa truy cập.

## 4. Tài liệu dẫn trong plan (chưa đọc lại trong phiên này)

a) When2Call (NAACL 2025), https://arxiv.org/abs/2504.18851.
b) Learning to Ask (EMNLP 2025), https://arxiv.org/abs/2409.00557.
c) τ-bench (ICLR 2025), https://arxiv.org/abs/2406.12045.
d) Huang et al. (ICLR 2024), https://arxiv.org/abs/2310.01798.
e) Panickssery et al. (NeurIPS 2024).
f) BFCL v3, https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html.

## 5. Phương ngữ (task_flow/tien_n1.md)

a) ViDia2Std, https://huggingface.co/datasets/Biu3010/ViDia2Std, bài báo https://arxiv.org/abs/2603.10211, truy cập 2026-11-13.
   1) Giấy phép theo trang dataset: CC BY-NC 4.0 (cấm dùng thương mại).
   2) 13.657 cặp câu (train 10.870, dev 1.184, test 1.603); trường `dialect`, `standard`, `region` (northern/central/southern); nguồn bình luận mạng xã hội.
   3) Quyết định 13/11/2026 (người dùng đồng ý): chỉ dùng để đo, không chép vào rules. Dùng để train cần BTC xác nhận: CHƯA XÁC MINH.
   4) Chưa tải.
b) which-dialect, https://github.com/nguyen18/which-dialect, truy cập 2026-11-13.
   1) Giấy phép: mã MIT; dữ liệu CC BY-SA 4.0 (Wiktionary qua Kaikki.org). Phải ghi nguồn Wiktionary khi dùng.
   2) Tiếng Việt: 1.330/42.419 nghĩa (3,1%) có nhãn miền, nhãn gắn theo nghĩa.
   3) Chưa tải.
