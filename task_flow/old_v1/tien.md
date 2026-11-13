# Plan: Qwen 4B + ToolEnv (Zalo AI Challenge 2026, LifeMate)

Ngày lập: 07/11/2026

Hai task dùng chung một nền là **ToolEnv của BTC**, nên cài ToolEnv trước.

- **Task 1:** dùng Claude Opus 5.5 sửa trọng số Qwen 4B (weight-space programming), đo bằng tập test của BTC.
- **Task 2:** khảo sát ToolEnv, sinh bộ dữ liệu (query → tool call), preprocess để finetune Qwen và huấn luyện Laya (LLM verifier) sau này.

> Chưa xác nhận: thể lệ 2026 (có cho dùng LLM ngoài sửa trọng số không, giới hạn gọi ToolEnv, tài nguyên) và giao diện ToolEnv. Đọc tại challenge.zalo.ai trước khi đầu tư nhiều.

---

## Thứ tự thực hiện

1. Cài ToolEnv, liệt kê tool, viết hàm chấm trạng thái cuối.
2. Chạy Qwen 4B gốc trên tập test BTC để có baseline.
3. Chạy song song Task 1 và Task 2.

---

## Phần A. Chuẩn bị chung

### A1. Cài ToolEnv
- Cài theo hướng dẫn của BTC.
- Chạy thử vài lệnh để chắc chắn môi trường hoạt động.

### A2. Tách dữ liệu ngay từ đầu
- **Train/dev:** dữ liệu tự sinh (Task 2), dùng cho vòng lặp của Opus và cho finetune.
- **Test:** tập của BTC, chỉ dùng để chấm.
- Loại mọi query trùng hoặc gần trùng với test khỏi train/dev.

### A3. Baseline
- Chạy Qwen 4B gốc trên test, mỗi mẫu chạy nhiều lần.
- Ghi điểm trung bình và độ ổn định (pass^k).

Nguồn: τ-bench chấm theo trạng thái cuối và dùng pass^k (xem Nguồn, mục 6).

---

## Phần B. Task 1: sửa trọng số Qwen 4B bằng Opus

### B1. Ý tưởng của method
Method: *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, 02/11/2026).
Link: https://rikiyat.github.io/blog/weight-space-programming/
Log thí nghiệm: https://github.com/RikiyaT/lm_understand-logs

Một LLM đa năng đọc và ghi trực tiếp trọng số mạng nơ-ron, không dùng gradient. Trong blog, Astra sửa Qwen3.5-0.8B: tìm ra hướng trong MLP gây lỗi lặp vô hạn, trừ hướng đó khỏi 0,016% trọng số. Kết quả MATH-500 tăng từ 7,9% lên 48,1% với 50 forward pass.

### B2. Vòng lặp đề xuất
1. Cho Opus một Python kernel có trọng số Qwen 4B (chỉ đọc/ghi tensor, chạy CPU).
2. Cho Opus một số lượng forward pass có giới hạn (blog thử 10, 25, 50). Mỗi forward pass: Qwen chạy các task trên **tập dev** và được chấm trong ToolEnv.
3. Opus đọc lỗi, đoán nguyên nhân trong trọng số, sửa, chạy lại.
4. Mỗi lần sửa xong, chạy **tập test BTC** để so với baseline. Không đưa log lỗi của test cho Opus.

### B3. Cần lưu ý
- Blog dùng GPT-6 Astra, không phải Opus. Chưa có bằng chứng Opus làm được tương tự.
- Blog làm trên 0,8B, chưa thử 4B.
- Method gỡ được lỗi (như loop), nhưng không thêm được năng lực. Với Gemma 3 270M không bị loop, Astra gần như không cải thiện (6,8% lên khoảng 7%).
- Blog dùng 10, 25, 50 forward pass: chỉ 50 lần mới cải thiện rõ. Cần dự trù đủ ngân sách chạy.
- Harness của blog chưa công bố, bạn phải tự dựng theo log.

### B4. Kiểm chứng
- So điểm test trước và sau, chạy nhiều lần mỗi mẫu.
- Kiểm tra Qwen có lỗi lặp hay lỗi định dạng rõ ràng không. Nếu không có, khả năng cải thiện thấp.
- Thử cả hai thứ tự: sửa trọng số rồi finetune, và finetune rồi sửa. Finetune có thể ghi đè bản sửa.

---

## Phần C. Task 2: khảo sát ToolEnv và tạo dữ liệu

### C1. Khảo sát tool
- Đọc schema hoặc danh sách tool mà ToolEnv cung cấp, thay vì gọi mù.
- Với mỗi tool, ghi lại: tên, tham số bắt buộc và tuỳ chọn, kiểu, giá trị hợp lệ, kết quả trả về, lỗi.
- Đánh dấu tool có tác dụng phụ không đảo ngược (xoá, gửi, thanh toán).
- Kiểm tra giới hạn số lần gọi.

### C2. Sinh query và gọi tool
- Dùng Opus sinh query tiếng Việt đa dạng: có phương ngữ, có lỗi ASR giả lập, có ngày giờ tương đối ("ngày mai").
- Cho Opus lập plan và gọi tool trong ToolEnv.
- Chỉ giữ trajectory thành công (trạng thái cuối khớp mục tiêu).

### C3. Ba loại nhãn
- `tool_call`: đủ thông tin, gọi tool.
- `ask_user`: thiếu thông tin, hỏi lại. Tạo bằng cách xoá một slot khỏi câu đủ thông tin.
- `cannot_answer`: không có tool phù hợp. Tạo bằng cách bỏ tool tương ứng.

Hai nhãn sau thường bị thiếu, nên cần chủ động sinh.

**Đặt tỉ lệ có chủ đích** (không để model tự sinh thì ra sao cũng được). Điểm xuất phát gợi ý, chỉnh lại sau khi xem điểm dev:

- Theo nhãn: khoảng 60% `tool_call`, 20% `ask_user`, 20% `cannot_answer`.
- Theo độ khó, trong mỗi nhãn:
  - Dễ: 1 tool, tham số rõ ràng.
  - Vừa: nhiều tham số, ngày giờ tương đối, phương ngữ hoặc lỗi ASR.
  - Khó: nhiều bước, hội thoại nhiều lượt, tool dễ nhầm lẫn.
- Không để mẫu dễ chiếm quá nửa.
- Sinh theo từng ô (nhãn x độ khó x tool), kiểm tra số lượng thực tế sau khi lọc trajectory thất bại, rồi bổ sung ô còn thiếu.

Nguồn: When2Call, Learning to Ask, BFCL v3 (mục 3, 4, 5 trong Nguồn).

### C4. Dữ liệu cho Laya (verifier)
- **ACCEPT:** plan đúng từ trajectory thành công.
- **REJECT:** làm hỏng plan đúng (bỏ arg, đổi giá trị, sai tool, thêm bước thừa, hành động không đảo ngược chưa xác nhận), kèm lý do ngắn và kiểm chứng được.
- Thêm lỗi thật của Qwen 4B trên dev set. Đây là phân phối lỗi gần thực tế nhất.
- Lỗi cấu trúc (thiếu arg, sai kiểu) để schema validator bằng code bắt. Laya chỉ kiểm ngữ nghĩa và an toàn.
- Giữ tỉ lệ ACCEPT/REJECT gần 50/50.
- Nên dùng model khác Planner làm verifier để tránh thiên lệch tự ưu ái.

Nguồn: Huang et al. và Panickssery et al. (mục 7, 8 trong Nguồn). Hai bài đo trên lập luận và tóm tắt, chưa đo trên kiểm tra plan tool, nên cần tự đo lại.

### C5. Preprocess
- Chuẩn hoá thành JSONL: `query`, `plan`, `nhãn`, `lý do` (cho Laya), `kết quả`.
- Loại bản trùng và gần trùng.
- Cân bằng theo tool, nhãn và độ khó.
- Tách train/dev trước khi dùng.

---

## Phần D. Môi trường chạy

### D0. Quyết định: sửa trọng số tại máy, chạy model trên Kaggle (bf16)
Quyết định 07/11/2026 (người dùng quyết định). Thay cho quyết định "chạy toàn bộ trên máy cá nhân" ghi cùng ngày.

Lý do không dùng lượng tử hoá (4-bit hoặc 8-bit):
a) Sửa trọng số nhạy với độ chính xác. Lượng tử hoá có nguy cơ làm mất bản sửa, hoặc làm điểm đo lệch so với model bf16 mà BTC chấm. CHƯA XÁC MINH bằng phép thử, nhưng người dùng chọn tránh rủi ro này.
b) Máy có GPU 6 GB, không chứa nổi Qwen3-4B ở bf16 (khoảng 8 GB) nếu không lượng tử hoá.

Lý do không tốn phí API: Opus chạy trên máy qua Claude Code đăng nhập gói Pro, không gọi API. Nếu máy có biến môi trường `ANTHROPIC_API_KEY` thì Claude Code dùng key đó và bị tính phí API, cần xoá biến này. Opus tiêu hao hạn mức Pro nhanh.

### D1. Luồng làm việc
1. **Tại máy:** Opus sinh script sửa trọng số (`weights.py`), xuất ra delta (các trọng số bị đổi, không lưu cả model). Bước này chạy được trên CPU, đã kiểm trên máy.
2. **Upload lên Kaggle** (dataset): script và delta.
3. **Trên Kaggle:** notebook nạp Qwen3-4B bf16, áp bản sửa để tạo model đã chỉnh, chạy tập **dev**, ghi điểm ra `/kaggle/working`. Các bước cần chạy model (tìm hướng gây lỗi `directions.py`, đo `run_round.py`) đều chạy ở đây.
4. **Tải kết quả về máy:** Opus đọc điểm, tạo bản sửa tiếp. Gom nhiều ứng viên sửa vào một lần chạy (nạp model một lần, áp từng delta trong bộ nhớ, đo, hoàn nguyên) để giảm số lần khởi động notebook.
5. **Cuối cùng:** chạy tập test BTC một lần để so với baseline. Vòng lặp chỉ dùng dev, không đưa log lỗi test cho Opus.

### D2. Lưu ý về Kaggle (con số gần đúng, cần đối chiếu trên trang Kaggle)
- GPU: 1x P100 hoặc 2x T4 (16 GB mỗi card), RAM khoảng 30 GB, hạn mức khoảng 30 giờ GPU mỗi tuần, một phiên tối đa khoảng 9 đến 12 giờ, ổ ghi khoảng 20 GB.
- Chạy kiểu batch (Save Version), vì phiên tương tác tính vào hạn mức kể cả khi không chạy gì.
- **T4 và P100 không hỗ trợ bf16 gốc.** Trọng số vẫn lưu ở bf16, nhưng phép tính có thể phải chạy ở fp16 hoặc bf16 mô phỏng (chậm hơn). CHƯA XÁC MINH kết quả khớp với bf16 mà BTC chấm, cần so thử trên vài mẫu trước khi tin điểm đo.
- ToolEnv của BTC phải cài được trên Kaggle (cần Internet hoặc upload như dataset).

### D3. Phương án đã bỏ
- Lượng tử hoá 4-bit hoặc 8-bit trên GPU 6 GB của máy: lý do ở D0. Có thể xét lại nếu có phép thử cho thấy bản sửa còn nguyên sau lượng tử hoá.
- Chạy bf16 trên CPU hoặc chia GPU và CPU: đúng model nhưng rất chậm, chỉ hợp để kiểm cuối.

### D4. Cần kiểm tra trước khi làm
- Thể lệ có cho phép dùng Opus sửa trọng số và dùng Kaggle không.
- Bản Qwen đang dùng (Qwen3-4B) khớp với bản thể lệ yêu cầu.
- Thời gian nạp model và thời gian đo 50 mẫu dev trên Kaggle, để tính số ứng viên mỗi lần chạy.
- Fp16 và bf16 trên Kaggle cho kết quả khớp (xem D2).

---

## Rủi ro chính

- **Rò rỉ test:** đã tránh nhờ thiết kế (tập test để riêng, dữ liệu tool call xử lý riêng). Chỉ cần giữ nguyên nguyên tắc này, không đưa log lỗi của test cho Opus.
- **Quá nhiều mẫu dễ:** xử lý bằng cách đặt tỉ lệ có chủ đích khi sinh (xem C3).
- **Giới hạn gọi ToolEnv:** "gọi nhiều hết sức" có thể chạm rate limit hoặc vi phạm thể lệ.
- **Finetune ghi đè bản sửa trọng số:** cần thử cả hai thứ tự.

---

## Nguồn

Method chính (weight-space programming):
1. Takehi, Teoh, Zhang, Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, 02/11/2026. https://rikiyat.github.io/blog/weight-space-programming/
2. Log đầy đủ các thí nghiệm của blog (dùng để dựng harness, xem Thí nghiệm 8). https://github.com/RikiyaT/lm_understand-logs
3. Qwen3.5-0.8B đã sửa trọng số từ Thí nghiệm 8. https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

Cơ sở cho thiết kế dữ liệu và đánh giá:
4. When2Call: When (not) to Call Tools, NAACL 2025. https://arxiv.org/abs/2504.18851
5. Learning to Ask: When LLM Agents Meet Unclear Instruction, EMNLP 2025. https://arxiv.org/abs/2409.00557
6. τ-bench: A Benchmark for Tool-Agent-User Interaction, ICLR 2025. https://arxiv.org/abs/2406.12045
7. Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024. https://arxiv.org/abs/2310.01798
8. Panickssery, Bowman, Feng. *LLM Evaluators Recognize and Favor Their Own Generations*, NeurIPS 2024. https://papers.nips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html
9. Berkeley Function Calling Leaderboard v3 (tài liệu EvalScope). https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html


Thông tin Kaggle (tổng hợp thứ cấp, cần đối chiếu trên trang chính thức của Kaggle):
10. Tổng hợp GPU miễn phí. https://aimultiple.com/free-cloud-gpu
11. Ghi chú về phiên batch và hạn mức tuần. https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html