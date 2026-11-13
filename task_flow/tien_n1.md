# Task flow: Tiên, n1 (chuẩn hoá phương ngữ sau ASR)

## Mục tiêu
Biến transcript ASR có từ địa phương thành tiếng Việt phổ thông, giữ nguyên tên, số tiền, giờ, để Planner hiểu đúng.
Điều kiện xong: điểm dev (300 mẫu, 9 app lạ) không thấp hơn khi không chuẩn hoá, tổng tham số dưới 6B, mỗi mẫu vẫn dưới 30 giây.

## Vị trí trong pipeline
Audio → ASR (PhoWhisper, Khánh) → transcript thô có phương ngữ
→ [Bước này] chuẩn hoá phương ngữ → câu phổ thông
→ Entity guard (Khánh): so số, giờ, tên trước và sau; lệch thì dùng lại transcript gốc
→ Planner (Qwen 4B).

Ví dụ: "Tui hông biết chi mô, chuyển cho Lan hai trăm nghìn đi"
→ "Tôi không biết gì đâu, chuyển cho Lan hai trăm nghìn đi" (tên "Lan" và "hai trăm nghìn" giữ nguyên).

## Hai phương án
A) Rules + prompt trong Qwen 4B (plan hiện tại của nhóm): từ điển phương ngữ lọc từ Wiktionary (CC BY-SA 4.0) đưa vào prompt Mapper. Không tốn tham số, không thêm model.
B) Model seq2seq riêng: ViT5-base hoặc mBART-large-50, fine-tune trên dữ liệu song song phương ngữ sang phổ thông.

A làm mốc. B là thí nghiệm, chỉ giữ nếu hơn A rõ và qua đủ cổng ở mục "Cổng go/no-go".

## Luồng dự định
0. Đo xem có đáng làm không  [ ] ClickUp subtask: Đo tỉ lệ phương ngữ
   - Làm: đếm trong `audio_text` train (gồm lượt history) bao nhiêu câu có từ địa phương, dùng từ điển Wiktionary có nhãn miền. Chạy PhoWhisper trên 100 audio train, xem ASR giữ hay tự chuẩn hoá từ địa phương.
   - Đo: % câu có phương ngữ, tách theo miền; % lỗi Planner trên dev có dính câu phương ngữ.
   - Dự phòng: dưới khoảng 5% câu có phương ngữ thì dừng B, chỉ làm A (ngưỡng đề xuất, chốt lại khi có số).
1. Phương án A: rules trong prompt  [ ] ClickUp subtask: Rules phương ngữ cho Mapper
   - Làm: lọc từ điển Wiktionary theo train và catalog, viết `prompts/rules/dialect_vi_<ngày>.md`, giao Kiệt và Vương chèn vào prompt Mapper.
   - Đo: độ phủ từ phương ngữ trên ViDia2Std (chỉ đo), số token thêm vào prompt, F1 dev.
2. Kiểm cổng cho B trước khi train  [ ] ClickUp subtask: Cổng giấy phép và tham số
   - Làm: hỏi BTC về dữ liệu CC BY-NC; đếm tham số thật của ViT5-base và mBART-large-50 bằng `scripts/count_params.py`; chốt với nhóm phần nào nhường tham số.
   - Đo: số tham số thật, tổng hệ thống sau khi thêm.
   - Dự phòng: không qua cổng thì dừng B.
3. Fine-tune B  [ ] ClickUp subtask: Fine-tune model chuẩn hoá phương ngữ
   - Làm: ưu tiên ViT5-base (nhỏ hơn, có tiền huấn luyện tiếng Việt). Fine-tune trên Kaggle T4. Dữ liệu: nguồn được phép theo cổng 2, thêm cặp "câu phổ thông → chính nó" từ `audio_text` train để model học giữ nguyên câu đã chuẩn.
   - Đo: BLEU/token-F1 trên tập test của nguồn dữ liệu; tỉ lệ làm hỏng thực thể (số, giờ, tên) trên `audio_text` train.
4. Gắn vào pipeline và ablation  [ ] ClickUp subtask: Ablation chuẩn hoá phương ngữ
   - Làm: 3 cấu hình trên dev: không chuẩn hoá, A, B (luôn có Entity guard).
   - Đo: F1 dev (act/ask/empty), độ trễ mỗi mẫu (cả lịch sử tới 24 lượt), tỉ lệ Entity guard phải chặn.
   - Dự phòng: B không hơn A rõ thì giữ A.
5. Chốt và ghi lại (trước 15/11)  [ ] ClickUp subtask: Chốt chuẩn hoá phương ngữ
   - Làm: ghi `docs/DECISIONS.md`, cập nhật `configs/budget.yaml` nếu giữ B.

## Cổng go/no-go cho phương án B
a) Giấy phép dữ liệu: ViDia2Std là CC BY-NC 4.0 (cấm thương mại). Model fine-tune trên nó rồi nộp trong Docker có thể vi phạm yêu cầu "giấy phép cho dùng thương mại". Ngày 13/11 nhóm đã chốt chỉ dùng ViDia2Std để đo. Muốn dùng để train phải hỏi BTC (aichallenge@zalo.me) và có văn bản trả lời. CHƯA XÁC MINH.
b) Ngân sách tham số: hệ thống hiện khoảng 5,9B, dư khoảng 0,1B. ViT5-base khoảng 0,2 đến 0,3B, mBART-large-50 khoảng 0,6B (CHƯA XÁC MINH, đếm bằng code). Cả hai đều vượt phần dư, nên phải hạ ASR xuống PhoWhisper-medium (lấy lại khoảng 0,8B) hoặc bỏ Laya (0,32B). Cần cả nhóm đồng ý.
c) Độ trễ: thêm một lượt sinh seq2seq cho mỗi lượt user, lịch sử tới 24 lượt, trần 30 giây trên A30.
d) Lệch domain: ViDia2Std là bình luận Facebook, không phải câu lệnh giọng nói hay transcript ASR. Cần đo trên `audio_text` train, không chỉ trên tập test của ViDia2Std.
e) Giấy phép model gốc: ViT5 (VietAI) và mBART-50 (Meta) cần kiểm giấy phép trên model card. CHƯA XÁC MINH.

## Nguồn
- ViDia2Std: https://huggingface.co/datasets/Biu3010/ViDia2Std, bài báo https://arxiv.org/abs/2603.10211 (CC BY-NC 4.0, 13.657 cặp câu, 3 miền).
- which-dialect (Wiktionary): https://github.com/nguyen18/which-dialect (dữ liệu CC BY-SA 4.0, 1.330/42.419 nghĩa có nhãn miền).

## Note tiến độ (mới nhất ở trên)
- 13/11: Lập flow. Chọn A làm mốc, B là thí nghiệm có cổng. Chưa chạy gì.

## Cần trao đổi
- Cả nhóm: có chấp nhận thêm model và nhường tham số (ASR medium hoặc bỏ Laya) không.
- Khánh: PhoWhisper có tự chuẩn hoá từ địa phương không; Entity guard nhận đầu ra bước này.
- Kiệt, Vương: chèn rules A vào prompt Mapper và đo dev.
