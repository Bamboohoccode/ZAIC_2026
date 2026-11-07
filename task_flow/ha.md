# Task flow: Hà (nhánh verifier/ha)

## Mục tiêu
Verifier dưới 0,4B tham số, chạy nhanh. Input là các action đã lưu trong dialogue state và plan của Planner, output chỉ là ĐÚNG hoặc SAI kèm confidence. Khi SAI thì hỏi thẳng user (`sys.ask_missing_info`), chỗ lệch do hàm diff bằng code chỉ ra.
Điều kiện xong: điểm cuối không thấp hơn khi không có Verifier, mỗi mẫu thêm dưới 100 ms (PROJECT_OVERVIEW.md mục 5 và 5.3).

## Luồng dự định
1. Smoke-test Laya-multilingual trên tiếng Việt (hạn 07/11)  [ ] ClickUp subtask: Smoke-test Laya tiếng Việt
   - Làm: chạy `lifemate/verifier/smoke_test_laya.py` trên Kaggle GPU, báo kết quả cho nhóm.
   - Đo: acc phân loại 5 app (có dấu, không dấu, phương ngữ, so với tiếng Anh), AUC câu hỏi có/không, AUC zero-shot bài Verifier, độ trễ p50/p95.
   - Dự phòng: tiếng Việt yếu (acc có dấu < 0,60 hoặc AUC có/không < 0,70) thì báo nhóm và chuyển sang bước 2 làm Verifier chính (luật + diff), theo 5.3d.
2. Trích action đã lưu và viết hàm diff `diff.py` (hạn 08/11)  [ ] ClickUp subtask: Hàm diff plan và state
   - Làm: lấy danh sách call đã chạy từ lịch sử hội thoại (trước khi có module state của Kiệt, Vương). So plan với state: lặp lại call ghi đã chạy, dùng ID chưa xuất hiện trong kết quả tool, thiếu hoặc thừa call. Hàm này vừa là đầu vào soạn câu hỏi user khi SAI, vừa là Verifier luật dùng làm mốc.
   - Đo: trên trace train (không dùng trace dev), tỉ lệ plan sai bị diff bắt và tỉ lệ báo nhầm trên plan đúng.
   - Dự phòng: nếu bắt được ít, thu hẹp diff về các lỗi chắc chắn làm lệch hash (lặp write, ID bịa).
3. Sinh dữ liệu fine-tune `data.py` (hạn 09/11)  [ ] ClickUp subtask: Sinh dữ liệu ĐÚNG/SAI
   - Làm: từ train, plan khớp gold là ĐÚNG. Sửa tên tool, đổi một tham số, bỏ một call, thêm call thừa, lặp write là SAI. Chia theo app, giữ app lạ để kiểm tra khái quát (cùng cách chia với `eval/splits/`).
   - Đo: số mẫu mỗi nhãn, mỗi kiểu lỗi, tỉ lệ ĐÚNG/SAI cân bằng.
   - Dự phòng: thiếu mẫu SAI thật thì thêm lỗi thật của Planner lấy từ trace trên phần train (không lấy từ dev để tránh rò rỉ).
4. Fine-tune Laya `train.py` (hạn 11/11)  [ ] ClickUp subtask: Fine-tune Laya
   - Làm: fine-tune `convaiinnovations/laya-multilingual` trên Kaggle 2xT4 theo notebook fine-tune của Laya. API fine-tune chưa xác minh từ README, xem thử trước.
   - Đo: AUC và acc trên app giữ lại, so với Verifier luật ở bước 2.
   - Dự phòng: fine-tune không chạy hoặc không hơn luật thì thử `laya.Agent(vegaml=True).fit(...)` (fit head trên đặc trưng đóng băng), sau đó tới Planner tự kiểm một lượt (5.3d).
5. Gắn vào pipeline và ablation (hạn 12/11, mốc M2)  [ ] ClickUp subtask: Ablation Verifier
   - Làm: phối hợp Tiên thay self-verify trong `plan_step` của `lifemate/pipeline.py` bằng Verifier của Hà (Hà không sửa trực tiếp file của Tiên).
   - Đo: F1 dev (act/ask/empty) và độ trễ của 3 cấu hình: không Verifier (`verify_rounds=0`), Verifier luật, Laya. Theo dõi riêng tỉ lệ hỏi thừa trên nhóm đủ thông tin.
   - Dự phòng: Laya không hơn luật thì giữ Verifier luật; cả hai không hơn "không Verifier" thì báo nhóm bỏ Verifier.
6. Chọn ngưỡng và hiệu chỉnh xác suất (hạn 14/11, mốc M3)  [ ] ClickUp subtask: Chọn ngưỡng Verifier
   - Làm: hiệu chỉnh nhiệt độ trên dữ liệu giữ lại (Laya ship chưa hiệu chỉnh), chọn ngưỡng theo đường cong điểm trên dev. Không chỉnh ngưỡng trên tập kiểm chứng cuối.
   - Đo: F1 cuối theo ngưỡng, tỉ lệ SAI nhầm (hỏi thừa) và bỏ sót SAI (lọt call sai).
7. Xuất ONNX, đo độ trễ, đếm tham số (hạn 14/11)  [ ] ClickUp subtask: Đo độ trễ Verifier
   - Làm: chạy PyTorch hoặc ONNX Runtime (`laya[onnx]`), đo trên GPU gần A30, cập nhật `configs/budget.yaml` qua Tiên.
   - Đo: ms mỗi mẫu (mục tiêu < 100 ms), số tham số thật (Hub ghi 321.908.998).
8. Đóng băng (15/11)  [ ] ClickUp subtask: Đóng băng Verifier
   - Làm: chốt checkpoint, ngưỡng, README module, ghi `docs/DECISIONS.md`, PR vào `main` trước tag `freeze-15-11`.

## Note tiến độ (mới nhất ở trên)
- 07/11: Đọc repo và tài liệu, tạo nhánh `verifier/ha`. Viết và push `lifemate/verifier/smoke_test_laya.py` (commit ea607b7): kiểm tra hiểu tiếng Việt, bắt chi tiết có/không, zero-shot bài Verifier, độ trễ. Đã chạy thử luồng script bằng module giả, CHƯA chạy model thật vì máy local không có GPU, sẽ chạy trên Kaggle. Tra model card: `laya-multilingual` 321,9M tham số, có `vi` trong danh sách ngôn ngữ, ship chưa hiệu chỉnh nhiệt độ, câu hỏi có/không có xu hướng báo "có" thấp.

## Quyết định và điểm đo
- 07/11: Chấm câu hỏi có/không của Laya bằng AUC và acc ở ngưỡng tốt nhất, không dùng ngưỡng 0,5 cố định, vì model card ghi Laya chưa hiệu chỉnh và báo "có" thấp.
- Mốc so sánh: không Verifier (`verify_rounds=0`) F1 dev 0,3109, 24,3 s/mẫu; self-verify Qwen 2 lượt F1 0,3117, 31,0 s/mẫu (Vương đo 06/11). Verifier của Hà phải hơn mốc không Verifier mà gần như không thêm thời gian.

## Cần trao đổi
- Tiên: `lifemate/contracts.py` còn `Verdict` cũ (`ACCEPT|REJECT|ASK`, `reason`), cần đổi sang `CORRECT|WRONG`, `confidence`, `diff` theo PROJECT_OVERVIEW.md mục 4. Cần thống nhất điểm cắm Verifier trong `plan_step` của `pipeline.py`.
- Kiệt, Vương: định dạng "action đã lưu" mà module state sẽ xuất ra. Trước khi có, Hà tự trích từ lịch sử hội thoại.
- Cả nhóm (mục 9g): khi Verifier báo SAI, `sys.ask_missing_info` lấy `tool_name` và `params` từ đâu. Đề xuất: tool của call bị lệch và các tham số mà diff chỉ ra.
