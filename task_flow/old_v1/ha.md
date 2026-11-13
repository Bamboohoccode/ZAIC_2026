# Task flow: Hà (nhánh verifier/ha)

## Mục tiêu
Verifier dưới 0,4B tham số, chạy nhanh. Input là các action đã lưu trong dialogue state và plan của Planner, output chỉ là ĐÚNG hoặc SAI kèm confidence. Khi SAI thì hỏi thẳng user (`sys.ask_missing_info`), chỗ lệch do hàm diff bằng code chỉ ra.
Điều kiện xong: điểm cuối không thấp hơn khi không có Verifier, mỗi mẫu thêm dưới 100 ms (PROJECT_OVERVIEW.md mục 5 và 5.3).

## Luồng dự định
1. Smoke-test Laya-multilingual trên tiếng Việt (hạn 07/11)  [ ] ClickUp subtask: Smoke-test Laya tiếng Việt  (07/11: xong trên CPU, còn đo độ trễ GPU trên Kaggle)
   - Làm: chạy `lifemate/verifier/smoke_test_laya.py` trên Kaggle GPU, báo kết quả cho nhóm.
   - Đo: acc phân loại 5 app (có dấu, không dấu, phương ngữ, so với tiếng Anh), AUC câu hỏi có/không, AUC zero-shot bài Verifier, độ trễ p50/p95.
   - Dự phòng: tiếng Việt yếu (acc có dấu < 0,60 hoặc AUC có/không < 0,70) thì báo nhóm và chuyển sang bước 2 làm Verifier chính (luật + diff), theo 5.3d.
2. Trích action đã lưu và viết hàm diff `diff.py` (hạn 08/11)  [x] ClickUp subtask: Hàm diff plan và state  (07/11: xong, đã đo trên dữ liệu BTC)
   - Làm: lấy danh sách call đã chạy từ lịch sử hội thoại (trước khi có module state của Kiệt, Vương). So plan với state: lặp lại call ghi đã chạy, dùng ID chưa xuất hiện trong kết quả tool, thiếu hoặc thừa call. Hàm này vừa là đầu vào soạn câu hỏi user khi SAI, vừa là Verifier luật dùng làm mốc.
   - Đo: trên trace train (không dùng trace dev), tỉ lệ plan sai bị diff bắt và tỉ lệ báo nhầm trên plan đúng.
   - Dự phòng: nếu bắt được ít, thu hẹp diff về các lỗi chắc chắn làm lệch hash (lặp write, ID bịa).
3. Sinh dữ liệu fine-tune `data.py` (hạn 09/11)  [x] ClickUp subtask: Sinh dữ liệu ĐÚNG/SAI  (08/11: xong trên dữ liệu BTC, chưa có lỗi thật từ dự đoán train)
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
- 08/11: Xong bước 3. Push `verifier/ha` (commit 910ef18).
  - `data.py` thêm kiểu lỗi `add_arg`: thêm tham số không bắt buộc mà gold không có (enum chọn bừa, đoạn cắt từ lời user, ngày giờ có trong state). Đây là lỗi thật nhiều nhất của Planner (35% call sai dev_v4), trước đó chưa có trong dữ liệu.
  - Chạy trên train BTC (dev = 300 mẫu của `eval/splits/dev_ids.txt`): không query train 4943 (ĐÚNG 2898) / dev 1063 (ĐÚNG 643); có query train 5701 / dev 1281; có query `--neg-per-pos 2` train 7894 / dev 1840. File ở `runs/` (local, không commit).
  - `oracle_ceiling.py` thêm chế độ chấm Verifier thật (`--rules`, `--model`, `--scores`): bỏ call có P(ĐÚNG) dưới ngưỡng, chấm lại bằng `eval/scorer.py`, báo điểm, số call bỏ đúng/bỏ nhầm, AUC trên call thật.
  - `diff.py` làm Verifier trên dev_v4: AUC 0,50 trên 546 call thật, bỏ 1 call sai, điểm không đổi (0,3597). Xác nhận Verifier không có query gần như vô dụng.
  - Chạy thử `--model` bằng checkpoint train trên dữ liệu giả (AUC 0,51): bỏ khoảng 500/546 call, điểm tụt còn khoảng 0,09. Verifier kém làm hại nặng, phải chọn ngưỡng thấp và luôn so với mốc 0,3597.
  - Tiếp theo (Kaggle GPU): train mmBERT-small trên `ver_q.jsonl`, chấm bằng `oracle_ceiling.py --model ... --with-query`. Điểm tăng dưới khoảng 0,01 thì dừng Verifier (bước 8), dồn sức sang Planner.
- 07/11 (khuya): Nhận dữ liệu BTC v3.2.0 từ Vương. Chạy `analyze_errors.py` chế độ full trên dự đoán dev của Vương (300 mẫu, --text):
  - v4: 546 call dự đoán, 255 đúng, 291 sai/thừa. Trong 291 call sai: thêm tham số gold không có 101 (35%), sai tool hoặc thừa 92 (32%), sai giá trị 72 (25%), hỏi thừa 21 (7%), `diff.py` bắt được 1 (0,3%), báo nhầm 0. no_verify và v1 tương tự (diff bắt 0 và 1).
  - Planner bỏ sót 388/643 call gold (60%). Đây là nguồn mất điểm lớn nhất, Verifier không sửa được (chỉ bỏ call, không thêm call).
- 07/11 (khuya): Viết `oracle_ceiling.py`: trần điểm khi Verifier đoán đúng 100% (dùng gold), chấm bằng `eval/scorer.py` (chấm lại v4 ra đúng 0,3597). Kết quả trên dev_v4:
  - SAI thì hỏi user rồi dừng (thiết kế hiện tại): 0,3063 (**-0,053**). Làm giảm điểm kể cả khi Verifier hoàn hảo.
  - Bỏ mọi call sai: 0,3760 (+0,016). Bỏ có chọn lọc: 0,4218 (+0,062). Phần tăng chủ yếu ở mẫu `empty` (10 mẫu) và `ask` (11 mẫu); bỏ call sai làm giảm 25 mẫu act.
  - Push `verifier/ha` (commit c1ecf3c, d7b95b9).
- 07/11 (đêm): **Phát hiện quan trọng cần báo nhóm.** Chạy `analyze_errors.py` (chế độ partial, không cần dữ liệu) trên dự đoán dev của Vương (no_verify, v1, v4; 300 mẫu): 0 call trùng lặp, 0 call sau ask, 30/30 mẫu có ask đặt ask ở cuối. Lý do: `pipeline.run_agent` đã chặn sẵn ID bịa, đọc lặp và dừng sau ask. Nghĩa là lỗi kiểu "lệch với state" gần như không lọt ra; 46% mẫu lệch hash nhiều khả năng là lỗi sai ý user, Verifier không có query không bắt được. Cần chạy chế độ full (có data) để xác nhận và đo phần "cần query". Đề xuất nhóm xem lại quyết định 06/11 (Verifier không dùng query), khớp với hướng của Tiên ở task_flow/tien.md mục C4.
- 07/11 (đêm): Push lên `verifier/ha` (commit 78036cf): `analyze_errors.py` (phân loại lỗi Planner), `data.py` (sinh ĐÚNG/SAI, có/không query, thêm lỗi thật bỏ dev), `train.py` (đổi backbone mmBERT/XLM-R/mDeBERTa, chạy thông trên CPU với mmBERT-small), README module, 22 test pass.
- 07/11 (tối): Chạy smoke-test Laya thật trên CPU local (laya 0.4.27, torch 2.15 CPU, `convaiinnovations/laya-multilingual`, đếm được 321.908.995 tham số). Kết quả (bộ thử nhỏ, 10 đến 15 câu mỗi phần, chỉ để định hướng):
  - A. Phân loại 5 app: có dấu 0,90, phương ngữ 0,90, tiếng Anh 1,00, không dấu 0,50 (đoán bừa 0,20). Tiếng Việt có dấu chỉ kém tiếng Anh 0,10.
  - B. Có/không về chi tiết (số tiền, giờ, phủ định): AUC vi 0,72, en 0,86. Vừa qua ngưỡng gợi ý 0,70, tiếng Việt kém tiếng Anh rõ.
  - C. Zero-shot bài Verifier: AUC 0,61. P(nhất quán) trung bình: plan đúng 0,63, ID bịa 0,22 (bắt được), lặp write 0,79 (KHÔNG bắt được, còn cao hơn plan đúng).
  - Độ trễ CPU: p50 315 ms, p95 435 ms mỗi câu hỏi. Chưa đại diện cho A30, cần đo lại trên GPU Kaggle.
  - Kết luận: Laya hiểu được tiếng Việt có dấu, đi tiếp fine-tune, chưa cần phương án dự phòng. Kết quả chi tiết: `runs/laya_smoke_cpu.json` (local, không commit).
- 07/11 (tối): Viết và push `lifemate/verifier/diff.py` + 15 test (commit dc75eca): bắt ID bịa, lặp write/read, call sau ask, tool lạ, hỏi lại tham số đã biết; có `verdict` (Verifier luật làm mốc), `suggest` (bỏ call hay hỏi user), `to_text` (input cho Laya). Chưa đo trên dữ liệu thật.
- 07/11: Đọc repo và tài liệu, tạo nhánh `verifier/ha`. Viết và push `lifemate/verifier/smoke_test_laya.py` (commit ea607b7): kiểm tra hiểu tiếng Việt, bắt chi tiết có/không, zero-shot bài Verifier, độ trễ. Đã chạy thử luồng script bằng module giả, CHƯA chạy model thật vì máy local không có GPU, sẽ chạy trên Kaggle. Tra model card: `laya-multilingual` 321,9M tham số, có `vi` trong danh sách ngôn ngữ, ship chưa hiệu chỉnh nhiệt độ, câu hỏi có/không có xu hướng báo "có" thấp.

## Quyết định và điểm đo
- 07/11: Chấm câu hỏi có/không của Laya bằng AUC và acc ở ngưỡng tốt nhất, không dùng ngưỡng 0,5 cố định, vì model card ghi Laya chưa hiệu chỉnh và báo "có" thấp.
- 07/11 (khuya): **Đề xuất đổi thiết kế:** Verifier báo SAI thì BỎ call (hoặc cho Planner làm lại), KHÔNG hỏi user. Lý do: trần điểm của "SAI thì hỏi user" là -0,053 so với không Verifier (oracle_ceiling.py). Chờ nhóm chốt.
- 07/11 (khuya): Trần đóng góp của Verifier thấp (tối đa +0,016 đến +0,062 với Verifier hoàn hảo). Đề xuất: giữ Verifier nhẹ, tập trung vào "có nên hành động không" (mẫu empty/ask), ưu tiên chung của nhóm dồn vào Planner.
- 07/11: Ghép hai tầng: `diff.py` bắt lỗi chắc chắn bằng code (lặp write, ID bịa), Laya chỉ lo phần ngữ nghĩa còn lại. Lý do: zero-shot Laya không bắt được lặp write (smoke-test phần C).
- 07/11: Ứng viên thay Laya nếu Laya fine-tune không đạt (đã kiểm giấy phép trên Hugging Face). Vừa ngân sách khoảng 0,4B: `jhu-clsp/mmBERT-base` (MIT, khoảng 307M, chính là nền của Laya), `FacebookAI/xlm-roberta-base` (MIT, 279M), `microsoft/mdeberta-v3-base` (MIT, khoảng 278M). Loại: PhoBERT-base-v2 (AGPL-3.0), ViDeBERTa (không ghi giấy phép). Quá ngân sách, chỉ khi hạ ASR: `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0, 596M). Viết `train.py` theo kiểu đổi được backbone bằng một tham số để so trên cùng dữ liệu.
- 08/11: Quy tắc dừng: nếu Verifier thật (ngưỡng tốt nhất trên dev) tăng dưới khoảng 0,01 so với dev_v4 (0,3597) thì đóng băng Verifier, không làm bước 5 đến 7. Lý do: trần tối đa chỉ +0,062, 300 mẫu dev không đủ để tin mức tăng nhỏ.
- Mốc so sánh: không Verifier (`verify_rounds=0`) F1 dev 0,3109, 24,3 s/mẫu; self-verify Qwen 2 lượt F1 0,3117, 31,0 s/mẫu (Vương đo 06/11). Verifier của Hà phải hơn mốc không Verifier mà gần như không thêm thời gian.

## Cần trao đổi
- Cả nhóm (gấp): đổi PROJECT_OVERVIEW mục 4c, "Verdict WRONG -> hỏi user" thành "WRONG -> bỏ call". Số liệu ở note 07/11 khuya.
- Vương: 101/291 call sai (35%) là do Planner thêm tham số mà gold không có (vd `tv.guide` thêm `channel`, `calendar.list_events` thêm ngày). Lỗi lớn nhất của Planner sau bỏ sót call.
- Vương: xin file dự đoán của Planner trên phần train (không thuộc dev) để `data.py --pred` thêm lỗi thật vào dữ liệu train Verifier.
- Tiên: `lifemate/contracts.py` còn `Verdict` cũ (`ACCEPT|REJECT|ASK`, `reason`), cần đổi sang `CORRECT|WRONG`, `confidence`, `diff` theo PROJECT_OVERVIEW.md mục 4. Cần thống nhất điểm cắm Verifier trong `plan_step` của `pipeline.py`.
- Kiệt, Vương: định dạng "action đã lưu" mà module state sẽ xuất ra. Trước khi có, Hà tự trích từ lịch sử hội thoại.
- Cả nhóm: Verifier có nhận lời user (query) không? Số liệu partial cho thấy không có query thì gần như không còn lỗi để bắt (xem note 07/11 đêm); chế độ full xác nhận (diff.py AUC 0,50 trên call thật, note 08/11). Đề xuất: có query.
- Cả nhóm (mục 9g): khi Verifier báo SAI, `sys.ask_missing_info` lấy `tool_name` và `params` từ đâu. Đề xuất: tool của call bị lệch và các tham số mà diff chỉ ra.
