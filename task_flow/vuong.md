# [Vương] Luồng làm việc: Planner, Mapper, State, Validator, GEPA

Phạm vi (ClickUp & update_5_flow): Nền chung cùng Kiệt (`lifemate/planner/`, `lifemate/mapper/`, `lifemate/state/`, `lifemate/validator/`, `eval/`, `synth/`, `optimize/gepa/metric/`). Thư mục riêng: `optimize/gepa/vuong/`, `prompts/gepa/vuong/`.
Nhánh: `planner/vuong` (hoặc `gepa/vuong`). Hạn: đóng băng 15/11, nộp Docker 16/11 23:59.

## Quyết định đã chốt & Dữ liệu thực tế
| # | Quyết định / Phát hiện | Ý nghĩa thực tế |
|---|---|---|
| D1 | Lấy `lifemate/pipeline.py` làm baseline chạy được | Không đập đi xây lại khi chưa có số đo |
| D2 | Replay gold 100 mẫu train: score 1.0 | Environment `zaic_tools` và scorer hoạt động hoàn hảo |
| D3 | Chạy Qwen3-4B trên Kaggle GPU (2xT4 / P100 miễn phí) qua notebook 1-click; local chỉ code & test | Máy local (RTX 4060 8GB) không đủ VRAM cho 4B bf16 + 16K context |
| D4 | Dev set 300 mẫu (`eval/splits/dev_ids.txt`): 9 app lạ (20% số app), tỉ lệ `act` (78.7%), `ask` (14.0%), `empty` (7.3%) | Đánh giá năng lực zero-shot app lạ chuẩn xác như public test |
| D5 | Thử nghiệm Planner bằng `--text` (audio_text) để cô lập khả năng suy luận logic | Tránh nhiễu từ ASR khi tối ưu prompt và validator |
| D6 | GEPA chạy thủ công trong chat trước (agent phân tích failure trace, sửa có kiểm chứng) | Miễn phí, linh hoạt, kiểm soát từng thay đổi |
| D7 | **Dữ liệu thực tế `sys.ask_missing_info`:** Trong 217 mẫu train, 100% nằm ở call cuối cùng (60 mẫu hỏi ngay, 157 mẫu hỏi sau khi gọi tool đọc). Đặc biệt: 78/217 mẫu (36%) hỏi tham số **không nằm trong `required`** (do user nói mập mờ). | Rule máy móc "thiếu required mới hỏi" chỉ bắt được tối đa ~64% mẫu gold ask. Validator & Planner cần phân biệt giữa thiếu required và thông tin mơ hồ. |

---

## Luồng nhiệm vụ (Chi tiết từng Subtask)

| # | Bước | Hạn | Điều kiện xong | Trạng thái |
|---|---|---|---|---|
| 1 | Tạo `eval/splits/dev_ids.txt` (chia theo app) + script chạy dev + báo điểm act/ask/empty | 06/11 | `dev_ids.txt` (300 mẫu, 9 held-out apps), `scorer.py` chẩn đoán chi tiết, `pipeline.py --ids` chạy được | **[XONG 06/11]** |
| 2 | Baseline Qwen3-4B trên Kaggle GPU: chạy `--text --ids eval/splits/dev_ids.txt`, lưu trace | 07/11 | Có file `runs/dev_baseline.jsonl` + điểm F1 baseline + latency ghi vào `docs/DECISIONS.md` | **Đang tiến hành** (Tạo notebook Kaggle) |
| 3 | Phân nhóm lỗi (script chẩn đoán): sai tool, sai tham số, read thừa, thiếu ask, ask thay vì làm, lệch hash | 07/11 | Bảng thống kê % lỗi theo nhóm từ kết quả Bước 2 | Kế tiếp sau Bước 2 |
| 4 | Refactor module: `lifemate/planner`, `lifemate/state`, `lifemate/validator` | 08/11 | Unit test pass, điểm dev không đổi sau khi tách | *Tạm dừng chờ Tiên & Kiệt chốt phân chia chi tiết* |
| 5 | Nâng cấp Validator luật 2 tầng: feedback loop + auto fallback `sys.ask_missing_info` | 09/11 | Điểm dev tăng, tỉ lệ call lỗi schema về 0 | *Tạm dừng chờ Kiệt thống nhất nền chung* |
| 6 | Ablation Verify Rounds: đo Planner thuần túy (`verify_rounds=0`) vs `verify_rounds=2` | 10/11 | Đo latency và F1; bàn giao kết quả làm baseline sạch cho Laya của Hà | Sẵn sàng đo sau Bước 2 |
| 7 | State module: Hybrid tracker (bảng Known Entities & Actions bền vững) | 11/11 | Không mất ID quan trọng khi cắt tỉa history, giảm token | *Tạm dừng chờ Kiệt thống nhất nền chung* |
| 8 | Ablation Mapper trên dev: không mapper vs Qwen3-0.6B vs mapper prompt trong Qwen3-4B | 11/11 | Đo điểm cuối + latency + ngân sách tham số | *Chờ xác nhận vai trò với Khánh & Tiên* |
| 9 | Vòng lặp GEPA (Planner `planner_vN.txt` & Mapper `mapper_vN.txt`): lưu version có rollback | 12/11-14/11 | Prompt tối ưu trong `prompts/gepa/vuong/` đạt dev score cao hơn baseline | Chờ sau Bước 6 & 8 |
| 10 | Kiểm tra End-to-End ASR + Mapper, latency $\le$ 30s/mẫu trên GPU A30 | 14/11 | Chốt bản nộp đơn của Vương, sẵn sàng cho Ensemble cùng Kiệt | |
| 11 | Ensemble prompt cùng Kiệt & Đóng băng giải pháp (`freeze-15-11`) | 15/11 | Nộp Docker trước 16/11 23:59 | |

---

## Cần trao đổi & Chờ xác nhận
- **Tiên:** Xác nhận phân công cụ thể giữa Kiệt và Vương trong nền chung (ai làm validator, ai làm state, ai làm synth/metric) để tránh sửa chồng chéo.
- **Khánh:** Khánh muốn làm trực tiếp với Mapper hay giữ ASR + Entity guard? Nếu Khánh làm Mapper thì Vương & Kiệt tập trung toàn lực vào Planner + Validator + State.
- **Tiên (Câu hỏi 9g):** Khi Verifier của Hà báo SAI và hỏi user `sys.ask_missing_info`, lấy `tool_name` và `params` từ đâu? (Đề xuất của Vương: lấy từ tool bị lệch và params do code diff chỉ ra).

---

## Nhật ký
- 05/11: Đọc đề + repo, chốt D1 đến D6, chuyển dữ liệu vào `data/`, replay gold 100 mẫu = 1.0.
- 06/11: Xong Subtask 1: sinh `eval/splits/dev_ids.txt` (300 mẫu, giữ 9 app lạ), cập nhật `eval/scorer.py` hỗ trợ chẩn đoán chi tiết theo `act`/`ask`/`empty`, thêm cờ `--ids` vào `lifemate/pipeline.py`.
- 06/11 (sáng): Đồng bộ commit `update_4_flow` & `update_5_flow` từ Tiên: Cập nhật lại lộ trình subtask, tạm dừng các phần refactor chờ chốt phân công, chuẩn bị Kaggle notebook chạy Baseline Bước 2.
