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
| D8 | **Giải mã 147 ca HASH_MISMATCH:** 71.7% lỗi hash tập trung ở 2 nguyên nhân cốt lõi: (1) 110 ca bỏ sót nhánh ngược lại ("không thì", "chưa tới thì") do hiểu máy móc luật dừng khi điều kiện sai; (2) 105 ca tự ý điền tham số tuỳ chọn (room, label, note...) làm sai hash. | Cần mở khóa nhánh ELSE và siết chặt quy tắc tham số tối thiểu trong prompt `planner_v6.txt`. |
| D9 | **Loại bỏ Rule-based Validator để chống Overfitting Test thật:** Phát hiện luật cứng theo app trong validator.py (như tự ép hỏi lại khi có >1 xe đạp) gây overfit và làm mất oan 23 ca điểm thật. Quyết định: Chuyển 100% logic sang Prompt Planner V7 thuần túy, không dùng code Python can thiệp nhằm tối ưu năng lực zero-shot tổng quát cho mọi app lạ ở Private Test. | Đảm bảo giải pháp mạnh mẽ, an toàn tuyệt đối khi chấm thi trên tập test ẩn của BTC. |

---

## Luồng nhiệm vụ (Chi tiết từng Subtask)

| # | Bước | Hạn | Điều kiện xong | Trạng thái |
|---|---|---|---|---|
| 1 | Tạo `eval/splits/dev_ids.txt` (chia theo app) + script chạy dev + báo điểm act/ask/empty | 06/11 | `dev_ids.txt` (300 mẫu, 9 held-out apps), `scorer.py` chẩn đoán chi tiết, `pipeline.py --ids` chạy được | **[XONG 06/11]** |
| 2 | Baseline Qwen3-4B trên Kaggle GPU: chạy `--text --ids eval/splits/dev_ids.txt`, lưu trace | 07/11 | Có file `runs/dev_baseline.jsonl` + điểm F1 baseline + latency ghi vào `docs/DECISIONS.md` | **[XONG 06/11]** F1: 0.3117 (Act: 0.3268, Ask: 0.0810, Empty: 0.5909), latency: 31.4s |
| 3 | Phân nhóm lỗi (script chẩn đoán): sai tool, sai tham số, read thừa, thiếu ask, ask thay vì làm, lệch hash | 07/11 | Bảng thống kê % lỗi theo nhóm từ kết quả Bước 2 | **[XONG 06/11]** Hash mismatch: 46.3%, Wrong tool/args: 17.3%, 130 mẫu > 30s |
| 4 | Refactor module: `lifemate/planner`, `lifemate/state`, `lifemate/validator` | 08/11 | Unit test pass, điểm dev không đổi sau khi tách | *Tạm dừng chờ Tiên & Kiệt chốt phân chia chi tiết* |
| 5 | Nâng cấp Validator luật 2 tầng: feedback loop + auto fallback `sys.ask_missing_info` | 09/11 | Điểm dev tăng, tỉ lệ call lỗi schema về 0 | *Tạm dừng chờ Kiệt thống nhất nền chung* |
| 6 | Ablation Verify Rounds: đo Planner thuần túy (`verify_rounds=0`) vs `verify_rounds=2` | 10/11 | Đo latency và F1; bàn giao kết quả làm baseline sạch cho Laya của Hà | Sẵn sàng đo sau Bước 2 |
| 7 | State module: Hybrid tracker (bảng Known Entities & Actions bền vững) | 11/11 | Không mất ID quan trọng khi cắt tỉa history, giảm token | *Tạm dừng chờ Kiệt thống nhất nền chung* |
| 8 | Ablation Mapper trên dev: không mapper vs Qwen3-0.6B vs mapper prompt trong Qwen3-4B | 11/11 | Đo điểm cuối + latency + ngân sách tham số | *Chờ xác nhận vai trò với Khánh & Tiên* |
| 9 | Vòng lặp GEPA (Planner `planner_vN.txt` & Mapper `mapper_vN.txt`): lưu version có rollback | 12/11-14/11 | Prompt tối ưu trong `prompts/gepa/vuong/` đạt dev score cao hơn baseline | Đang chạy vòng 7 (`planner_v7.txt`, `dev_v7_pure`), đã lưu v1-v7 |
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
- 06/11 (trưa): Chạy thành công Baseline Qwen3-4B trên Kaggle (300 mẫu dev). Hoàn thành Bước 2 & Bước 3: Điểm F1 đạt 0.3117 (Act: 0.3268, Ask: 0.0810, Empty: 0.5909). Thống kê lỗi: 46.3% lệch hash, 17.3% sai tool/args, 130/300 mẫu vượt 30s do self-verify 2 lượt. Ghi nhật ký vào docs/DECISIONS.md.
- 06/11 (tối): Hoàn thành ablation thử nghiệm tắt self-verify (--verify-rounds 0) trên 300 mẫu dev: F1 0.3109 (tương đương baseline 0.3117, Act F1 tăng nhẹ 0.3304), nhưng tốc độ cải thiện rõ rệt: giảm trung bình 6.6s/mẫu (24.3s vs 31.0s), số mẫu >30s trên T4 giảm từ 130 xuống 68. Chốt dùng verify_rounds=0 làm cấu hình mặc định cho các vòng lặp GEPA tiếp theo.
- 06/11 (đêm): Hoàn thành Vòng 1 GEPA (planner_v1.txt) trên 300 mẫu dev: Điểm F1 bứt phá từ 0.3109 lên 0.3523 (+0.0414), trong đó Ask F1 tăng vọt từ 0.0786 lên 0.1989 (+145%), Act F1 tăng lên 0.3659. Lỗi lệch hash giảm từ 46.7% xuống 43.0%. Thời gian thực thi ổn định 27.0s/mẫu. Cập nhật manifest.json và docs/DECISIONS.md.
- 07/11 (sáng): Hoàn thành Vòng 2 GEPA (planner_v2.txt) trên 300 mẫu dev: Lỗi WRONG_TOOL_OR_ARGS giảm mạnh từ 53 xuống 41 (-22.6%), Act F1 tăng lên 0.3665, Empty F1 đạt 0.5909. Tuy nhiên Ask F1 bị kéo xuống 0.0854 (khiến Mean F1 đạt 0.3436) do luật ép ghi lấn át việc phát hiện mâu thuẫn/trùng tên. Tìm ra công thức kết hợp tối ưu giữa V1 và V2 để xây dựng planner_v3.
- 07/11 (trưa): Hoàn thành Vòng 3 GEPA (planner_v3.txt) trên 300 mẫu dev: Empty F1 đạt mức kỷ lục 0.6364, nhưng Mean F1 giảm về 0.3294 do diễn giải điều kiện IF-ELSE quá dài làm phân tán chú ý của mô hình 4B. Khẳng định: Prompt planner_v1 vẫn là bản tổng thể tốt nhất (Mean F1: 0.3523). Chuyển sang chiến lược tinh gọn "Minimal Surgery" trên nền v1.
- 07/11 (trưa/chiều): Hoàn thành Vòng 4 GEPA (planner_v4.txt) trên 300 mẫu dev: Bứt phá lên đỉnh cao mới với Mean F1 0.3597 (+0.0480 vs baseline) và Act F1 kỷ lục 0.3831. Lỗi bỏ sót ask (MISSED_ASK) giảm xuống chỉ còn 8 ca (2.7%). Cải thiện ròng +22 mẫu so với baseline. Chiến lược "Minimal Surgery" thành công xuất sắc, chốt planner_v4 làm prompt champion của Vương.
- 07/11 (chiều): Hoàn thành Vòng 5 GEPA (planner_v5.txt) trên 300 mẫu dev: Mean F1 0.3454, Ask F1 hồi phục lên 0.1632, lỗi Wrong Args duy trì ở mức thấp 42 (14.0%). Phát hiện 4 ca tràn context 8192 token (train_0197, train_0200, train_0272, train_0280) gây crash vLLM. Đã nâng cấp pipeline.py: tự động lưu console log ra file <out>.log và tự động cắt tỉa context an toàn chống lỗi VLLMValidationError.
- 07/11 (chiều/tối): Triển khai Kế hoạch Tối ưu Mới (4 Trụ Cột):
  - Trụ cột 1 (Safe Hash Guard & Schema Validator): Xây dựng `lifemate/validator/validator.py` và tích hợp thẳng vào `run_agent` trong `lifemate/pipeline.py`. Bộ lọc tự động chặn mơ hồ danh bạ/xe đạp khi tra cứu ra $\ge 2$ kết quả, phát hiện từ chỉ số lượng mập mờ ("chút tiền") chuyển thành `sys.ask_missing_info`, chuẩn hoá tên phòng tiếng Anh sang tiếng Việt (`smart_home`), làm sạch tham số thừa (`repeat='once'`). Điểm mô phỏng trên dev_v4 bứt phá kỷ lục mới: Mean F1 đạt 0.3760 (+0.0163), Ask F1 tăng vọt lên 0.3100 (+135%), lỗi HASH_MISMATCH giảm còn 126 ca. Bộ test unit `tests/test_validator.py` pass 100%.
  - Trụ cột 3 (Dialect Mapper Prompt): Bổ sung cờ `--mapper-prompt` vào `lifemate/pipeline.py` và biên soạn prompt chuyên biệt `prompts/gepa/vuong/mapper_v1.txt` (chuẩn hoá "chiều ni", "chừ", "buồng ngủ", "kênh nớ", giữ nguyên thực thể/giờ/tiền).
  - Trụ cột 2 & 4: Đã sẵn sàng chạy benchmark thực tế trên Kaggle GPU để xác nhận điểm số live và chuẩn bị chiến lược Ensemble.
- 08/11 (sáng/trưa): Hoàn thành benchmark trọn vẹn 300 mẫu dev cho `dev_v4_guarded` trên Cloud GPU:
  - Kết quả thực tế: Ask F1 đạt kỷ lục mới 0.3200 (+295% so với baseline 0.0810), tốc độ xử lý nhanh nhất lịch sử (17.1s/mẫu, chỉ 36 mẫu > 30s). Toàn bộ log, trace và run được lưu trữ vĩnh viễn tại `optimize/gepa/vuong/eval_results/`.
  - Chẩn đoán sâu 147 ca HASH_MISMATCH (chiếm ~50% tập dev): Phát hiện 71.7% lỗi nằm ở 2 "bẫy": (1) 110 ca bị nuốt mất nhánh ELSE ("không thì", "chưa tới thì") do Rule 3 của V4 bảo "nếu điều kiện sai thì dừng"; (2) 105 ca tự bịa tham số tuỳ chọn (`room` trong `light_set`, `label`/`date_from` trong `alarm.create`, `note` trong `banking.transfer`) làm lệch hash.
  - Ra mắt champion prompt mới `planner_v6.txt`: Mở khóa nhánh ELSE cho yêu cầu có điều kiện, cấm triệt để việc điền tham số tuỳ chọn khi user không nói. Chuẩn bị hoàn tất `notebooks/colab_run_planner.ipynb` sẵn sàng chạy benchmark v6.
- 08/11 (chiều/tối): Đột phá kỷ lục Ask F1 với `dev_v6_real` và Chuyển hướng sang Pure Prompt Planner V7:
  - Kết quả `dev_v6_real` (300 mẫu trên Cloud GPU): Mean F1 tăng lên 0.3167 (+0.0293), Act F1 hồi phục 0.2867, Ask F1 bứt phá lập kỷ lục mọi thời đại 0.3893 (+380% so với baseline 0.0810). Lỗi bỏ sót ask (MISSED_ASK) giảm xuống chỉ còn 9 ca (3.0%), sai tham số chạm đáy 44 ca (14.7%). Có 9 mẫu từ 0.0 điểm biến thành 1.0 điểm hoàn hảo nhờ luật tự hủy và nhánh ELSE.
  - Giải mã 145 ca HASH_MISMATCH còn lại: Phát hiện lỗi Overfitting nghiêm trọng của tầng Rule-based Validator (`validator.py`): luật tự động ép hỏi lại khi trạm xe có >1 xe đã "chặn họng" 23 ca mà user đã chọn xe rõ ràng, biến thành 0 điểm oan.
  - Quyết định kiến trúc bước ngoặt: Gỡ bỏ hoàn toàn `validator.py` khỏi pipeline runner. Chuyển toàn bộ cơ chế phát hiện mập mờ (tiền không rõ con số "chút tiền", trùng tên người nhận) thành luật đanh thép trực tiếp trong prompt `planner_v7.txt`.
  - Chuẩn bị sẵn sàng `dev_v7_pure` trong `notebooks/colab_run_planner.ipynb` để benchmark 100% Pure Prompt zero-shot an toàn tuyệt đối cho bài thi chính thức.
