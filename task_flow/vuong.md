# [Vương] Luồng làm việc: Planner, state, validator, GEPA

Phạm vi (ClickUp): `lifemate/planner/`, `lifemate/state/`, `lifemate/validator/`, `optimize/gepa/run/`, `prompts/gepa/`.
Nhánh: `planner/vuong`. Hạn: đóng băng 15/11, nộp Docker 16/11 23:59.

## Quyết định đã chốt (05/11)
| # | Quyết định | Lý do |
|---|---|---|
| D1 | Lấy `pipeline.py` làm baseline. Tách planner, validation, history-trimming sang `lifemate/planner|state|validator` **giữ nguyên hành vi**, rồi cải tiến từng bước có đo | Code đã chạy được, tránh viết lại |
| D2 | Dữ liệu ở `data/` (đã gitignore). Replay gold trên 100 mẫu train: score 1.0, nghĩa là env và scorer OK | Kiểm tra môi trường |
| D3 | GPU: chạy Qwen3-4B bf16 + vLLM trên cloud (Kaggle/Colab/A30). Máy local (RTX 4060 8 GB) chỉ dùng cho code, unit test, replay gold | 8 GB không đủ cho 4B bf16 + 16K context |
| D4 | Dev set cố định khoảng 300 mẫu: giữ lại khoảng 20% app, giữ tỉ lệ act/ask/empty, seed cố định. Chỉ commit danh sách ID (`eval/splits/dev_ids.txt`), chia sẻ với Kiệt | Public test có 11 app không có trong train |
| D5 | Thí nghiệm planner chạy `--text` (dùng audio_text). Trước khi đóng băng chạy kiểm tra end-to-end với ASR + mapper | Cô lập module |
| D6 | GEPA: chạy tay trong chat trước (agent đóng vai reflection LM, sửa có đo). Sau 12/11 nếu còn thời gian và vòng tay hết cải thiện thì chuyển sang `dspy.GEPA` với DeepSeek làm reflection LM (giới hạn khoảng 5 USD) | Miễn phí, bắt đầu được ngay |

Số liệu train: 1.545 mẫu (act 1.217, ask 217, empty 111), trung bình 2,3 call mỗi mẫu, 1.018 mẫu có history. 46 app train và 36 app public, chung 25 app.

## Luồng (mỗi bước là 1 subtask trong ClickUp "[Vương] Start luồng")
| # | Bước | Hạn | Điều kiện xong |
|---|---|---|---|
| 1 | Tạo `eval/splits/dev_ids.txt` (chia theo app) + script chạy dev + báo điểm theo act/ask/empty | 06/11 | **[XONG 06/11]** `dev_ids.txt` (300 mẫu, 9 held-out apps), `scorer.py` (báo điểm tổng + act/ask/empty + diagnostics), `pipeline.py --ids` chạy chuẩn |
| 2 | Baseline trên GPU cloud: `pipeline.py --text` trên dev, lưu trace, đo thời gian mỗi mẫu | 07/11 | Có điểm baseline + latency, ghi vào `docs/DECISIONS.md` |
| 3 | Phân nhóm lỗi (script): sai tool, sai tham số, read thừa, thiếu ask, ask thay vì làm, phải rỗng, lệch hash | 07/11 | Bảng % lỗi theo nhóm |
| 4 | Refactor: `lifemate/planner` (prompt, `plan_step`, `run_agent`), `lifemate/state` (`build_messages`, cắt history), `lifemate/validator` (`env.validate` + `unseen_ids`). Điểm trước và sau phải bằng nhau | 08/11 | Unit test pass, dev score không đổi |
| 5 | Validator luật: thiếu tham số bắt buộc thì chuyển sang `sys.ask_missing_info`; x-one-of; enum; định dạng ISO time và tiền nguyên; chặn write trùng | 09/11 | Dev score tăng hoặc giữ nguyên, tỉ lệ call lỗi schema gần 0 |
| 6 | Ablation self-verifier `--verify-rounds 0/1/2`: điểm và latency | 10/11 | Chọn N, ghi DECISIONS |
| 7 | State module: tóm tắt history có cấu trúc (slot đã biết, ID đã đọc, action đã làm), giảm token và read lặp | 11/11 | Dev score không giảm, số token giảm |
| 8 | Vòng GEPA tay (3 đến 5 vòng mỗi ngày): đọc failure trace, sửa một chỗ, giữ nếu dev tăng. Theo dõi riêng nhóm act để tránh "hỏi lại cho an toàn" | 12/11 đến 14/11 | Prompt chốt trong `prompts/gepa/` có version + điểm |
| 9 | (Tuỳ chọn) `dspy.GEPA` + DeepSeek nếu vòng tay hết cải thiện | 12/11 đến 14/11 | Chỉ nhận khi dev tăng |
| 10 | Kiểm tra end-to-end với ASR + mapper, latency ≤ 30 s/mẫu trên GPU giống A30, seed/greedy | 14/11 | Bàn giao cho Tiến đóng Docker |
| 11 | Đóng băng: tag `freeze-15-11`, chỉ sửa lỗi | 15/11 | |

## Cần trao đổi
- Tiến: ai sở hữu `pipeline.py` sau refactor (Vương sửa planner/state/validator, Tiến giữ phần glue/ASR/Docker)?
- Kiệt: dùng chung `dev_ids.txt`, scorer hiện tại là `test_pipeline.py::score`.
- Hà: verifier Laya so với self-verify bằng Qwen; kết quả bước 6 dùng làm baseline để so.
- Tài liệu `Overview_PJ.md` lệch với code (PhoWhisper/Laya so với ChunkFormer/Qwen3-0.6B/self-verify). Cần cập nhật.

## Nhật ký
- 05/11: Đọc đề + repo, chốt D1 đến D6, chuyển dữ liệu vào `data/`, replay gold 100 mẫu = 1.0.
- 06/11: Xong Subtask 1: sinh `eval/splits/dev_ids.txt` (300 mẫu, giữ 9 app lạ), cập nhật `eval/scorer.py` hỗ trợ chẩn đoán chi tiết theo `act`/`ask`/`empty`, thêm cờ `--ids` vào `lifemate/pipeline.py`. Đồng bộ sạch sẽ với `update_3_flow` của Tiên.
