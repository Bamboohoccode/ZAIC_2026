# lifemate/verifier (Hà)

Verifier kiểm plan của Planner trước khi chạm ToolEnv: so với action đã lưu (và, nếu nhóm chốt, với lời user), trả ĐÚNG hoặc SAI. SAI thì bỏ call hoặc hỏi user bằng `sys.ask_missing_info`. Xem PROJECT_OVERVIEW.md mục 5.3, luồng việc ở `task_flow/ha.md`.

## Vào, ra
- Vào: danh sách action đã chạy (`diff.actions_from_item` cho history của mẫu, `diff.actions_from_msgs` cho `msgs` của pipeline) và plan (list `{"name", "arguments"}`).
- Ra: `{"label": "CORRECT"|"WRONG", "diff": [...]}` (`diff.verdict`), gợi ý `{"drop": [...], "ask": (tool_name, params)}` (`diff.suggest`). `contracts.py` còn `Verdict` cũ (ACCEPT/REJECT/ASK), chờ Tiên đổi.

## File
| File | Việc | Cần dữ liệu BTC |
|---|---|---|
| `diff.py` | Verifier luật: ID bịa, lặp write/read, call sau ask, tool lạ, hỏi lại tham số đã biết. `to_text` chuỗi hoá input cho model | không |
| `analyze_errors.py` | Phân loại call sai của Planner: diff bắt được, hỏi thừa, cần query, thiếu call | full: có. partial: không |
| `oracle_ceiling.py` | Trần điểm của Verifier hoàn hảo (dùng gold), và điểm thật khi gắn Verifier (`--rules`, `--model`, `--scores`) theo từng ngưỡng | có |
| `data.py` | Sinh mẫu ĐÚNG/SAI từ train, có/không kèm query (có `add_arg`: thêm tham số thừa, lỗi thật nhiều nhất), thêm lỗi thật từ dự đoán (bỏ dev) | có |
| `train.py` | Fine-tune bộ phân loại, đổi model nền: mmBERT-base/small, XLM-R, mDeBERTa | không (cần file từ data.py) |
| `chay_verifier.bat` | Windows có GPU NVIDIA: kiểm GPU, chạy test, train, chấm Verifier thật trên dev (`lifemateerifier\chay_verifier.bat .. mmbert-small`) | có (chép tay, không commit) |
| `smoke_test_laya.py` | Kiểm Laya-multilingual có hiểu tiếng Việt, độ trễ | không |
| `tests/` | 23 test, chạy không cần pytest | không |

## Cách chạy
```bash
# test
python lifemate/verifier/tests/test_diff.py
python lifemate/verifier/tests/test_data_analyze.py

# phân tích lỗi (partial: chỉ cần file dự đoán + loại tool)
python lifemate/verifier/analyze_errors.py --pred dev_v4.jsonl --kinds test/results_train.jsonl
# phân tích lỗi (full, cần data/)
python lifemate/verifier/analyze_errors.py --root data/raw/train+public_test --split train \
    --pred dev_v4.jsonl --ids eval/splits/dev_ids.txt --out runs/errors_v4.json

# trần điểm Verifier hoàn hảo (cần data/)
python lifemate/verifier/oracle_ceiling.py --root data/raw/train+public_test --pred dev_v4.jsonl \
    --ids eval/splits/dev_ids.txt --out runs/oracle_v4.json
# điểm thật khi gắn Verifier: luật, hoặc model đã train (quét ngưỡng, bỏ call có P(ĐÚNG) < ngưỡng)
python lifemate/verifier/oracle_ceiling.py ... --rules
python lifemate/verifier/oracle_ceiling.py ... --model runs/ver_q_mmbert --with-query --save-scores runs/p_v4.jsonl

# sinh dữ liệu và train (Kaggle GPU)
python lifemate/verifier/data.py --root data/raw/train+public_test --out runs/ver.jsonl
python lifemate/verifier/data.py --root data/raw/train+public_test --with-query --out runs/ver_q.jsonl
python lifemate/verifier/data.py --root data/raw/train+public_test --with-query --neg-per-pos 2 --out runs/ver_q2.jsonl
python lifemate/verifier/train.py --data runs/ver_q.jsonl --backbone mmbert-base --out runs/ver_q_mmbert

# smoke-test Laya
pip install laya torch
python lifemate/verifier/smoke_test_laya.py --device cuda --out runs/laya_smoke.json
```

## Điểm hiện tại (07/11, dữ liệu BTC v3.2.0, dev 300 mẫu, dự đoán `dev_v4` của Vương, chế độ --text)
- `analyze_errors.py` full: 546 call dự đoán, 255 đúng, 291 sai/thừa. Trong 291 call sai: thêm tham số gold không có 101 (35%), sai tool hoặc thừa 92 (32%), sai giá trị 72 (25%), hỏi thừa 21 (7%), `diff.py` bắt được **1 (0,3%, UNSEEN_ID)**, báo nhầm 0. Planner bỏ sót 388/643 call gold (60%), Verifier không thêm call được.
- `data.py` trên train (1545 mẫu, dev = 300 mẫu của `eval/splits/dev_ids.txt`):
  - không query: train 4943 (ĐÚNG 2898) | dev 1063 (ĐÚNG 643)
  - có query: train 5701 (ĐÚNG 2898) | dev 1281 (ĐÚNG 643); `add_arg` 418 train / 63 dev
  - có query, `--neg-per-pos 2`: train 7894 | dev 1840
- `oracle_ceiling.py --rules` (diff.py làm Verifier): AUC trên 546 call thật 0,50, bỏ 1 call sai, điểm không đổi (0,3597).
- `oracle_ceiling.py` (Verifier đoán đúng 100%): gốc 0,3597 | SAI thì hỏi user 0,3063 (**-0,053**) | bỏ mọi call sai 0,3760 (+0,016) | bỏ có chọn lọc 0,4218 (+0,062). Thiết kế "SAI thì hỏi user" làm giảm điểm kể cả khi Verifier hoàn hảo; phần tăng chủ yếu ở mẫu `empty` và `ask`. Bỏ call sai làm giảm 25 mẫu act (call không khớp gold nhưng vẫn ra đúng state).

### Ghi chú trước đó

- Smoke-test Laya (CPU, bộ thử nhỏ): phân loại app có dấu 0,90 (tiếng Anh 1,00), có/không AUC vi 0,72, zero-shot bài Verifier AUC 0,61, độ trễ CPU p50 315 ms. Chưa đo GPU.
- Phân tích lỗi partial trên dự đoán dev của Vương (no_verify, v1, v4; 300 mẫu, 503 đến 546 call): **0 lỗi kiểu state-only** (lặp write, lặp read, call sau ask). Pipeline đã chặn sẵn ID bịa, đọc lặp, call sau ask (`pipeline.run_agent`). Nghĩa là Verifier không dùng query gần như không còn gì để bắt trên pipeline hiện tại; cần chạy chế độ full để xác nhận và đo phần "cần query".
- `train.py` đã chạy thông trên CPU với mmBERT-small và dữ liệu giả (140.642.306 tham số, p50 129 ms CPU). Điểm trên dữ liệu giả không có ý nghĩa.
