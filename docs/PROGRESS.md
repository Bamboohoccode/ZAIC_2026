# PROGRESS: tiến độ phần việc của Tiên

Quy tắc ghi theo `CLAUDE.md` mục 2: chỉ đánh dấu xong khi có đầu ra kiểm chứng được, mọi con số truy được về script hoặc tệp đầu ra, số đo ghi kèm ngày đo.
Kế hoạch chi tiết: `own_flow/plan_paramater.md` (Task 1), `own_flow/plan_data_tool.md` (Task 2), `task_flow/tien.md`.
Hạn cá nhân: 2026-11-18 08:00. Hạn nhóm: freeze 2026-11-15, nộp Docker 2026-11-16 23:59.

---

## A. Chuẩn bị chung

- [x] A1. Giải nén dữ liệu BTC (2026-11-07)
  a) Nguồn: `data/train+public_test.zip`, SHA-256 trong `data/manifest/raw_sha256_20261107.txt`.
  b) Đích: `data/raw/train+public_test/` (bỏ `__MACOSX/` và `audio/`, nên chưa có WAV trên máy).
  c) Số đếm khớp tệp VERSION: train 1545, public_test 634.
- [x] A2. Kiểm ToolEnv chạy được (2026-11-07)
  a) Liệt kê app: train 46 app / 147 tool, public_test 36 app / 122 tool, 0 tool chưa cài đặt (lệnh `python -m zaic_tools --apps`).
  b) Gọi thử mọi tool: 147/147 và 122/122 trả `ok` (script `test/call_all_tools.py`, đầu ra `runs/call_all_tools/`; `--validate` báo 0 lỗi schema).
- [x] A3. Kiểm chuỗi chấm điểm của nhóm (2026-11-07)
  a) `lifemate/pipeline.py --llm gold --mapper none --text --ids eval/splits/dev_ids.txt`, đầu ra `runs/gold_dev.jsonl` (300 dòng).
  b) `eval/scorer.py` trên 300 mẫu dev: Mean 1,0000 (act 1,0000 n=236; ask 1,0000 n=42; empty 1,0000 n=22).
  c) Ý nghĩa: harness (pipeline, ToolEnv, scorer) dùng được cho baseline và WSP.
- [x] A4. So `state` sau replay gold với `label.final_state` từng mẫu (2026-11-11). 1544/1544 mẫu (train 1244 + dev 300) khớp `tool_results` và `final_state`, 0 call bị từ chối. Nguồn: `reports/sft_multiturn_20261111.md` mục 2. (1 mẫu trùng `audio_text` bị loại trước khi kiểm.)
- [x] A5. Tải Qwen3-4B (2026-11-07, người dùng đồng ý tải)
  a) Repo `Qwen/Qwen3-4B` (bản hybrid, không phải Instruct-2507), vào `Zalo_Ai_Challenge/models/Qwen3-4B/` ngoài repo, 3 shard, khoảng 7,6 GB.
  b) Chưa load thử, chưa kiểm checksum từng shard. CHƯA XÁC MINH tính nguyên vẹn.

## B. Task 2: dữ liệu call tool

- [x] B1. Khảo sát tool (2026-11-07): `data/derived/TOOL_SURVEY.md`.
- [x] B2. Planner train/dev từ nhãn BTC (2026-11-07)
  a) `data/derived/planner_train.jsonl` 1244 mẫu (act 980, ask 175, empty 89), `planner_dev.jsonl` 300 mẫu (act 236, ask 42, empty 22).
  b) Dev theo `eval/splits/dev_ids.txt` của nhóm. Đã loại 1 mẫu train trùng `audio_text`.
  c) Thiếu `context`, `history`, `tool_specs`. Cần bổ sung (B6).
- [x] B3. Verifier train/dev (2026-11-07)
  a) `verifier_train.jsonl`: ACCEPT 980, REJECT 495. `verifier_dev.jsonl`: ACCEPT 236, REJECT 89.
  b) Mỗi REJECT được kiểm bằng ToolEnv: hash state cuối khác gold.
  c) Seed dùng 0, chưa theo quy tắc seed 42.
- [x] B4. Nhãn ask_user v2 (2026-11-07)
  a) Phân tích 217 mẫu ask của train: luôn hỏi đúng 1 tham số, ask là call cuối; 131 mẫu hỏi tham số x-ref chỉ sau khi tra cứu ra từ 2 kết quả; 86 mẫu hỏi giá trị người dùng chưa nói.
  b) `data/derived/ask_v2/planner_ask_v2.jsonl`: 65 mẫu (14 viết lại từ mẫu act một ý của BTC, 51 câu mới). Script `test/build_ask_v2.py`.
  c) Kiểm ToolEnv 65/65 đạt, không trùng câu train, không dùng app chỉ có trong dev.
  d) Bản v1 (`planner_ask_template.jsonl`, 143 mẫu) đã xoá vì nhãn sai (hỏi thẳng tham số x-ref, bỏ nhiều tham số nhưng nhãn chỉ hỏi 1). Xoá khi chưa hỏi người dùng, trái `CLAUDE.md` mục 6a.
- [ ] B5. Nhãn cannot_answer sinh bằng code (bỏ app khỏi `installed_apps`). CHƯA LÀM.
- [x] B6. Bổ sung `context`, `history`, `tool_specs` cho planner (2026-11-11, người dùng chọn hội thoại nhiều lượt, chỉ từ train BTC).
  a) Script `scripts/build_sft_multiturn.py`; đầu ra `data/processed/sft_multiturn_train_20261111.jsonl` (1244: act 980, ask 175, empty 89) và `sft_multiturn_dev_20261111.jsonl` (300: act 236, ask 42, empty 22). SHA-256 trong `data/manifest/sft_multiturn_sha256_20261111.txt`.
  b) Mỗi dòng: `messages` (system có context, history, user cuối, rồi assistant tool_call, tool result thật BTC xen kẽ; cờ `train` gợi ý mask loss) và `tools` (schema theo installed_apps). Không cắt `tool_results`.
  c) Độ dài token (tokenizer Qwen3.8-4B-Distilled, kèm tool schema): trung vị 4891, p99 8459, max 9384; 23 mẫu > 8192. Cột `n_tokens` để lọc.
  d) Dựng thử chat template 1 mẫu ask có history: render đúng `<tool_call>` và `<tool_response>`. CHƯA XÁC MINH: trainer cụ thể chấp nhận `arguments` dạng dict; nhãn BTC không có câu trả lời bằng lời sau call cuối.
- [ ] B7. Cân bằng ACCEPT/REJECT về khoảng 1:1. CHƯA LÀM.
- [ ] B8. Xuất bản v2 có ngày trong tên tệp, seed 42. CHƯA LÀM.

## C. Task 1: tham số Qwen3-4B

- [ ] C1. Chạy Qwen3-4B trên Kaggle (VRAM, giây/lượt). CHƯA LÀM, cần Kaggle.
- [x] C2. Thêm cờ `--ids` cho `lifemate/pipeline.py` (2026-11-07). Kiểm: gold trên `dev_ids.txt` ra 300 dòng, scorer 1,0000. Vương ghi đã thêm cờ cùng tên ở nhánh riêng: cần đối chiếu khi gộp.
- [x] C2b. Thêm cờ `--llm-think on|off` và `--llm-dtype` cho `pipeline.py` (2026-11-07). Mặc định giữ hành vi cũ. Kiểm: gold replay chạy được; chạy thật với vLLM CHƯA XÁC MINH.
- [ ] C3. Baseline Qwen3-4B trên tập đo, cả `--think on` và `off`. CHƯA LÀM, cần Kaggle.
- [x] C4. Công cụ phân nhóm lỗi `optimize/wsp/analyze.py` (2026-11-07). Kiểm trên bản chạy giả: bắt đúng thinking bị cắt, lặp, sai định dạng, lỗi pipeline, call thừa. Chạy trên baseline thật: CHƯA LÀM.
- [~] C5. LoRA SFT: BỎ (người dùng quyết định 2026-11-07, chỉ làm WSP).
- [x] C6. Mã WSP trong `optimize/wsp/` (2026-11-07): `weights.py`, `make_subset.py`, `analyze.py`, `directions.py`, `run_round.py`, `README.md`.
  a) Kernel kiểm trên tensor thật: Qwen3-4B 4.022.468.096 tham số; sửa 64 cột tầng 35 đổi 158.541 phần tử (0,0039%); lưu và đọc lại đúng, model gốc không đổi.
  b) Tập đo `dev_subset50_20261107.txt` (act 39, ask 7, empty 4); tập practice `practice100_20261107.txt` (train, giao dev = 0).
  c) Luồng `run_round.py --llm-gold` chạy hết vòng 0 và vòng 1 có sửa.
  d) Mã gói cho Kaggle: `Zalo_Ai_Challenge/kaggle_upload/zaic_code_20261107.zip` (51 tệp, không có `data/` hay `.jsonl`).
- [x] C6a. Gói chạy Kaggle (2026-11-07, người dùng chọn chạy trên Kaggle thay vì local):
  a) `kaggle_upload/zaic_data_noaudio_20261107.zip` (6,3 MB, 2.286 tệp, không audio), `kaggle_upload/zaic_code_20261107.zip` (52 tệp).
  b) Notebook `kaggle_upload/wsp_step1_2_20261107.ipynb` (bản sao `optimize/wsp/kaggle_wsp_step1_2.ipynb`, sinh bởi `optimize/wsp/make_kaggle_notebook.py`). Model cố định snapshot `1cfa9a72...`.
  c) Kiểm trên máy: cú pháp mọi ô; giả lập thư mục Kaggle (zip chưa giải nén và đã giải nén), chạy một vòng `--llm-gold` thành công ở cả hai.
  d) Hướng dẫn: `kaggle_upload/HUONG_DAN.md`.
- [ ] C6b. Chạy vòng sửa trên Kaggle (10, 25, 50 vòng). CHƯA LÀM: cần người dùng tải dữ liệu và mã lên Kaggle, rồi gửi lại `r00.summary.json` và `dirs.stats.json`.
- [x] C6c. Bước 0.2 kiểm tràn fp16 trên trọng số (2026-11-07). Script: scratchpad `fp16_check.py` (không nằm trong repo).
  a) 72 tensor `mlp.down_proj` + `mlp.up_proj` (36 tầng), bf16, 1.793.064.960 phần tử.
  b) |x| > 65504: 0. Thành inf sau fp16: 0. Max |x| toàn bộ: 1,234375 (tầng 16).
  c) Khác 0 nhưng về 0 khi sang fp16 (underflow): 2.418 phần tử, khoảng 0,00013% số phần tử khác 0. Rải đều các tầng, không đáng kể.
  d) Kết luận: không tràn ở trọng số. Chưa kiểm activation: tràn fp16 ở activation (residual stream) vẫn có thể xảy ra, phải xem ở bước 1 (NaN/inf trong logits, điểm khác bất thường). CHƯA XÁC MINH.
- [~] C6d. Bước 0.4 đóng gói lại mã (2026-11-07). Zip cũ `zaic_code_20261107.zip` (18:14) đã lệch repo: `run_round.py`, `directions.py`, `lifemate/pipeline.py`. Đã tạo `kaggle_upload/zaic_code_20261107_v2.zip` (51 tệp, khớp repo từng byte, bỏ `docs/PIPELINE_PROPOSAL.md` đã xoá). Chưa thay zip cũ: lệnh ghi đè bị từ chối. Cần người dùng quyết định xoá zip cũ và đổi tên v2, hoặc cho phép ghi đè. Kaggle cần tải zip v2 (hoặc zip cũ đã thay).
- [x] C6e. Công cụ tự sửa trọng số chạy trên Kaggle, không gọi API (2026-11-07, Opus 5.5 viết):
  a) `optimize/wsp/batch_agent.py`: chạy agent theo lô (mọi mẫu song song, vLLM generate), chia mẫu cho 2 GPU T4. Kiểm: `test_batch_agent.py` so với `pipeline.main` bằng LLM giả trên 150 mẫu train: submission và trace giống từng byte (284 verify, 123 REJECT, 16 mẫu lỗi giữa chừng).
  b) `optimize/wsp/auto_search.py`: chính sách tìm bản sửa (pha A từng tầng, B ghép tầng, C số cột/alpha), luật giữ cố định. Kiểm: `--dry-run` 12 vòng đúng luật; chạy tiếp sau khi ngắt không chạy lại vòng cũ.
  c) `run_round.py`: thêm `--runner batch|seq`, `--gpus`, `--model-out`, ghi `gen_tokens_mean`. `directions.py`: bỏ mẫu có NaN/inf (fp16).
  d) Giả lập luồng notebook trên máy bằng replay nhãn (`--llm-gold`, 2 phần): baseline dev 10 mẫu, practice 20 mẫu, 2 vòng sửa thật trên trọng số (159.177 và 159.305 phần tử), gộp kết quả đủ 20 mẫu, model tạm tự xoá. Thư mục `runs/wsp_nbsim/`.
  e) Gói Kaggle: `kaggle_upload/zaic_code_20261107_v3.zip` (54 tệp; mọi tệp mã khớp repo từng byte; 3 tệp tài liệu `docs/DECISIONS.md`, `docs/PROGRESS.md`, `optimize/wsp/README.md` trong zip là bản trước lần ghi chép này, không ảnh hưởng khi chạy), notebook `kaggle_upload/wsp_step1_2_20261107.ipynb` (13 ô, cú pháp đạt), `kaggle_upload/HUONG_DAN.md`.
  f) CHƯA XÁC MINH: vLLM chạy trên T4; tốc độ mỗi vòng; prompt dựng bằng `apply_chat_template` + `generate` có giống hệt `vLLM chat` không (mọi vòng, kể cả baseline, dùng cùng bộ chạy nên vẫn so được).
- [ ] C6f. Chạy notebook trên Kaggle: lần 1 `QUICK = True` (đo tốc độ), lần 2 đầy đủ. CHƯA LÀM, cần người dùng.
  a) Lần 1 QUICK bằng notebook cũ, trước khi sửa ô 12 (2026-11-08, `wsp_results.zip`, giải nén tại scratchpad `wsp_results_x/`). Chỉ 10 mẫu dev, 20 mẫu practice, không kết luận được. Lần chạy notebook mới chưa có kết quả.
  b) Dev, think on: mean 0,3000 (n=10), gen_tokens TB 763,4, think_truncated 5/10. Think off: mean 0,3238, gen_tokens TB 184,6. Nguồn: `dev_on/r00.summary.json`, `dev_off/r00.summary.json`.
  c) Practice, think on: mean 0,0500 (n=20), think_truncated 15/20, HASH_MISMATCH 18/20. Nhãn xuất ra: bad 15, good 1 (`search/r00.export.jsonl`).
  d) Ô 7 không đủ mẫu `good` (cần >= 2), nên không tạo `dirs.pt`, không chạy tìm hướng và không có bản sửa nào. Lỗi ô 12 là hệ quả.
  e) Thời gian: `run_seconds` 511 s cho 10 mẫu dev (think on). `avg_seconds` 227,6 mỗi mẫu theo script đo, khác xa trần 30 giây. CHƯA XÁC MINH script đo có khớp cách đo độ trễ mục 7.3 không.
  f) Việc cần làm: chạy lại với `QUICK = False` để practice đủ 100 mẫu; thử có đủ `good` không. Cân nhắc đổi `MAX_NEW` hoặc cờ think vì 75% practice bị cắt ở 512 token.
- [x] C6g. Tách notebook thành 2 lần Commit, model mới (2026-11-09, DECISIONS 09/11):
  a) `optimize/wsp/make_kaggle_notebook.py --out-dir ../kaggle_upload` sinh `wsp_A_baseline_20261109.ipynb` (9 ô) và `wsp_B_search_20261109.ipynb` (12 ô). Model `Ma7ee7/Qwen3.8_4B_Distilled@f33da883`.
  b) A: baseline dev on+off, practice, tìm hướng; ghi `run_config.json`; đóng gói sau mỗi bước; thông báo phân biệt thiếu bad và thiếu good. B: nạp output A (thư mục hoặc zip), kiểm cùng model, dùng lại vòng 0, `auto_search.py --deadline-ts` (11 giờ trừ 3,5 giờ cho bước 5), kiểm dev, ô tay bỏ qua khi spec rỗng.
  c) `auto_search.py --dry-run`: không hạn 10 vòng; hạn đã qua thì 0 vòng, giữ baseline.
  d) Giả lập A rồi B trên máy (replay nhãn, script scratchpad `nbsim.py`): A in "THIẾU MẪU BAD" như dự kiến (gold toàn good); B nạp output A, không chạy lại vòng 0, 2 vòng sửa thật trên trọng số model mới (159.085 và 159.149 phần tử, 0,0040%); lần hạn giờ đã qua chỉ có vòng 0. Bước 5 (có `best_spec`) chưa giả lập được vì replay nhãn luôn 1,0.
  e) Gói `kaggle_upload/zaic_code_20261109_v4.zip` (54 tệp, theo danh sách v3; `test/` đổi thành `test_data/` theo repo). `HUONG_DAN.md` viết lại cho luồng A/B.
  f) CHƯA XÁC MINH: vLLM nạp model mới trên T4; `think off` có tắt thinking ở model gốc Thinking-2507 không; tốc độ; Kaggle có giữ Output khi Commit bị dừng không.
  g) Cập nhật 11/11: A/B được thay bằng một notebook (xem C6h). Mục a)-e) ở trên không còn đúng với notebook hiện tại.
- [x] C6h. Kết quả đợt đầu trên Kaggle, notebook cũ `wsp_step1_2_20261107.ipynb` (đo 09/11/2026, ghi 11/11/2026). Chi tiết và nguồn: `reports/wsp_run1_20261109.md`.
  a) dev think on và off đều mean 0,4694 (n=10), đầu ra giống hệt nhau: cờ `think off` không có tác dụng với setup này.
  b) practice mean 0,1792 (n=20): 14 mẫu HASH_MISMATCH. Export có 2 mẫu good, 0 bad, nên không chạy vòng sửa nào (`n_changed = 0`).
  c) Output Kaggle tải về được đủ. Tốc độ QUICK: dev 10 mẫu khoảng 7–9 phút, practice 20 mẫu khoảng 32 phút (CHƯA XÁC MINH cho bản đầy đủ).
- [x] C6i. Gộp A/B thành một notebook (2026-11-11, người dùng yêu cầu): `optimize/wsp/make_kaggle_notebook.py` sinh `kaggle_upload/wsp_single_20261111.ipynb` (12 ô). Bỏ baseline think off. Chưa chạy trên Kaggle.
  a) Tham số đổi và lý do: xem `reports/wsp_run1_20261109.md` mục 3.
  b) `HUONG_DAN.md` còn mô tả A/B, CHƯA CẬP NHẬT.
  c) Hai notebook A/B cũ còn trong `kaggle_upload/`, chờ người dùng xác nhận xoá.
- [x] C6j. Notebook nhanh để có bản sửa (2026-11-11, người dùng yêu cầu): `kaggle_upload/wsp_fast_20261111.ipynb` (10 ô) và `kaggle_upload/zaic_code_20261111_v5.zip` (55 tệp). Lý do và bảng tham số: `reports/wsp_run1_20261109.md` mục 5 và 6.
  a) `analyze.py` và `run_round.py` thêm `--label-rule score`; mặc định vẫn là luật cũ.
  b) Đã kiểm trên máy: luật nhãn mới với replay nhãn (bad 9, good 1 khi làm hỏng 9 mẫu); `auto_search.py --dry-run` chạy đủ 3 vòng pha A.
  c) CHƯA XÁC MINH: chạy trên Kaggle (thời gian, `directions.py` trên GPU, có bản nào được giữ không).
  d) `wsp_single_20261111.ipynb` sinh lại theo v5, vẫn dùng luật nhãn cũ.
- [x] C6k. Sửa OOM `directions.py` (2026-11-11): lần chạy notebook nhanh dừng ở `directions.py` do prompt 5.889 đến 8.491 token. Sửa bằng forward theo đoạn có KV cache, gói `kaggle_upload/zaic_code_20261111_v6.zip` (55 tệp). Chi tiết: `reports/wsp_run1_20261109.md` mục 7.
  a) Kiểm trên máy: trạng thái ẩn theo đoạn khớp forward đầy đủ (sai khác tối đa 9e-8, model nhỏ trên CPU).
  b) CHƯA XÁC MINH: chạy `directions.py` mới trên T4, ô 7 (`auto_search.py`) và ô 8.
- [x] C6l. Chạy notebook nhanh v6 trên Kaggle (2026-11-11): luồng WSP chạy hết bước tìm, ra `best_spec` (tầng 0, 64 cột, alpha 1,0; practice 10 mẫu 0,3583 -> 0,5250). Chi tiết và giới hạn: `reports/wsp_run1_20261109.md` mục 8.
  a) Chưa phải bằng chứng cải thiện: chọn và đo trên cùng 10 mẫu, 7 mẫu dùng tính hướng.
  b) Kiểm dev 10 mẫu (11/11): gốc 0,4694, WSP 0,4694 (không đổi), gen_tokens 369,0 lên 404,7 (+9,7%, vượt ngưỡng 105%). Bản sửa không cải thiện trên dev. Chi tiết: `reports/wsp_run1_20261109.md` mục 9.
- [ ] C7. Go/no-go WSP (2026-11-13), tiêu chí ở `optimize/wsp/README.md` mục 5.

## D. Dọn tệp (2026-11-07, người dùng cho phép gộp hoặc xoá tệp thừa)

a) Xoá `runs/gold_train.jsonl` và trace: thay bằng `runs/gold_dev.jsonl`, sinh lại được.
b) Chuyển `test/results_*.jsonl`, `test/submission_*.jsonl` sang `runs/call_all_tools/` (gitignore). Script đã đổi đường ghi.
c) Chuyển `test/check_data.md` sang `data/derived/check_data.md` (gitignore) vì chứa giá trị thật từ dữ liệu BTC. `test/inventory_train.py` giờ ghi vào tệp này (chạy lại sẽ ghi đè phần sửa tay mục 5).
d) `test/results_*.jsonl` đã từng được commit vào git. Cần commit việc xoá và kiểm xem đã push chưa. CHƯA XỬ LÝ (không tự commit).

## E. Việc tiếp theo

### Hiện trạng (cập nhật 2026-11-08)

Đã xong trên máy:
- Kiểm tràn fp16 trên trọng số: 0 tràn, 0 inf (C6c).
- Bộ chạy WSP theo lô, kiểm khớp từng byte với pipeline gốc (C6e).
- Notebook 13 ô, HUONG_DAN đã cập nhật, zip code `_v3` (54 tệp, khớp repo).

Đã làm thêm (2026-11-08):
- Người dùng duyệt DECISIONS k và l. Ghi vào DECISIONS (08/11).
- Người dùng chọn Qwen3-4B (`Qwen/Qwen3-4B`) cho WSP. `lifemate/pipeline.py` `LLM_MODEL` đã đổi sang bản này (dòng 23). Zip `_v3` chưa có thay đổi này (xem DECISIONS 08/11).
- Lỗi ô 12 (vòng sửa tay) trên Kaggle lần chạy QUICK (08/11): `dirs.pt` không có vì ô 7 không đủ mẫu bad/good (ô 8 in `best_spec: []`), nên `run_round.py` lỗi FileNotFoundError. Ô 12 vẫn có spec ví dụ `VÍ DỤ`. Sửa trong `make_kaggle_notebook.py`: spec rỗng thì dừng, không có `dirs.pt` thì dừng, rồi sinh lại `kaggle_upload/wsp_step1_2_20261107.ipynb` (13 ô). Chưa biết lý do ô 7 không đủ mẫu: cần đầu ra ô 7 của lần chạy.
- Xoá `kaggle_upload/zaic_code_20261107_new.zip` (08/11, người dùng đồng ý). Bản trung gian 9 tệp, không dùng ở notebook hay HUONG_DAN. Hai zip cũ `zaic_code_20261107.zip` và `_v2.zip` đã không còn trong thư mục.

Chưa làm:
- Chạy notebook trên Kaggle. Chưa có số đo nào trên GPU (C1, C3, C6f).
- A4, B5–B8: làm trên máy được, chưa làm.

Việc tiếp theo, theo thứ tự:
a) Người dùng: tạo dataset Private từ `zaic_code_20261107_v3.zip` và `zaic_data_noaudio_20261107.zip`. Import `wsp_step1_2_20261107.ipynb`, chọn GPU T4 x2, Internet On. Chạy lần 1 với `QUICK = True` và gửi lại đầu ra cùng `wsp_results.zip`.
b) Người dùng: duyệt hoặc đổi DECISIONS k và l trước khi chạy lần 2.
c) Opus: đọc kết quả lần 1, chọn `BUDGET`, hướng dẫn lần 2 (`QUICK = False`, Save & Run All).
d) Người dùng: xoá hai zip cũ trong `kaggle_upload/`, hoặc cho phép mình xoá.
e) Go/no-go ngày 13/11 (C7).
