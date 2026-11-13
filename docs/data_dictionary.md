# Từ điển dữ liệu

Chỉ mô tả cấu trúc, không chép giá trị thật từ dữ liệu BTC (dữ liệu không được phát tán). Nguồn: README trong gói dữ liệu v3.2.0 và kiểm tra trực tiếp ngày 2026-11-07.

---

## 1. Dữ liệu thô (`data/raw/train+public_test/`)

### 1.1 `<split>_tools_catalog.json`: luật gọi tool, chung cho cả split
a) `schema_version`, `note`, `apps[]`.
b) Mỗi app: `id`, `name_vi`, `group`, `description`, `tools[]`.
c) Mỗi tool: `name` (dạng `app.tool`), `kind` (`read`, `write`, `meta`), `description`, `parameters` (JSON Schema), `returns`, `app`.
d) Phần mở rộng trong `parameters`:
   1) `x-one-of`: bắt buộc đủ một trong các nhóm tham số.
   2) `x-ref: "app.collection"`: giá trị là `id` lấy từ collection đó trong state.
   3) `x-match: fuzzy`: văn bản tự do, chấm gần đúng (token-F1 > 0,6).
e) Số đếm (2026-11-07): train 46 app, 147 tool (62 read, 84 write, 1 meta); public_test 36 app, 122 tool (52 read, 69 write, 1 meta).

### 1.2 `<split>.jsonl`: một dòng một mẫu
a) `id`, `audio` (WAV yêu cầu cuối).
b) `context`: `current_time`, `timezone`, `location`, `user_profile`, `installed_apps` (app và tool được phép gọi trong mẫu).
c) `history[]`: lượt `user` (`audio`, train có `audio_text`) và lượt `assistant` (`text`, `tool_calls`, `tool_results`).
d) Chỉ train: `audio_text`, `label.tool_calls`, `label.tool_results`, `label.final_state`.

### 1.3 `<split>_environment/<id>/state.json`: dữ liệu app của mẫu
a) `_meta`: `question_id`, `current_time`, `tools`.
b) Mỗi app một khối, bên trong là collection (`list`, `dict` hoặc giá trị đơn). Train có 93 collection (đếm 2026-11-07).
c) Agent không đọc trực tiếp tệp này; luật cấm. Dữ liệu chỉ xem qua tool đọc.

### 1.4 Khoá liên kết
a) `id` mẫu = tên thư mục `<split>_environment/<id>` = `_meta.question_id`.
b) Khoá trong `installed_apps` là `app.id` của catalog.
c) Tham số có `x-ref` lấy `id` từ `state[app][collection]`.

---

## 2. Dữ liệu suy ra (`data/derived/`, nhãn "derived")

### 2.1 `planner_train.jsonl`, `planner_dev.jsonl` (2026-11-07)
a) Trường: `id`, `split`, `query` (= `audio_text`), `label` (`act`, `ask`, `empty`), `tool_calls` (gold), `apps`.
b) Chưa có `context`, `history`, `tool_specs`.

### 2.2 `verifier_train.jsonl`, `verifier_dev.jsonl` (2026-11-07)
a) Trường: `id`, `plan`, `verdict` (`ACCEPT`, `REJECT`), `reason`, `perturb` (`drop_arg`, `wrong_value`, `wrong_tool`, `extra_write`), `query`.

### 2.3 `ask_v2/planner_ask_v2.jsonl` (2026-11-07)
a) Trường: `id`, `split` (`train_synth`), `source` (`btc_rewrite`, `opus_new`), `base_sample`, `query`, `label` (`ask_user`), `context`, `history`, `tool_calls` (một call `sys.ask_missing_info`), `apps`, `tool`, `missing_param`.

### 2.4 `TOOL_SURVEY.md`
a) Bảng tool theo app: kind, tham số bắt buộc, tuỳ chọn, enum, x-ref, fuzzy, x-one-of.

---

## 3. Định dạng nộp bài (README mục 4)

a) UTF-8 JSONL, đúng một dòng cho mỗi mẫu của split.
b) Trường: `id`, `tool_calls` (`[{"name", "arguments"}]`, gồm cả call bị từ chối), `state_hash`.

---

## 4. Bộ data nhiều lượt (derived/processed, 2026-11-11)

Tệp: `data/processed/sft_multiturn_{train,dev}_20261111.jsonl`, sinh bởi `scripts/build_sft_multiturn.py`.
a) `id`, `split` (train|dev), `source` ("btc"), `category` (act|ask|empty), `apps`, `n_calls`, `n_history`, `n_tokens`.
b) `messages`: role system|user|assistant|tool. Assistant có `tool_calls` [{id, type, function{name, arguments(dict)}}]; tool có `tool_call_id`, `name`, `content` (JSON chuỗi, kết quả thật BTC). `train`: true chỉ ở assistant của lượt cuối.
c) `tools`: schema function-calling của tool thuộc installed_apps, cộng `sys.ask_missing_info`.
