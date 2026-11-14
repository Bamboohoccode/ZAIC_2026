# Kế Hoạch Triển Khai DPO Hai Giai Đoạn Cho `empero-ai/Qwen3.8-4B-Distill` (`Task 2`)

> Tài liệu kế hoạch chính thức được lưu trực tiếp trong repository (`optimize/dpo/PLAN.md`) để toàn bộ thành viên theo dõi, tái lập và chạy huấn luyện thống nhất.

---

## 1. Mục Tiêu & Bối Cảnh

- **Bài toán (Task 2):** Bổ sung tri thức phân biệt tool đúng/sai, tham số bắt buộc/tùy chọn, trình tự đọc-trước-ghi và cổng chốt chặn mơ hồ (`sys.ask_missing_info`) cho model Planner **`empero-ai/Qwen3.8-4B-Distill`** (Qwen 3.8 4B trên kiến trúc `Qwen3.5-4B` / `qwen3_5`).
- **Đặc điểm kiến trúc & chat template của `empero-ai/Qwen3.8-4B-Distill`:**
  - Kiến trúc lai Gated DeltaNet (`24` lớp `linear_attention` + `8` lớp `full_attention`, tổng `32` lớp). `AutoModelForCausalLM.from_pretrained` nạp `Qwen3_5ForCausalLM` phần text sạch sẽ; trên Colab GPU có thể cài thêm `flash-linear-attention` và `causal-conv1d` để tăng tốc `linear_attention` (nếu thiếu sẽ tự fallback sang PyTorch ops chuẩn).
  - Prompt sau `apply_chat_template(..., add_generation_prompt=True)` kết thúc sẵn bằng `<|im_start|>assistant\n<think>\n`; `train_dpo.py` tự nhận diện `prompt_opens_think` để đóng `</think>\n\n` sau phần suy luận `BƯỚC 1/2/3` mà **không tạo thẻ `<think>` kép**.
  - `lifemate/pipeline.py` (`parse_calls`) đã hỗ trợ lọc khối `<think>...</think>` (kể cả phần trước `</think>` khi `<think>` nằm ở prompt) và parse song song cả JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>` lẫn Qwen3.5 XML `<tool_call><function=...><parameter=...></parameter></function></tool_call>`.
- **Vấn đề hiện tại:**
  - Prompt + Tool Schema đã rất dài (~5.9K–8.5K tokens, max ~9.4K tokens). Tiếp tục nhồi thêm quy tắc bằng chữ vào prompt dễ gây loãng chú ý và tăng chi phí suy luận (ràng buộc thi đấu BTC: A30 24GB, <= 30s/mẫu).
  - Trên tập `dev_ids.txt` (300 mẫu), `dev_v15` baseline ghi nhận các nhóm lỗi lớn nhất là `HASH_MISMATCH` (139/300 = 46.3%) và `WRONG_TOOL_OR_ARGS` (44/300 = 14.7%).
- **Giải pháp cốt lõi:** **Pure LoRA DPO** trên cặp `(prompt, chosen, rejected)` giữ nguyên hoàn toàn system prompt V15 (`prompts/gepa/vuong/planner_v15.txt` / `lifemate.pipeline.SYSTEM`) và JSON Schema của `zaic_tools`.

---

## 2. Phân Chia Dữ Liệu Không Rò Rỉ (0% App & Text Leakage)

Toàn bộ 1,545 mẫu từ `data/train/train.jsonl` được tách cố định tại [eval/splits/](../../eval/splits/README.md):

| Tập | Đường dẫn | Số mẫu | App giữ riêng hoàn toàn | Tỉ lệ `act / ask / empty` | Vai trò |
|---|---|---:|---|---|---|
| **Val (Dev)** | `eval/splits/dev_ids.txt` | **300** | **9 app lạ**: `bike_rental`, `calculator`, `exchange_rate`, `fitness`, `lottery`, `package_tracking`, `fuel_price`, `gold_price`, `tv` (228 mẫu) + 72 mẫu stratified | 236 / 42 / 22 | Đo cải thiện Trước vs Sau theo từng app & chọn checkpoint tốt nhất |
| **Test** | `eval/splits/test_ids.txt` | **100** | **3 app lạ**: `library` (24), `home_energy` (27), `medication` (27) + 22 mẫu stratified | 79 / 14 / 7 | Kiểm thử tổng quát hóa sang app chưa từng thấy |
| **DPO Train** | `eval/splits/dpo_train_ids.txt` | **1,145** | 34 app còn lại (0 giao cắt với Val/Test) | 902 / 161 / 82 | Nguồn sinh cặp DPO Giai đoạn 1 & lấy mẫu On-Policy Giai đoạn 2 |

---

## 3. Chiến Lược DPO Hai Giai Đoạn (Two-Stage Pipeline)

### Giai đoạn 1 (Chạy trước): Programmatic ToolEnv-Verified DPO
Sử dụng bộ dữ liệu đã được sinh và kiểm chứng `gold_hash_verified=1144, gold_hash_failed=0` bằng `scripts/build_dpo_dataset.py`:
- `optimize/dpo/dataset/dpo_train.jsonl`: **4,798 cặp** (1,087 mẫu train, 49.672 MB).
- `optimize/dpo/dataset/dpo_val.jsonl`: **248 cặp** (57 mẫu nội bộ từ `dpo_train_ids.txt`, 2.53 MB).
- `optimize/dpo/dataset/tool_specs_catalog.json`: Bảng JSON Schema 147 tools.
- Phân bố 5 nhóm lỗi hard negative (`optimize/dpo/dataset/dpo_stats.json`):
  - `ambiguity_gate_error`: **1,712** (Missed Ask chọn đại ID / False Ask trên write & read)
  - `multi_action_or_scope_error`: **1,141** (Early stop bỏ sót nhánh/việc thứ 2, False Act trên `empty`/`no_app`)
  - `confusable_tool`: **1,097** (Đánh tráo tool hay nhầm cùng nhóm/liên app)
  - `hallucinated_optional_args`: **724** (Tự điền tham số tùy chọn hoặc làm biến dạng literal)
  - `premature_write_or_ask`: **372** (Nhảy cóc gọi tool ghi hoặc hỏi sớm khi chưa gọi tool đọc)

**Lệnh huấn luyện Giai đoạn 1:**
```bash
# Trên G4 96GB / A100 (bf16 LoRA thuần, không cần 4-bit)
python -m lifemate.planner.train_dpo \
  --model empero-ai/Qwen3.8-4B-Distill \
  --train-data optimize/dpo/dataset/dpo_train.jsonl \
  --val-data optimize/dpo/dataset/dpo_val.jsonl \
  --out-dir checkpoints/planner_dpo_qwen38_lora \
  --epochs 2.0 --lr 5e-6 --beta 0.1 --rank 16 --wrap-think \
  --max-prompt-length 9216 --max-length 10240 \
  --precompute-ref-log-probs --bs 2 --grad-accum 8  # dùng --bs 1 --grad-accum 16 nếu là A100 40GB

# Trên L4 22.5GB (bắt buộc bật QLoRA 4-bit)
python -m lifemate.planner.train_dpo \
  --model empero-ai/Qwen3.8-4B-Distill \
  --train-data optimize/dpo/dataset/dpo_train.jsonl \
  --val-data optimize/dpo/dataset/dpo_val.jsonl \
  --out-dir checkpoints/planner_dpo_qwen38_lora \
  --epochs 2.0 --lr 5e-6 --beta 0.1 --rank 16 --wrap-think \
  --max-prompt-length 9216 --max-length 10240 \
  --load-in-4bit --precompute-ref-log-probs --bs 1 --grad-accum 16
```

> **Lưu ý đặc thù về `<think>` của `empero-ai/Qwen3.8-4B-Distill`:**
> `chat_template.jinja` của `empero-ai/Qwen3.8-4B-Distill` chèn sẵn `<think>\n` ở cuối `add_generation_prompt=True` (hoặc `<think>\n\n</think>\n\n` khi truyền `enable_thinking=False`).
> - `train_dpo.py` tự kiểm tra `prompt_opens_think`: khi prompt đã mở `<think>\n`, cả `chosen` và `rejected` đều được đóng đối xứng bằng `</think>\n\n` sau phần `BƯỚC 1/2/3` trước `<tool_call>` (nên bật kèm `--wrap-think` để đồng bộ cả ở mọi chế độ).
> - Tránh tuyệt đối việc một bên có `<think>`/`</think>` còn bên kia không có, vì DPO sẽ học lầm vào độ dài/sự tồn tại của `<think>` thay vì tri thức chọn tool.

**Đo lường Trước vs Sau (Giai đoạn 1):**
```bash
# 1. Baseline Qwen 3.8 4B trên Val (300) và Test (100)
python -m lifemate.planner.runner --root data --split train --text --llm empero-ai/Qwen3.8-4B-Distill --mapper none --gpu-mem 0.90 --max-ctx 12288 --verify-rounds 0 --prompt prompts/gepa/vuong/planner_v15.txt --ids eval/splits/dev_ids.txt --resume --out runs/dev_qwen38_base.jsonl
python -m lifemate.planner.runner --root data --split train --text --llm empero-ai/Qwen3.8-4B-Distill --mapper none --gpu-mem 0.90 --max-ctx 12288 --verify-rounds 0 --prompt prompts/gepa/vuong/planner_v15.txt --ids eval/splits/test_ids.txt --resume --out runs/test_qwen38_base.jsonl

# 2. Sau DPO Giai đoạn 1
python -m lifemate.planner.runner --root data --split train --text --llm empero-ai/Qwen3.8-4B-Distill --llm-lora checkpoints/planner_dpo_qwen38_lora --mapper none --gpu-mem 0.90 --max-ctx 12288 --verify-rounds 0 --prompt prompts/gepa/vuong/planner_v15.txt --ids eval/splits/dev_ids.txt --resume --out runs/dev_qwen38_dpo_s1.jsonl
python -m lifemate.planner.runner --root data --split train --text --llm empero-ai/Qwen3.8-4B-Distill --llm-lora checkpoints/planner_dpo_qwen38_lora --mapper none --gpu-mem 0.90 --max-ctx 12288 --verify-rounds 0 --prompt prompts/gepa/vuong/planner_v15.txt --ids eval/splits/test_ids.txt --resume --out runs/test_qwen38_dpo_s1.jsonl

# 3. Chấm điểm chia theo app (Val 300 & Test 100)
python eval/scorer.py --sub runs/dev_qwen38_base.jsonl --compare runs/dev_qwen38_dpo_s1.jsonl --dev-ids eval/splits/dev_ids.txt --by-app
python eval/scorer.py --sub runs/test_qwen38_base.jsonl --compare runs/test_qwen38_dpo_s1.jsonl --dev-ids eval/splits/test_ids.txt --by-app
```

---

### Giai đoạn 2 (Fallback nếu Giai đoạn 1 chưa đạt kỳ vọng): On-Policy Rollout Sampling trên Colab
**Điều kiện kích hoạt Giai đoạn 2:**
- Điểm Mean Score trên `dev_ids.txt` (300 mẫu) sau Giai đoạn 1 tăng < `+0.0300` so với baseline `empero-ai/Qwen3.8-4B-Distill`, HOẶC
- Lỗi sinh format `<think>...</tool_call>` không ổn định do lệch phân phối off-policy giữa dữ liệu programmatic và thói quen sinh chuỗi `<think>` của `empero-ai/Qwen3.8-4B-Distill`.

**Cơ chế (`scripts/sample_onpolicy_dpo.py`):**
1. Dùng `vLLM` tải trực tiếp `empero-ai/Qwen3.8-4B-Distill` (hoặc kèm `--lora checkpoints/planner_dpo_qwen38_lora`) để sinh phản hồi trên `eval/splits/dpo_train_ids.txt` (1,145 mẫu): 1 rollout greedy (`temp=0.0`) + $K-1$ rollout ngẫu nhiên (`temp=0.7, top_p=0.9`).
2. Replay song song bước gold qua `zaic_tools.ToolEnv` để có trạng thái & `gold_calls` chuẩn ở từng lượt.
3. Đối chiếu từng rollout của chính model với `gold_calls` / `ToolEnv`:
   - Nếu model gọi đúng 100% tool & args ở bước đó: lưu làm ứng viên `chosen` tự nhiên (giữ nguyên `<think>...</think>` thật của model).
   - Nếu model gọi sai (`confusable_tool`, thiếu/thừa tham số, `FALSE_ACT`, `MISSED_ASK`, `FALSE_ASK`, sai cú pháp): ghép làm **`rejected` on-policy** đối xứng với **`chosen` đã xác thực qua `ToolEnv`**.
   - Căn chỉnh thẻ `<think>` giữa `chosen` và `rejected` theo `--think-mode prefix|strip|keep` (mặc định `prefix`) để đảm bảo cả 2 nhánh đối xứng về format `<think>`.
4. Huấn luyện tiếp DPO (LR nhỏ `2e-6`, `1.0` epoch) trên `optimize/dpo/dataset/dpo_onpolicy_train.jsonl` và đánh giá lại trên `dev_ids.txt` + `test_ids.txt`.

---

## 4. Hướng Dẫn Chọn Phần Cứng Trên Google Colab (`G4` vs `L4` vs `T4` vs `A100` vs `TPU v5e-1 / v6e-1`)

Vì prompt chứa danh sách JSON Schema các tool dài **5.9K–8.5K tokens** (tối đa ~9.4K tokens), DPO cần tính log-prob cho cả `chosen` và `rejected` trên chuỗi lên tới `9,216–10,240` tokens:

| Runtime Colab | Bộ nhớ / Kiến trúc | Hỗ trợ `bf16` & CUDA/vLLM | Cấu hình DPO khuyến nghị | Thời gian ước tính (Train + Eval) | Khuyến nghị |
|---|---|---:|---|---|---|
| **G4 (`RTX PRO 6000` Blackwell)** | **96 GB GDDR7** | Có (Blackwell, native `bf16`) | `bf16` LoRA thuần (tắt 4-bit), `--precompute-ref-log-probs --bs 2 --grad-accum 8` | **~25–40 phút** | **Khuyên dùng #1 (khi A100 hết slot)** — Dư VRAM, chạy cực nhanh cả DPO lẫn vLLM |
| **L4** | **22.5 GB VRAM** | Có (Ada SM89, native `bf16`) | Bắt buộc `--load-in-4bit --precompute-ref-log-probs --bs 1 --grad-accum 16` | **~2.5–3.5 giờ** | **Lựa chọn #2 (Tiết kiệm Compute Units)** — Ổn định, đủ chạy hết pipeline |
| **A100** | **40 GB / 80 GB VRAM** | Có (Ampere SM80) | `bf16` LoRA thuần, `--bs 1 --grad-accum 16` | **~45–60 phút** | Rất tốt nhưng hiện Colab đang hết suất (nếu trống lại vẫn dùng tốt) |
| **T4** | **15–16 GB VRAM** | **Không** (Turing SM75, chỉ `fp16`) | Dễ OOM / quá chậm khi train DPO trên context 6K–9K tokens | > 6 giờ / dễ OOM | **KHÔNG dùng T4 để train DPO** (chỉ dùng cho inference vLLM nếu cần) |
| **TPU (`v5e-1` / `v6e-1`)** | **16–32 GB HBM** (XLA) | **Không** (không có CUDA / vLLM / `bitsandbytes`) | Phải viết lại toàn bộ trainer sang `torch_xla` + biên dịch XLA lại theo độ dài prompt biến thiên | Không tương thích | **KHÔNG dùng TPU** — Stack hiện tại (`trl`/`peft`/`vllm`) chỉ chạy trên NVIDIA CUDA GPU |

> **Mẹo triển khai thực tế:** Mở `notebooks/colab_train_dpo_qwen38_4b.ipynb`, notebook tự nhận diện G4 / A100 / L4 / T4 (và báo lỗi rõ nếu chọn nhầm TPU) để tự điền sẵn cờ `--load-in-4bit`, `--bs`, `--grad-accum`, `--precompute-ref-log-probs` phù hợp.

---

## 5. Danh Sách Tệp Liên Quan Trong Repo

- **Notebook Colab hoàn chỉnh:** [notebooks/colab_train_dpo_qwen38_4b.ipynb](../../notebooks/colab_train_dpo_qwen38_4b.ipynb)
- **Script train DPO:** [lifemate/planner/train_dpo.py](../../lifemate/planner/train_dpo.py)
- **Script sinh dữ liệu DPO Giai đoạn 1:** [scripts/build_dpo_dataset.py](../../scripts/build_dpo_dataset.py)
- **Script lấy mẫu On-Policy Giai đoạn 2:** [scripts/sample_onpolicy_dpo.py](../../scripts/sample_onpolicy_dpo.py)
- **Dữ liệu DPO Giai đoạn 1:** [optimize/dpo/dataset/](README.md)
- **Scorer chia theo App (Trước vs Sau):** [eval/scorer.py](../../eval/scorer.py)
