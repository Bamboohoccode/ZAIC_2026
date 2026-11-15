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

### Giai đoạn 1 (Chạy trước): Programmatic ToolEnv-Verified DPO (v2 — Balanced & Symmetric CoT)
Sử dụng bộ dữ liệu đã được sinh và kiểm chứng `gold_hash_verified=1144, gold_hash_failed=0` bằng `scripts/build_dpo_dataset.py` (loại bỏ `train_0066` trùng lặp nội bộ từ 1,145 IDs):
- `optimize/dpo/dataset/dpo_train.jsonl`: **4,099 cặp** (1,087 mẫu train, **44.679 MB**).
- `optimize/dpo/dataset/dpo_val.jsonl`: **203 cặp** (57 mẫu nội bộ từ `dpo_train_ids.txt`, **2.173 MB**).
- Total: **4,302 cặp** (`tool_specs_catalog.json`: **0.082 MB**, Bảng JSON Schema 147 tools).
- Phân bố 5 nhóm lỗi hard negative (`optimize/dpo/dataset/dpo_stats.json`):
  - `multi_action_or_scope_error`: **1,776** (Early stop bỏ sót nhánh/việc thứ 2, False Act trên `empty`/`no_app`, và extra write distractor via `DISTRACTOR_WRITE_TOOLS`)
  - `hallucinated_optional_args`: **1,023** (Tự điền tham số tùy chọn hoặc làm biến dạng literal)
  - `ambiguity_gate_error`: **875** (Missed Ask chọn đại ID / False Ask trên write & read)
  - `premature_write_or_ask`: **329** (Nhảy cóc gọi tool ghi hoặc hỏi sớm khi chưa gọi tool đọc)
  - `confusable_tool`: **299** (Đánh tráo tool hay nhầm cùng nhóm/liên app — cân bằng `swap_balance` hai chiều)

> **3 cải tiến cốt lõi ở Dataset & Trainer v2 (khắc phục lỗi v1):**
> 1. **Shared byte-for-byte `<think>` prefix (`make_shared_think_prefix` + `align_pair_think`):** Cả `chosen` và `rejected` dùng chung nguyên văn phần `<think>...</think>` trung tính kiểu V15 3-step (hoặc cùng rollout prefix ở Stage 2), chỉ khác đúng khối `<tool_call>` quyết định — triệt tiêu hiện tượng DPO bão hòa loss theo độ dài/văn phong `BƯỚC 1/2/3`.
> 2. **Cân bằng `confusable_tool` + thêm `spurious_extra_write`:** `gen_negative_confusable` đa bước dùng cơ chế `swap_balance` theo cặp tool (không dồn âm quá mức vào `banking.list_beneficiaries`, `banking.transfer`, `smart_home.switch_set`, `calendar.create_event`, `ride_hailing.get_quotes`), kết hợp âm bản gọi dư tool ghi (`DISTRACTOR_WRITE_TOOLS`).
> 3. **Cân bằng `step>=1` & tách thư mục resume (`select_balanced_pairs` + run-scoped Drive resume):** Chọn tối đa `K=2` cặp/mẫu ưu tiên quay vòng lượt sau (`45.8%` `step>=1`: 937/2,046 cặp) và cân bằng `tool_net`, đồng thời `resolve_resume_checkpoint` khoanh vùng Drive fallback theo tên run (`planner_dpo_qwen38_v2_lora`) để không kéo nhầm `checkpoint-100` v1 cũ.

**Lệnh huấn luyện Giai đoạn 1 (tối ưu tốc độ + nohup + lưu checkpoint & weight LoRA mỗi 10 step):**
```bash
# Trên G4 96GB / A100 (bf16 LoRA thuần, tắt --precompute-ref-log-probs để PEFT toggle adapter tức thì)
nohup python -u -X utf8 -m lifemate.planner.train_dpo \
  --model empero-ai/Qwen3.8-4B-Distill \
  --train-data optimize/dpo/dataset/dpo_train.jsonl \
  --val-data optimize/dpo/dataset/dpo_val.jsonl \
  --out-dir checkpoints/planner_dpo_qwen38_v2_lora \
  --epochs 1.0 --lr 5e-6 --beta 0.1 --rank 16 --wrap-think \
  --max-pairs-per-sample 2 --max-prompt-length 9216 --max-length 10240 \
  --bs 2 --grad-accum 8 --save-steps 10 --eval-steps 50 --logging-steps 2 \
  --auto-resume --drive-backup-dir /content/drive/MyDrive/ZAIC_2026_runs/checkpoints/planner_dpo_qwen38_v2_lora \
  > /content/drive/MyDrive/ZAIC_2026_runs/train_dpo_s1_v2.log 2>&1 &

# Trên L4 22.5GB (QLoRA 4-bit + TF32 + SDPA + --auto-resume + lưu weight mỗi 10 step)
nohup python -u -X utf8 -m lifemate.planner.train_dpo \
  --model empero-ai/Qwen3.8-4B-Distill \
  --train-data optimize/dpo/dataset/dpo_train.jsonl \
  --val-data optimize/dpo/dataset/dpo_val.jsonl \
  --out-dir checkpoints/planner_dpo_qwen38_v2_lora \
  --epochs 1.0 --lr 5e-6 --beta 0.1 --rank 16 --wrap-think \
  --max-pairs-per-sample 2 --max-prompt-length 9216 --max-length 10240 \
  --load-in-4bit --bs 1 --grad-accum 16 --save-steps 10 --eval-steps 50 --logging-steps 2 \
  --auto-resume --drive-backup-dir /content/drive/MyDrive/ZAIC_2026_runs/checkpoints/planner_dpo_qwen38_v2_lora \
  > /content/drive/MyDrive/ZAIC_2026_runs/train_dpo_s1_v2.log 2>&1 &
```

> **Bài học từ phiên Colab L4 >500 phút & ngắt kết nối giữa chừng (đã khắc phục triệt để):**
> 1. **Chạy nền qua `nohup` + auto-reattach:** Cell 6B & 8B chạy `nohup` (detached session, ghi log thẳng vào Drive `/content/drive/MyDrive/ZAIC_2026_runs/train_dpo_s1_v2.log`) kèm tail log trực tiếp. Nếu rớt mạng/mất kết nối tab trình duyệt mà VM vẫn sống, chạy lại cell sẽ tự gắn vào PID đang chạy mà không train lại từ đầu.
> 2. **Lưu checkpoint + weight LoRA hoàn chỉnh mỗi 10 step (~8%):** Với ~128 step (1 epoch trên 2,046 cặp đã cân bằng), `--save-steps 10` xuất cả `checkpoint-<step>` lẫn weight LoRA dùng ngay (`adapter_config.json`, `adapter_model.safetensors`, `latest_checkpoint_info.json`) lên Google Drive sau mỗi 10 step; bật `--auto-resume` sẽ tự khôi phục từ checkpoint mới nhất của cùng tên run trên Drive nếu Colab bị thu hồi VM.
> 3. **Tắt `--precompute-ref-log-probs` & bật TF32/SDPA:** Trên PEFT LoRA, `DPOTrainer` tự tắt/bật adapter (`ref_model=None`) không tốn thêm VRAM và vào Step 1 ngay lập tức. Kết hợp `--max-pairs-per-sample 2 --epochs 1.0` (`4,099 -> 2,046` cặp phủ đủ 1,087 mẫu train, `45.8%` `step>=1` & 5 nhóm lỗi) giảm thời gian L4 xuống ~45–60 phút (G4/A100 ~15–30 phút).

> **Lưu ý đặc thù về `<think>` của `empero-ai/Qwen3.8-4B-Distill`:**
> `chat_template.jinja` của `empero-ai/Qwen3.8-4B-Distill` chèn sẵn `<think>\n` ở cuối `add_generation_prompt=True` (hoặc `<think>\n\n</think>\n\n` khi truyền `enable_thinking=False`).
> - `train_dpo.py` tự kiểm tra `prompt_opens_think` và kết hợp với `make_shared_think_prefix` trong dataset v2: cả `chosen` và `rejected` đều mang **cùng một khối `<think>...</think>` giống hệt từng byte** trước `<tool_call>` (bật `--wrap-think` để đồng bộ ở mọi chế độ).
> - Điều này đảm bảo tuyệt đối DPO chỉ tối ưu hóa quyết định `<tool_call>` chứ không học lầm vào độ dài hay văn phong `<think>`.

**Baseline `empero-ai/Qwen3.8-4B-Distill` đã chạy xong & lưu sẵn trong repo (`optimize/dpo/baseline_runs/`):**
- **Val (`dev_ids.txt`, N=300):** Mean Score = `0.3940` (`act=0.4149`, `ask=0.0778`, `empty=0.7727`; `held_out_val=0.3697`, `seen_train=0.2488`).
- **Test (`test_ids.txt`, N=100 — KHÓA TRONG LÚC TUNE):** Mean Score = `0.4394` (`act=0.4954`, `ask=0.0571`, `empty=0.5714`; `held_out_test=0.4804`, `seen_train=0.3007`). Trong suốt quá trình lặp chỉnh DPO, **chỉ đánh giá & tune trên Val (`dev_ids.txt`, N=300)**; mọi bước chạy/chấm `test_ids.txt` bị khóa sau `RUN_FINAL_TEST_EVAL = False` trong notebook để tránh rò rỉ (leak) tập Test.
- Cell 6A của notebook tự khôi phục baseline từ `optimize/dpo/baseline_runs/` (hoặc Google Drive) và bỏ qua bước chạy vLLM Baseline để dành toàn bộ GPU cho Train DPO.

**Phân tích kết quả v1 (`checkpoint-100`) & lý do chuyển sang v2 (Strict Val-only Tuning):**
- **Kết quả run v1 (`checkpoints/planner_dpo_qwen38_lora/checkpoint-100`):** Test N=100 tăng `0.4394 -> 0.4877 (+0.0483)`, nhưng **Val N=300 giảm nhẹ `0.3940 -> 0.3888 (-0.0052)`** (lỗi `HASH_MISMATCH` tăng `122 -> 131`) chủ yếu do `banking` tụt `0.2165 -> 0.0520 (-0.1645)`, kéo theo sụt điểm trên các mẫu `gold_price` (`15/33` mẫu dùng chung `banking`) và `exchange_rate` (`3/25` mẫu) trong `dev_ids.txt`.
- **Nguyên nhân gốc đã kiểm chứng trên code & dữ liệu v1:**
  1. `select_balanced_pairs(..., 2)` trước đây chọn **100% `step=0` và 0% `step>=1`** (`2,027` cặp), nên model mất phản xạ nối bước sau kết quả tra cứu đầu tiên.
  2. `gen_negative_confusable` trước đây thiên lệch đánh tráo âm nặng vào một số tool (`banking.list_beneficiaries` net `-106`, `smart_home.switch_set` `-59`, `calendar.create_event` `-40`, `ride_hailing.get_quotes` `-33`, `banking.transfer` `-27`), khiến model né gọi `banking`.
  3. Chênh lệch độ dài `chosen` (~400 ký tự `BƯỚC 1/2/3`) vs `rejected` (~100 ký tự) làm DPO loss bão hòa sớm (`0.0012`) theo văn phong `<think>` thay vì học chọn `<tool_call>`.
- **Cách chạy v2 sạch (`checkpoints/planner_dpo_qwen38_v2_lora`):** Notebook Cell 6B mặc định train run v2 hoàn toàn mới từ dữ liệu v2 đã cân bằng (Drive auto-resume tự khoanh vùng theo `planner_dpo_qwen38_v2_lora`, không lẫn `checkpoint-100` v1) và Cell 7 chấm điểm riêng lên `runs/dev_qwen38_dpo_s1_v2.jsonl` (chỉ trên `dev_ids.txt`).

**Đo lường Sau DPO (Giai đoạn 1 — Chỉ chấm Val trong lúc tune):**
```bash
# 1. Sau DPO Giai đoạn 1 v2 (chỉ chạy & chấm Val N=300; test_ids.txt giữ nguyên ở RUN_FINAL_TEST_EVAL = False)
python -m lifemate.planner.runner --root data --split train --text --llm empero-ai/Qwen3.8-4B-Distill --llm-lora checkpoints/planner_dpo_qwen38_v2_lora --mapper none --gpu-mem 0.90 --max-ctx 12288 --verify-rounds 0 --prompt prompts/gepa/vuong/planner_v15.txt --ids eval/splits/dev_ids.txt --resume --out runs/dev_qwen38_dpo_s1_v2.jsonl

# 2. Chấm điểm chia theo app trên Val 300 (Trước vs Sau)
python eval/scorer.py --sub optimize/dpo/baseline_runs/dev_qwen38_base.jsonl --compare runs/dev_qwen38_dpo_s1_v2.jsonl --dev-ids eval/splits/dev_ids.txt --by-app
```

---

### Giai đoạn 2 (Chỉ chạy nếu Val 300 chưa đạt ngưỡng sau Giai đoạn 1 v2): On-Policy Self-Play DPO
**Điều kiện kích hoạt (chỉ dựa trên Val `dev_ids.txt`):**
- Điểm Mean Score trên `dev_ids.txt` (300 mẫu) sau Giai đoạn 1 tăng < `+0.0300` so với baseline `empero-ai/Qwen3.8-4B-Distill`, HOẶC
- Lỗi sinh format `<think>...</tool_call>` không ổn định do lệch phân phối off-policy giữa dữ liệu programmatic và thói quen sinh chuỗi `<think>` của `empero-ai/Qwen3.8-4B-Distill`.

**Cơ chế (`scripts/sample_onpolicy_dpo.py`):**
1. Dùng `vLLM` tải trực tiếp `empero-ai/Qwen3.8-4B-Distill` (hoặc kèm `--lora checkpoints/planner_dpo_qwen38_v2_lora`) để sinh phản hồi trên `eval/splits/dpo_train_ids.txt` (1,145 mẫu): 1 rollout greedy (`temp=0.0`) + $K-1$ rollout ngẫu nhiên (`temp=0.7, top_p=0.9`).
2. Replay song song bước gold qua `zaic_tools.ToolEnv` để có trạng thái & `gold_calls` chuẩn ở từng lượt.
3. Đối chiếu từng rollout của chính model với `gold_calls` / `ToolEnv`:
   - Nếu model gọi đúng 100% tool & args ở bước đó: lưu làm ứng viên `chosen` tự nhiên (giữ nguyên `<think>...</think>` thật của model).
   - Nếu model gọi sai (`confusable_tool`, thiếu/thừa tham số, `FALSE_ACT`, `MISSED_ASK`, `FALSE_ASK`, sai cú pháp): ghép làm **`rejected` on-policy** đối xứng với **`chosen` đã xác thực qua `ToolEnv`**.
   - Căn chỉnh thẻ `<think>` qua `align_pair_think` theo `--think-mode prefix|strip|keep` (mặc định `prefix`) để hai nhánh dùng chung **cùng một khối `<think>...</think>` nguyên văn** (kể cả kiểu rollout chỉ kết thúc bằng `</think>`), đảm bảo 100% đối xứng.
4. Huấn luyện tiếp DPO (LR nhỏ `2e-6`, `1.0` epoch) trên `optimize/dpo/dataset/dpo_onpolicy_train.jsonl` và đánh giá lại trên `dev_ids.txt` (chỉ mở khóa `test_ids.txt` qua `RUN_FINAL_TEST_EVAL = True` khi chốt mô hình cuối cùng).

---

## 4. Hướng Dẫn Chọn Phần Cứng Trên Google Colab (`G4` vs `L4` vs `T4` vs `A100` vs `TPU v5e-1 / v6e-1`)

Vì prompt chứa danh sách JSON Schema các tool dài **5.9K–8.5K tokens** (tối đa ~9.4K tokens), DPO cần tính log-prob cho cả `chosen` và `rejected` trên chuỗi lên tới `9,216–10,240` tokens:

| Runtime Colab | Bộ nhớ / Kiến trúc | Hỗ trợ `bf16` & CUDA/vLLM | Cấu hình DPO khuyến nghị (`--max-pairs-per-sample 2 --epochs 1.0`, tắt `precompute_ref`) | Thời gian ước tính (Train) | Khuyến nghị |
|---|---|---:|---|---|---|
| **G4 (`RTX PRO 6000` Blackwell)** | **96 GB GDDR7** | Có (Blackwell, native `bf16`) | `bf16` LoRA thuần (tắt 4-bit), `--bs 2 --grad-accum 8`, TF32+SDPA | **~15–20 phút** | **Khuyên dùng #1 (khi A100 hết slot)** — Dư VRAM, chạy cực nhanh cả DPO lẫn vLLM |
| **L4** | **22.5 GB VRAM** | Có (Ada SM89, native `bf16`) | QLoRA `--load-in-4bit --bs 1 --grad-accum 16`, TF32+SDPA, `--auto-resume` | **~45–60 phút** | **Lựa chọn #2 (Tiết kiệm Compute Units)** — Auto-backup Drive mỗi 10 step |
| **A100** | **40 GB / 80 GB VRAM** | Có (Ampere SM80) | `bf16` LoRA thuần, `--bs 1 --grad-accum 16`, TF32+SDPA | **~20–30 phút** | Rất tốt nhưng hiện Colab đang hết suất (nếu trống lại vẫn dùng tốt) |
| **T4** | **15–16 GB VRAM** | **Không** (Turing SM75, chỉ `fp16`) | Dễ OOM / quá chậm khi train DPO trên context 6K–9K tokens | > 4 giờ / dễ OOM | **KHÔNG dùng T4 để train DPO** (chỉ dùng cho inference vLLM nếu cần) |
| **TPU (`v5e-1` / `v6e-1`)** | **16–32 GB HBM** (XLA) | **Không** (không có CUDA / vLLM / `bitsandbytes`) | Phải viết lại toàn bộ trainer sang `torch_xla` + biên dịch XLA lại theo độ dài prompt biến thiên | Không tương thích | **KHÔNG dùng TPU** — Stack hiện tại (`trl`/`peft`/`vllm`) chỉ chạy trên NVIDIA CUDA GPU |

> **Mẹo triển khai thực tế:** Mở `notebooks/colab_train_dpo_qwen38_4b.ipynb`, notebook tự nhận diện G4 / A100 / L4 / T4 (và báo lỗi rõ nếu chọn nhầm TPU), tự điền sẵn cờ `--load-in-4bit`, `--bs`, `--grad-accum`, bỏ qua bước chạy Baseline nếu `optimize/dpo/baseline_runs/` đã đủ mẫu, chạy nền qua `nohup` (tự re-attach log nếu mất kết nối UI), và tự lưu checkpoint + weight LoRA lên Google Drive mỗi 10 step để `--auto-resume` bất kỳ lúc nào.

---

## 5. Danh Sách Tệp Liên Quan Trong Repo

- **Notebook Colab hoàn chỉnh:** [notebooks/colab_train_dpo_qwen38_4b.ipynb](../../notebooks/colab_train_dpo_qwen38_4b.ipynb)
- **Script train DPO:** [lifemate/planner/train_dpo.py](../../lifemate/planner/train_dpo.py)
- **Script sinh dữ liệu DPO Giai đoạn 1:** [scripts/build_dpo_dataset.py](../../scripts/build_dpo_dataset.py)
- **Script lấy mẫu On-Policy Giai đoạn 2:** [scripts/sample_onpolicy_dpo.py](../../scripts/sample_onpolicy_dpo.py)
- **Dữ liệu DPO Giai đoạn 1:** [optimize/dpo/dataset/](README.md)
- **Baseline Qwen 3.8 4B đã hoàn tất (Val 300 & Test 100):** [optimize/dpo/baseline_runs/](baseline_runs/baseline_summary.txt)
- **Scorer chia theo App (Trước vs Sau):** [eval/scorer.py](../../eval/scorer.py)
