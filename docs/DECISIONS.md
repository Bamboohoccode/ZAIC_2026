# Nhật ký quyết định và kết quả ablation

Mỗi thay đổi model hoặc prompt ghi một dòng: thay gì, điểm trước và sau, độ trễ.

| Ngày | Người | Thay đổi | Điểm trước → sau | Độ trễ | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-07 | Tiên | LLM cho Task 1 là Qwen3-4B (`Qwen/Qwen3-4B`), chạy trên Kaggle | chưa đo | chưa đo | Người dùng đồng ý. Lệch với `pipeline.py` (Instruct-2507), chưa đổi `pipeline.py` |
| 2026-11-07 | Tiên | Baseline chạy `--text` (dùng `audio_text`), tắt mapper | chưa đo | chưa đo | Cô lập planner |
| 2026-11-07 | Tiên | Dev dùng `eval/splits/dev_ids.txt` của nhóm | gold dev = 1,0000 | - | Kiểm chuỗi pipeline + scorer |
| 2026-11-07 | Tiên | Nhãn ask_user: bỏ template v1, thay bằng v2 (65 mẫu) | - | - | Người dùng yêu cầu template; v1 nhãn sai nên thay (xem PROGRESS B4) |
| 2026-11-07 | Tiên | Kiểm tràn fp16 trên `mlp.down_proj`/`mlp.up_proj` (72 tensor): 0 tràn, 0 inf, 2.418 phần tử underflow | - | - | Chỉ kiểm trọng số, activation chưa kiểm (xem PROGRESS C6c) |
| 2026-11-08 | Tiên | `lifemate/pipeline.py`: `LLM_MODEL` đổi từ `Qwen/Qwen3-4B-Instruct-2507` sang `Qwen/Qwen3-4B`, khớp quyết định ngày 07/11 | chưa đo | chưa đo | Người dùng chọn Qwen3-4B (08/11). Zip `zaic_code_20261107_v3.zip` chưa chứa thay đổi này; lệnh WSP truyền `--llm` tường minh nên không ảnh hưởng |
| 2026-11-08 | Tiên | Duyệt quyết định k (vòng tìm đo trên practice 100 mẫu train) và l (luật giữ bản sửa: mean tăng >= 0,01 trên practice, gen_tokens TB <= 105% baseline, pipeline_error không tăng) | - | - | Người dùng đồng ý 08/11. Đặt trước khi có số đo, không chỉnh theo kết quả |
| 2026-11-09 | Tiên | LLM cho Task 1 đổi sang `Ma7ee7/Qwen3.8_4B_Distilled` (revision `f33da883`), tải về `models/Qwen3.8-4B-Distilled/`. Notebook WSP sinh lại, dùng model này | chưa đo | chưa đo | Người dùng chọn. Lý do: `empero-ai/Qwen3.8-4B` là kiến trúc Qwen3.5 (lệch `Qwen3ForCausalLM`, safetensors ~5B, sát trần 5B). Bản Ma7ee7 là `Qwen3ForCausalLM`, 4.022.468.096 tham số, Apache-2.0, tạo 30/07/2026. Bản này là fine-tune cộng đồng từ Qwen3-4B-Thinking-2507, không phải Qwen chính thức |
| 2026-11-09 | Tiên | Tách notebook WSP thành 2 lần Commit: A (baseline dev on+off, practice, tìm hướng) và B (đọc output A, tự tìm bản sửa, kiểm dev). `auto_search.py` thêm `--deadline-ts` (không bắt đầu vòng mới nếu sẽ vượt hạn). Gói `zaic_code_20261109_v4.zip` | - | - | Người dùng chọn giữ so sánh đủ (cả 2 baseline 300 mẫu dev). Ước 1 notebook > 12 giờ (35 giây/mẫu, lần QUICK 08/11). Giữ `MODE = "on"`: nhãn bad chỉ là thinking bị cắt/lặp, chế độ off gần như không có bad (`dev_off` QUICK: good 2, bad 0) |
| 2026-11-11 | Tiên | Gộp A/B thành 1 notebook `wsp_single_20261111.ipynb` (1 lần Commit), bỏ baseline think off. Sinh bởi `optimize/wsp/make_kaggle_notebook.py` | chưa đo | chưa đo | Người dùng yêu cầu 11/11. Lý do bỏ off: đo 09/11 cho thấy `dev_on` và `dev_off` giống hệt từng byte (`reports/wsp_run1_20261109.md` mục 2a). Thay thế quyết định 09/11 về tách A/B (dòng trên) |
| 2026-11-11 | Tiên | Notebook nhanh `wsp_fast_20261111.ipynb`: nhãn theo điểm (`--label-rule score`: bad = điểm 0, good = điểm 1,0), 10 mẫu `fast10_20261111.txt`, 3 vòng sửa, `--min-d 0`. Gói `zaic_code_20261111_v5.zip` | chưa đo | chưa đo | Người dùng yêu cầu 11/11 (cần ra bản sửa để báo cáo). Lần 09/11 không có vòng sửa vì practice bad 0 theo luật cắt/lặp, không phải timeout. Luật giữ (l) không đổi. 10 mẫu chọn theo kết quả cũ nên không dùng để kết luận go/no-go (`reports/wsp_run1_20261109.md` mục 6) |
| 2026-11-11 | Tiên | `directions.py`: forward theo đoạn 1024 token có KV cache, `--max-tokens` đổi nghĩa thành số token tối đa của lượt sinh (mặc định 2048, không còn tính cả prompt). Gói `zaic_code_20261111_v6.zip` thay v5 | chưa đo | chưa đo | Lần chạy 11/11 bị OOM ở `directions.py` (prompt 5.889 đến 8.491 token); mã cũ còn cắt output còn 1 token khi prompt > 8.192 (`reports/wsp_run1_20261109.md` mục 7) |

## Ghi chú quyết định (dạng chữ)

a) Sửa 2026-11-07 (người dùng đồng ý): dùng Qwen3-4B thay vì giới hạn "dưới 4B" đã nêu trước đó trong phiên.
b) 2026-11-07 (người dùng yêu cầu): ask_user tạo bằng template thay vì LLM viết lại. Sau khi đổi sang Opus, câu mới do Opus viết tay theo đúng cách BTC gán nhãn; không gọi API ngoài.
c) 2026-11-07: không dùng app chỉ có trong dev (9 app) cho dữ liệu sinh ra, để giữ dev là "app lạ".
e) 2026-11-07 (người dùng quyết định): Task 1 chỉ làm WSP, bỏ LoRA.
i) 2026-11-07 (người dùng quyết định): DỪNG lượng tử hoá (4-bit/8-bit). Lý do: WSP sửa khoảng 0,004% trọng số, lượng tử hoá có thể làm mất bản sửa hoặc lệch điểm so với bf16. Đã dừng tiến trình cài torch CUDA, xoá `.venv_wsp`. Luồng: chỉnh trọng số trên máy (CPU) hoặc bằng script do Opus local viết, chạy model bf16 trên Kaggle, đo trên BTC. Mã `HFLLM` và cờ `--hf-quant` còn trong repo nhưng không dùng.
h) 2026-11-07 (người dùng quyết định): chạy model cho WSP trên Kaggle, không chạy local (GPU local 6 GB, torch bản CPU). Phần sửa trọng số vẫn chạy được trên CPU local. Dữ liệu đưa lên Kaggle là bản không audio, dataset Private, do người dùng tự tải lên.
f) 2026-11-07: thêm cờ `--llm-think on|off` và `--llm-dtype` vào `lifemate/pipeline.py`, mặc định giữ hành vi cũ. Lý do: Qwen3-4B bật thinking mặc định với giới hạn 512 token mỗi lượt, có thể bị cắt trước `<tool_call>`; phải đo baseline cả hai chế độ để không quy nhầm lợi ích cho WSP. T4 không có bf16 nên cần `--llm-dtype half`.
g) 2026-11-07: WSP tìm hướng trên tập practice 100 mẫu train (không thuộc dev), đo trên 50 mẫu dev, kiểm cuối trên 300 mẫu dev. Lý do: tránh khớp quá mức vào tập đo.
j) 2026-11-07 (người dùng nhắc): Kaggle chỉ chạy được, không gọi được API Claude. Vì vậy vai "người sửa" của blog được thay bằng chính sách cố định trong `optimize/wsp/auto_search.py` do Opus viết trên máy. Opus chỉ suy luận thêm khi người dùng mang kết quả về máy.
k) 2026-11-07 (Opus chọn, ĐÃ được người dùng duyệt 08/11): sửa plan bước 4, vòng tìm đo trên tập practice 100 mẫu train thay cho 50 mẫu dev; dev 300 mẫu chỉ dùng cho baseline và kiểm cuối. Lý do: blog cho agent luyện trên tập train, đo trên tập riêng; chọn bản sửa trên dev 50 dễ khớp quá mức dev.
l) 2026-11-07 (Opus chọn, ĐÃ được người dùng duyệt 08/11): luật giữ bản sửa = mean tăng >= 0,01 trên practice, gen_tokens TB không vượt 105% baseline, pipeline_error không tăng. Đặt trước khi có số đo, không chỉnh theo kết quả.
m) 2026-11-07: vòng WSP chạy bằng bộ chạy theo lô `batch_agent.py` (cùng logic agent, đã kiểm từng byte bằng LLM giả). Chi phí so bằng gen_tokens; độ trễ 30 giây đo riêng bằng pipeline gốc (plan bước 7.3).
d) 2026-11-07: WSP cần Opus làm bước sửa trọng số. Người dùng đã chuyển phiên sang Opus; vòng sửa vẫn cần GPU Kaggle để chạy forward pass.
n) 2026-11-11 (người dùng quyết định): bộ data fine-tune là hội thoại nhiều lượt có tool result, chỉ dùng train BTC, không sinh thêm; cách fine-tune chưa chọn (mục e vẫn nguyên: Task 1 chỉ WSP). Loại `train_0066` vì trùng `audio_text` trong train.
