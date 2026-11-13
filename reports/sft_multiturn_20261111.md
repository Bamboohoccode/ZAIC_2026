# Bộ data nhiều lượt có tool result (derived, sinh 20261111)

Tự sinh bởi `scripts/build_sft_multiturn.py`. Nguồn: train BTC. Không đọc public_test.

## 1. Số đếm

- Nguồn: 1545 mẫu. Loại vì trùng audio_text với dev: 0 [].
- train: 1244, dev: 300, tổng 1544.
- Loại vì trùng audio_text trong nội bộ train (giữ mẫu đầu): 1 ['train_0066'].
- train theo category: act 980, ask 175, empty 89.
- dev theo category: act 236, ask 42, empty 22.

## 2. Kiểm ToolEnv (replay gold trên state.json)

- Call ngoài tool được cài: 0 []
- Mẫu có call bị từ chối schema (ok=false): 0 []
- Mẫu có tool_results lệch nhãn BTC: 0 []
- Mẫu có final_state lệch nhãn BTC: 0 []
- Không sửa nhãn; mẫu lệch vẫn được xuất, báo để người dùng quyết định.

## 3. Độ dài token (chat template của tokenizer Qwen3.8-4B-Distilled)

- trung vị 4891, p90 7032, p99 8459, max 9384.
- mẫu > 4096: 1089; > 8192: 23; > 16384: 0.
- Lỗi chat template: 0.
- Không cắt tool_results. Cột `n_tokens` trong file để lọc theo ngưỡng.

## 4. Phân bố

- Số call mỗi mẫu: {0: 111, 1: 368, 2: 428, 3: 359, 4: 161, 5: 107, 6: 10}
- Số lượt history: {0: 526, 2: 288, 4: 267, 6: 137, 8: 81, 10: 76, 12: 41, 18: 33, 20: 47, 22: 35, 24: 13}

## 5. Chưa xác minh

- Nhãn BTC không có câu trả lời bằng lời của assistant sau tool result cuối ở lượt cuối; data kết thúc ở call cuối.
- `arguments` lưu dạng dict (không phải chuỗi JSON). Cần đổi nếu trainer yêu cầu chuỗi.
- `train: true/false` trong messages là mask loss gợi ý, tuỳ trainer có dùng không.
