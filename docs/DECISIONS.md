# Nhật ký quyết định và kết quả ablation

Mỗi thay đổi model hoặc prompt ghi một dòng: thay gì, điểm trước và sau, độ trễ.

| Ngày | Người | Thay đổi | Điểm trước → sau | Độ trễ | Ghi chú |
|---|---|---|---|---|---|
| 06/11 | Vương | Baseline Qwen3-4B-Instruct (--text, mapper=none, verify_rounds=2, max_ctx=8192) trên dev 300 mẫu | N/A → 0.3117 (Act: 0.3268, Ask: 0.0810, Empty: 0.5909) | 31.4s/mẫu (130/300 mẫu > 30s) | Hash mismatch: 46.3%, Wrong tool/args: 17.3%, Missed ask: 5.3%. 130 mẫu quá 30s do vòng lặp self-verify 2 lượt |
