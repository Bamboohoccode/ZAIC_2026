# Thư mục Prompt GEPA của Vương (`prompts/gepa/vuong/`)

Nơi lưu trữ các phiên bản prompt Planner do Vương tối ưu qua các vòng lặp GEPA (manual / automated).

## Quy ước lưu trữ
- `system_v1.txt`, `system_v2.txt`, ...: Từng phiên bản system prompt của Planner.
- `manifest.json`: Bảng theo dõi điểm số dev F1 (Mean, Act, Ask, Empty), latency và commit hash tương ứng của từng version.
- Chỉ cập nhật phiên bản tốt nhất vào pipeline khi điểm số trên tập dev (`eval/splits/dev_ids.txt`) cao hơn baseline.
