# Thư mục Prompt GEPA của Vương (`prompts/gepa/vuong/`)

Nơi lưu trữ các phiên bản prompt Planner do Vương tối ưu qua các vòng lặp GEPA (manual / automated) trên tập 300 mẫu dev (`eval/splits/dev_ids.txt`).

---

## 📊 Báo cáo & Biểu đồ chi tiết
👉 **Xem báo cáo đầy đủ, biểu đồ và phân tích lỗi tại: [`RESULTS.md`](RESULTS.md)**  
*(Kèm ảnh biểu đồ phân tích 4 góc nhìn: [`gepa_benchmark_plots.png`](gepa_benchmark_plots.png))*

---

## 🏆 Phiên bản Champion hiện tại
* **`planner_v4.txt`**: **Mean F1: 0.3597** (Act: 0.3831, Ask: 0.1314, Empty: 0.5455). Tăng +0.0480 (+15.4%) so với Baseline ban đầu.

---

## Danh mục tài liệu:
- `planner_v0.txt` $\rightarrow$ `planner_v5.txt`: Các phiên bản prompt qua từng vòng lặp.
- `manifest.json`: Bảng JSON theo dõi thông số định lượng từng version.
- `RESULTS.md`: Báo cáo so sánh toàn diện, bảng chẩn đoán lỗi chi tiết và biểu đồ tiến trình.
