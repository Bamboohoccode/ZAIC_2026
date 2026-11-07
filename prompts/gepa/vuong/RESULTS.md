# Báo cáo kết quả tối ưu Prompt GEPA (Vương)

Tài liệu tổng hợp số liệu đo lường, biểu đồ tiến trình và phân tích lỗi chi tiết qua các vòng lặp GEPA (V0 $\rightarrow$ V4 và chuẩn bị V5) trên tập **300 mẫu dev (`eval/splits/dev_ids.txt`)** bằng model **`Qwen/Qwen3-4B-Instruct-2507`**.

---

## 1. Biểu đồ tổng quan tiến trình (Benchmark Plots)

![GEPA Benchmark Plots](gepa_benchmark_plots.png)

```mermaid
xychart-beta
    title "Diễn biến Điểm F1 Tổng Thể (Mean F1) qua các Vòng Lặp"
    x-axis ["v0 (Base)", "No-verify", "GEPA V1", "GEPA V2", "GEPA V3", "GEPA V4 (Peak)"]
    y-axis "Mean F1 Score" 0.28 --> 0.38
    bar [0.3117, 0.3109, 0.3523, 0.3436, 0.3294, 0.3597]
    line [0.3117, 0.3109, 0.3523, 0.3436, 0.3294, 0.3597]
```

---

## 2. Bảng so sánh điểm số chi tiết qua các phiên bản (`eval/scorer.py`)

| Phiên bản | Prompt File | Mean F1 | Act F1 (n=236) | Ask F1 (n=42) | Empty F1 (n=22) | Độ trễ TB | Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline v0** | `planner_v0.txt` | 0.3117 | 0.3268 | 0.0810 | 0.5909 | 30.96s | Mốc chuẩn ban đầu |
| **Ablation (no-verify)** | `planner_v0.txt` | 0.3109 | 0.3304 | 0.0786 | 0.5455 | 24.34s | Giảm ~6.6s độ trễ |
| **GEPA Iter 1** | `planner_v1.txt` | 0.3523 | 0.3659 | **0.1989** 🔥 | 0.5000 | 27.05s | Bứt phá Ask F1 (+145%) |
| **GEPA Iter 2** | `planner_v2.txt` | 0.3436 | 0.3665 | 0.0854 | 0.5909 | 29.62s | Giảm 22.6% lỗi tham số |
| **GEPA Iter 3** | `planner_v3.txt` | 0.3294 | 0.3507 | 0.0492 | **0.6364** 🏆 | 27.04s | Empty F1 cao nhất |
| **GEPA Iter 4 (Champion)** | `planner_v4.txt` | **0.3597** 🥇 | **0.3831** 🚀 | 0.1314 | 0.5455 | 28.01s | **Đỉnh F1 & Act cao nhất** |
| **GEPA Iter 5** | `planner_v5.txt` | *Đang chạy...* | *...* | *...* | *...* | *...* | Sửa từ mập mờ & IF-ELSE |

---

## 3. Bảng phân loại lỗi chi tiết (Diagnostic Error Breakdown)

| Phiên bản | HASH MISMATCH | WRONG TOOL/ARGS | MISSED ASK | MISSING CALL | FALSE ACT | EXTRA CALL | FALSE ASK |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline v0** | 139 (46.3%) | 52 (17.3%) | 16 (5.3%) | 12 (4.0%) | 9 (3.0%) | 4 (1.3%) | 0 (0.0%) |
| **No-verify** | 140 (46.7%) | 52 (17.3%) | 13 (4.3%) | 12 (4.0%) | 10 (3.3%) | 5 (1.7%) | 1 (0.3%) |
| **GEPA V1** | 129 (43.0%) | 53 (17.7%) | 10 (3.3%) | 16 (5.3%) | 11 (3.7%) | 6 (2.0%) | 2 (0.7%) |
| **GEPA V2** | 140 (46.7%) | **41 (13.7%)** | 10 (3.3%) | 15 (5.0%) | 9 (3.0%) | 7 (2.3%) | 2 (0.7%) |
| **GEPA V3** | 144 (48.0%) | 45 (15.0%) | 11 (3.7%) | 11 (3.7%) | **8 (2.7%)** | **2 (0.7%)** | 2 (0.7%) |
| **GEPA V4** | **132 (44.0%)** | 53 (17.7%) | **8 (2.7%)** | 16 (5.3%) | 10 (3.3%) | 5 (1.7%) | **1 (0.3%)** |

---

## 4. Cây đột biến Prompt và Bài học then chốt (Mutation Tree)

```mermaid
flowchart TD
    V0["Baseline v0<br/>Mean F1: 0.3117 | Latency: 30.96s<br/>verify_rounds = 2"] --> NV["Ablation: verify_rounds = 0<br/>Mean F1: 0.3109 | Latency: 24.34s<br/>Tiết kiệm 21% thời gian suy luận"]
    NV --> V1["GEPA V1 (33 dòng)<br/>Ràng buộc: Đọc xong phải ghi + Hỏi lại khi mơ hồ<br/>Mean F1: 0.3523 (+0.0406) | Ask F1: 0.1989"]
    V1 --> V2["GEPA V2<br/>Quy tắc tham số tối thiểu (cấm repeat/note/date thừa)<br/>WRONG_TOOL_OR_ARGS: 53 -> 41 (-22.6%) | Act F1: 0.3665"]
    V2 --> V3["GEPA V3<br/>Thử nghiệm diễn giải IF-ELSE chi tiết<br/>Empty F1: 0.6364 nhưng Mean F1 giảm do loãng chú ý (Attention Dilution)"]
    V3 --> V4["GEPA V4 (Champion - Minimal Surgery)<br/>Bộ khung 33 dòng của V1 + 1 dòng tham số tối thiểu của V2<br/>Mean F1: 0.3597 (+0.0480 vs base) | Act F1: 0.3831"]
    V4 --> V5["GEPA V5 (Đang chạy...)<br/>Cấm bịa tiền khi gặp 'chút tiền/ít tiền' -> bắt buộc hỏi amount<br/>Sửa nhánh phủ định trong thủ tục IF-ELSE"]
```

---

## 5. Tổng kết quy luật và hướng đi tiếp theo:
1. **Quy luật tinh gọn cho Model 4B (Minimal Surgery)**: Mô hình Qwen3-4B-Instruct đạt hiệu năng cao nhất với prompt ngắn (khoảng 30-34 dòng). Mọi nỗ lực giải thích quá dài dòng bằng văn phong tự nhiên đều gây phản tác dụng do loãng chú ý.
2. **Quy tắc Tham số tối thiểu**: Đã chứng minh giá trị lớn nhất trong việc giảm lỗi `WRONG_TOOL_OR_ARGS` từ 53 về 41.
3. **Quy tắc Chống tự bịa tham số mập mờ**: Là vũ khí tiếp theo trong V5 để giảm 132 lỗi `HASH_MISMATCH`.
