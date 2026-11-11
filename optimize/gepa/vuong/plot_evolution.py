import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(".").resolve()))

# 1. Load data
manifest_path = Path("prompts/gepa/vuong/manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# Versions to include (Strictly tested runs only)
versions_info = [
    ("Baseline", "baseline_v0", "#6c757d"),
    ("V8", "planner_v8", "#8ecae6"),
    ("V10", "planner_v10", "#219ebc"),
    ("V11", "planner_v11", "#023047"),
    ("V12 (Champion)", "planner_v12", "#d90429"),       # Champion Act & Mean
    ("V13", "planner_v13", "#fb8500"),
    ("V14", "planner_v14", "#ffb703"),
    ("V15 (Co-Champ)", "planner_v15", "#7209b7")        # Champion Hash & Co-Mean
]

# Metrics to display
metrics_keys = ["mean_score", "act", "ask", "empty"]
metrics_labels = [
    "Mean Score\n(N=300 Dev)",
    "Act F1\n(Actions, n=236)",
    "Ask F1\n(Missing Info, n=42)",
    "Empty F1\n(Out-of-Scope, n=22)"
]

base_scores = [manifest["baseline_v0"][k] for k in metrics_keys]

# Collect scores for each version
scores_by_version = {}
for label, key, color in versions_info:
    m = manifest[key]
    scores_by_version[label] = [m[k] for k in metrics_keys]

# 2. Setup Figure (Spacious Vertical Grouped Barplot)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(19, 10), dpi=300)

n_groups = len(metrics_keys)
n_bars = len(versions_info)
group_indices = np.arange(n_groups)

# Bar width and spacing
total_group_width = 0.82
bar_width = total_group_width / n_bars
start_offset = -total_group_width / 2 + bar_width / 2

for bar_idx, (label, key, color) in enumerate(versions_info):
    offset = start_offset + bar_idx * bar_width
    bar_positions = group_indices + offset
    scores = scores_by_version[label]
    
    # Highlight champion bars with distinct border
    is_special = "Champion" in label or "Co-Champ" in label
    edgecolor = "#0f172a" if is_special else "none"
    linewidth = 1.8 if is_special else 0.5
    
    rects = ax.bar(
        bar_positions, scores, width=bar_width * 0.92,
        color=color, label=label, edgecolor=edgecolor, linewidth=linewidth,
        alpha=0.92, zorder=3
    )
    
    # Annotate percentage relative to baseline above each bar
    for g_idx, rect in enumerate(rects):
        score = scores[g_idx]
        base = base_scores[g_idx]
        pct_change = ((score - base) / base) * 100
        
        # Position of text
        x_pos = rect.get_x() + rect.get_width() / 2
        y_pos = rect.get_height() + 0.008
        
        if label == "Baseline":
            txt = f"{score:.3f}\n(Base)"
            color_txt = "#4b5563"
            weight = "bold"
        else:
            sign = "+" if pct_change >= 0 else ""
            txt = f"{score:.3f}\n({sign}{pct_change:.1f}%)"
            if pct_change > 0:
                color_txt = "#b91c1c" if "Champion" in label else "#7209b7" if "Co-Champ" in label else "#047857"
            else:
                color_txt = "#b91c1c"
            weight = "bold" if abs(pct_change) >= 8 or is_special else "normal"
            
        ax.text(
            x_pos, y_pos, txt, ha="center", va="bottom",
            fontsize=8.0, color=color_txt, fontweight=weight, rotation=0
        )

# Styling and labels
ax.set_xticks(group_indices)
ax.set_xticklabels(metrics_labels, fontsize=13, fontweight="bold", color="#111827")
ax.set_ylabel("Evaluation Metric Score (F1 / Accuracy)", fontsize=13, fontweight="bold", labelpad=12, color="#111827")
ax.set_title(
    "ZAIC 2026 LifeMate: Tiến Hoá Benchmark Prompt Planner (Baseline -> V15)\n"
    "300 Mẫu Dev Set (9 Held-out Apps) — Qwen3-4B-Instruct trên GPU Cloud",
    fontsize=15, fontweight="bold", pad=20, color="#0f172a"
)

# Reference baseline horizontal markers
for g_idx, base in enumerate(base_scores):
    x_min = group_indices[g_idx] - total_group_width / 2 - 0.03
    x_max = group_indices[g_idx] + total_group_width / 2 + 0.03
    ax.hlines(base, x_min, x_max, colors="#94a3b8", linestyles="--", linewidth=1.2, zorder=2)

ax.set_ylim(0, 0.72)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=1)
ax.xaxis.grid(False)

# Legend positioning
legend = ax.legend(
    title="Phiên Bản Planner (Đã chạy Dev)",
    title_fontsize=11,
    fontsize=10,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#cbd5e1",
    ncol=4
)
legend.get_title().set_fontweight("bold")

# Add informative annotation box
info_text = (
    "[*] Ghi chú tiến hoá thực nghiệm:\n"
    "• V12 (Champion): Đỉnh cao Act F1 (0.361, +10.4%) & Mean F1 (0.336, +7.7% vs base).\n"
    "• V15 (Co-Champion): Kỷ lục Lệch Hash thấp nhất lịch sử (139 ca, 46.3%), Mean F1 0.335.\n"
    "• V14: Kỷ lục sai tham số (Wrong Tool/Args) thấp nhất lịch sử (36 ca, 12.0%).\n"
    "• Ask F1: V11 & V13 đạt ~0.14 nhờ Ambiguity Gate & Anti-Hallucination (+70% vs base).\n"
    "• Số phần trăm trên mỗi cột thể hiện mức Tăng/Giảm tương đối so với Baseline."
)
ax.text(
    0.015, 0.96, info_text, transform=ax.transAxes,
    fontsize=9.5, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95)
)

plt.tight_layout()

# Save plot
output_dir = Path("optimize/gepa/vuong/eval_results")
output_path = output_dir / "planner_benchmark_evolution.png"
plt.savefig(output_path, dpi=300)
print(f"Chart successfully saved to: {output_path}")

plt.close()
