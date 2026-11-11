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

# Chronological versions list
# Colorblind-safe palette (Okabe-Ito / CUD):
# - Baseline: Slate-700 (#334155)
# - Ordinary stepping stones: Single cold slate (#94a3b8) to reflect time, not individual importance
# - Meaningful results:
#   * V4 [Early Peak]: Amber (#e69f00)
#   * V6 [Guarded Peak]: Sky Blue (#0284c7)
#   * V12 [Champion]: Vermilion (#d55e00)
#   * V15 [Co-Champ]: Bluish Green (#009e73)

COLD_COLOR = "#94a3b8"

versions_raw = [
    ("Base", "baseline_v0", "#334155", True, "Baseline"),
    ("V1", "planner_v1", COLD_COLOR, False, "V1"),
    ("V2", "planner_v2", COLD_COLOR, False, "V2"),
    ("V3", "planner_v3", COLD_COLOR, False, "V3"),
    ("V4", "planner_v4", "#e69f00", True, "V4 [Early Peak]"),
    ("V5", "planner_v5", COLD_COLOR, False, "V5"),
    ("V6", "planner_v6_real", "#0284c7", True, "V6 [Guarded Peak]"),
    ("V7", "planner_v7_pure", COLD_COLOR, False, "V7"),
    ("V8", "planner_v8", COLD_COLOR, False, "V8"),
    ("V9", "planner_v9", COLD_COLOR, False, "V9"),
    ("V10", "planner_v10", COLD_COLOR, False, "V10"),
    ("V11", "planner_v11", COLD_COLOR, False, "V11"),
    ("V12", "planner_v12", "#d55e00", True, "V12 [Champion]"),
    ("V13", "planner_v13", COLD_COLOR, False, "V13"),
    ("V14", "planner_v14", COLD_COLOR, False, "V14"),
    ("V15", "planner_v15", "#009e73", True, "V15 [Co-Champ]")
]

metrics_keys = ["mean_score", "act", "ask", "empty"]
metrics_labels = [
    "Mean Score\n(N=300 Dev)",
    "Act F1\n(Actions, n=236)",
    "Ask F1\n(Missing Info, n=42)",
    "Empty F1\n(Out-of-Scope, n=22)"
]

base_scores = [manifest["baseline_v0"][k] for k in metrics_keys]

# 2. Setup Figure
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(24, 11), dpi=300)

n_groups = len(metrics_keys)
n_bars = len(versions_raw)
group_indices = np.arange(n_groups)

total_group_width = 0.88
bar_width = total_group_width / n_bars
start_offset = -total_group_width / 2 + bar_width / 2

for bar_idx, (short_name, key, color, is_meaningful, full_label) in enumerate(versions_raw):
    m = manifest[key]
    scores = [m[k] for k in metrics_keys]
    
    offset = start_offset + bar_idx * bar_width
    bar_positions = group_indices + offset
    
    # Meaningful versions get strong border, cold ones get subtle border
    edgecolor = "#0f172a" if is_meaningful else "#cbd5e1"
    linewidth = 1.6 if is_meaningful else 0.4
    alpha = 0.95 if is_meaningful else 0.70
    
    rects = ax.bar(
        bar_positions, scores, width=bar_width * 0.90,
        color=color, edgecolor=edgecolor, linewidth=linewidth,
        alpha=alpha, zorder=3
    )
    
    # Version tags at bottom of each bar
    for g_idx, rect in enumerate(rects):
        x_pos = rect.get_x() + rect.get_width() / 2
        
        # Label inside bottom of bar
        txt_tag_color = "#ffffff" if is_meaningful else "#1e293b"
        tag_weight = "bold" if is_meaningful else "normal"
        ax.text(
            x_pos, 0.015, short_name, ha="center", va="bottom",
            rotation=90, fontsize=8.0, color=txt_tag_color, fontweight=tag_weight, zorder=5
        )
        
        # Annotate score on top of bar
        score = scores[g_idx]
        base = base_scores[g_idx]
        pct_change = ((score - base) / base) * 100
        y_pos = rect.get_height() + 0.006
        
        if short_name == "Base":
            txt = f"{score:.3f}\n(Base)"
            txt_color = "#334155"
            weight = "bold"
            fsize = 7.5
        elif is_meaningful:
            sign = "+" if pct_change >= 0 else ""
            txt = f"{score:.3f}\n({sign}{pct_change:.1f}%)"
            txt_color = color
            weight = "bold"
            fsize = 8.0
        else:
            # Cold ordinary bar: muted text, smaller font to not distract
            sign = "+" if pct_change >= 0 else ""
            txt = f"{score:.3f}\n({sign}{pct_change:.1f}%)"
            txt_color = "#64748b"
            weight = "normal"
            fsize = 6.5
            
        ax.text(
            x_pos, y_pos, txt, ha="center", va="bottom",
            fontsize=fsize, color=txt_color, fontweight=weight, rotation=0, zorder=4
        )

# Styling and labels
ax.set_xticks(group_indices)
ax.set_xticklabels(metrics_labels, fontsize=14, fontweight="bold", color="#0f172a")
ax.set_ylabel("Evaluation Metric Score (F1 / Accuracy)", fontsize=13, fontweight="bold", labelpad=12, color="#0f172a")
ax.set_title(
    "ZAIC 2026 LifeMate: Toàn Cảnh Tiến Hoá Benchmark Prompt Planner (Baseline -> V15)\n"
    "300 Mẫu Dev Set (9 Held-out Apps) — Qwen3-4B-Instruct trên GPU Cloud",
    fontsize=16, fontweight="bold", pad=22, color="#0f172a"
)

# Reference baseline horizontal markers
for g_idx, base in enumerate(base_scores):
    x_min = group_indices[g_idx] - total_group_width / 2 - 0.04
    x_max = group_indices[g_idx] + total_group_width / 2 + 0.04
    ax.hlines(base, x_min, x_max, colors="#475569", linestyles="--", linewidth=1.2, zorder=2)

ax.set_ylim(0, 0.74)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=1)
ax.xaxis.grid(False)

# Concise Legend: only meaningful milestones + cold timeline bar
legend_elements = [
    mpatches.Patch(facecolor="#334155", edgecolor="#0f172a", label="Baseline (Mốc tham chiếu gốc)"),
    mpatches.Patch(facecolor=COLD_COLOR, edgecolor="#cbd5e1", alpha=0.7, label="Các bước trung gian V1-V15 (Cold slate: phản ánh tiến trình thời gian)"),
    mpatches.Patch(facecolor="#e69f00", edgecolor="#0f172a", label="V4 [Early Peak] (Đỉnh cao Phase 1 - Mean: 0.3597)"),
    mpatches.Patch(facecolor="#0284c7", edgecolor="#0f172a", label="V6 [Guarded Peak] (Đỉnh cao Ask có Guard - Ask: 0.3893)"),
    mpatches.Patch(facecolor="#d55e00", edgecolor="#0f172a", label="V12 [Champion] (Quán quân Pure Prompt - Mean: 0.3357, Act: 0.3608)"),
    mpatches.Patch(facecolor="#009e73", edgecolor="#0f172a", label="V15 [Co-Champ] (Đồng quán quân - Mean: 0.3353, Kỷ lục lệch hash 139)")
]

legend = ax.legend(
    handles=legend_elements,
    title="Các Mốc Tiến Hóa Trọng Yếu",
    title_fontsize=11.5,
    fontsize=10,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#cbd5e1",
    ncol=2
)
legend.get_title().set_fontweight("bold")

# Strict user requirement: The note box "Ghi chu" ONLY keeps the explanation for the number above the bar
info_text = "Ghi chú: Con số trên đầu mỗi cột gồm Điểm F1 tuyệt đối và (% Tăng/Giảm tương đối so với Baseline)."
ax.text(
    0.012, 0.965, info_text, transform=ax.transAxes,
    fontsize=10.5, fontweight="bold", verticalalignment="top", color="#1e293b",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
)

plt.tight_layout()

# Save plot
output_dir = Path("optimize/gepa/vuong/eval_results")
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "planner_benchmark_evolution.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Plot successfully saved to {plot_path}")
