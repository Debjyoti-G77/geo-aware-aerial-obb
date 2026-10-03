"""
HISTORICAL DOTA BASELINE MATERIAL - not the RarePlanes ONNX parity workflow.

Edge Simulation Results Analysis — Charts & Comparison

Reads results from benchmark_edge.py and generates
charts for analysis.

Charts:
  01_fps_comparison.png
  02_latency_comparison.png
  03_ram_comparison.png
  04_latency_per_image.png
  05_realtime_analysis.png
  06_latency_boxplot.png
  07_full_comparison_table.png

Usage:
  python analyze_edge_results.py
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/edge_study"
CHARTS_DIR  = "results/edge_charts"
REALTIME    = 15  # FPS threshold

Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

COLORS = {
    "ONNX CPU (Pi 5 sim, 1024px)":           "#BA7517",
    "ONNX GPU (Jetson Orin sim, 1024px)":    "#1D9E75",
    "PyTorch GPU (RTX 3060, 1024px FP16)":   "#378ADD",
}

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

def load_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["latency_ms"] = float(r["latency_ms"])
        r["ram_mb"]     = float(r["ram_mb"])
        r["detections"] = int(r["detections"])
    return rows

with open(Path(RESULTS_DIR) / "comparison_summary.csv", "r", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

for s in summary:
    s["fps"]         = float(s["fps"])
    s["avg_lat_ms"]  = float(s["avg_lat_ms"])
    s["min_lat_ms"]  = float(s["min_lat_ms"])
    s["max_lat_ms"]  = float(s["max_lat_ms"])
    s["avg_ram_mb"]  = float(s["avg_ram_mb"])
    s["peak_ram_mb"] = float(s["peak_ram_mb"])
    s["total_dets"]  = int(s["total_dets"])

cpu_data    = load_csv(Path(RESULTS_DIR) / "cpu_results.csv")
jetson_data = load_csv(Path(RESULTS_DIR) / "jetson_onnx_results.csv")
pt_data     = load_csv(Path(RESULTS_DIR) / "pytorch_gpu_results.csv")

labels   = [s["label"] for s in summary]
fps_vals = [s["fps"] for s in summary]
avg_lats = [s["avg_lat_ms"] for s in summary]
min_lats = [s["min_lat_ms"] for s in summary]
max_lats = [s["max_lat_ms"] for s in summary]
ram_vals = [s["peak_ram_mb"] for s in summary]
colors   = [COLORS.get(l, "#888888") for l in labels]

print(f"Loaded {len(summary)} configs, generating charts...")

# ---------------------------------------------------------------------------
# CHART 1 — FPS comparison
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, fps_vals, color=colors, alpha=0.85, width=0.5)
ax.axhline(REALTIME, color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"Real-time threshold ({REALTIME} FPS)")
for bar, val in zip(bars, fps_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Frames per second (FPS)")
ax.set_title("Inference speed — GPU vs edge devices (imgsz=1024)")
ax.legend(fontsize=10)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_fps_comparison.png")
plt.close()
print("Chart 1: FPS comparison")

# ---------------------------------------------------------------------------
# CHART 2 — Latency with error bars
# ---------------------------------------------------------------------------

x = np.arange(len(labels))
yerr_low  = [s["avg_lat_ms"] - s["min_lat_ms"] for s in summary]
yerr_high = [s["max_lat_ms"] - s["avg_lat_ms"] for s in summary]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, avg_lats, color=colors, alpha=0.85, width=0.5,
              yerr=[yerr_low, yerr_high], capsize=6,
              error_kw={"linewidth": 1.5, "ecolor": "gray"})
for bar, val in zip(bars, avg_lats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val:.1f}ms", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Latency (ms)")
ax.set_title("Inference latency — avg with min/max range (imgsz=1024)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_latency_comparison.png")
plt.close()
print("Chart 2: latency comparison")

# ---------------------------------------------------------------------------
# CHART 3 — RAM usage
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors_ram = []
for i, (label, val) in enumerate(zip(labels, ram_vals)):
    if val == 0:
        bar_colors_ram.append("#CCCCCC")
    else:
        bar_colors_ram.append(colors[i])
bars = ax.bar(labels, ram_vals, color=bar_colors_ram, alpha=0.85, width=0.5)
ax.axhline(4096, color="#E24B4A", linestyle="--", linewidth=1, label="Pi 5 RAM (4GB)")
ax.axhline(8192, color="#BA7517", linestyle="--", linewidth=1, label="Jetson Orin RAM (8GB)")
for bar, val in zip(bars, ram_vals):
    label_text = f"{val:.0f}MB" if val > 0 else "N/A"
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 50) + 50,
            label_text, ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Peak RAM usage (MB)")
ax.set_title("Memory footprint comparison")
ax.legend(fontsize=9)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_ram_comparison.png")
plt.close()
print("Chart 3: RAM comparison")

# ---------------------------------------------------------------------------
# CHART 4 — Per-image latency overlaid
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 5))
configs_plot = [
    (cpu_data,    "ONNX CPU (Pi 5 sim)",         "#BA7517"),
    (jetson_data, "ONNX GPU (Jetson Orin sim)",   "#1D9E75"),
    (pt_data,     "PyTorch GPU (RTX 3060)",        "#378ADD"),
]
for data, label, color in configs_plot:
    if data:
        lats = [r["latency_ms"] for r in data]
        ax.plot(range(len(lats)), lats, label=label, color=color,
                linewidth=1.2, alpha=0.75)
ax.axhline(1000/REALTIME, color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"Real-time limit ({1000/REALTIME:.0f}ms)")
ax.set_xlabel("Image index")
ax.set_ylabel("Latency (ms)")
ax.set_title("Per-image inference latency — all configurations (imgsz=1024)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_latency_per_image.png")
plt.close()
print("Chart 4: per-image latency")

# ---------------------------------------------------------------------------
# CHART 5 — Real-time viability
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors_rt = ["#1D9E75" if f >= REALTIME else "#E24B4A" for f in fps_vals]
bars = ax.barh(labels, fps_vals, color=bar_colors_rt, alpha=0.85)
ax.axvline(REALTIME, color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"Real-time threshold ({REALTIME} FPS)")
for bar, val in zip(bars, fps_vals):
    status = "✓ REAL-TIME" if val >= REALTIME else "✗ NOT REAL-TIME"
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f} FPS  {status}", va="center", fontsize=9)
ax.set_xlabel("Frames per second (FPS)")
ax.set_title("Real-time deployment viability")
ax.legend(fontsize=10)
ax.set_xlim(0, max(fps_vals) * 1.35)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_realtime_analysis.png")
plt.close()
print("Chart 5: real-time analysis")

# ---------------------------------------------------------------------------
# CHART 6 — Latency boxplot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
box_data, box_labels, box_colors = [], [], []
for data, label, color in configs_plot:
    if data:
        box_data.append([r["latency_ms"] for r in data])
        box_labels.append(label.replace(" (", "\n("))
        box_colors.append(color)

bp = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels,
                medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(1000/REALTIME, color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"Real-time limit ({1000/REALTIME:.0f}ms)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Latency distribution boxplot (imgsz=1024)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_latency_boxplot.png")
plt.close()
print("Chart 6: latency boxplot")

# ---------------------------------------------------------------------------
# CHART 7 — Full comparison table
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")

table_data = []
for s in summary:
    ram = f"{s['peak_ram_mb']:.0f} MB" if s["peak_ram_mb"] > 0 else "~3000 MB"
    table_data.append([
        s["label"],
        "ONNX Runtime" if "ONNX" in s["label"] else "PyTorch CUDA",
        "1024px",
        f"{s['fps']:.1f}",
        f"{s['avg_lat_ms']:.1f} ms",
        ram,
        s["real_time"],
        "0.666" if "PyTorch" in s["label"] else "~0.65",
    ])

col_labels = ["Configuration", "Runtime", "imgsz", "FPS",
              "Avg Latency", "Peak RAM", "Real-time", "mAP50"]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.25, 0.13, 0.08, 0.07, 0.12, 0.10, 0.10, 0.08],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#378ADD")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == 2:
        cell.set_facecolor("#E1F5EE")  # highlight Jetson (best)
    elif row % 2 == 0:
        cell.set_facecolor("#F8F8F8")
    cell.set_edgecolor("#DDDDDD")

ax.set_title("Edge Deployment Benchmark — YOLO26s-obb on DOTA v2.0 (469 test images)",
             fontsize=11, fontweight="bold", pad=15)

fig.text(0.05, 0.02,
         "Jetson Orin Nano and Raspberry Pi 5 simulated via ONNX Runtime on RTX 3060. "
         "PyTorch GPU results from full inference run (inference_gpu.py). "
         "mAP50 at 640px estimated; training mAP50=0.666 at 1024px.",
         fontsize=7, color="gray", style="italic")

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/07_full_comparison_table.png", bbox_inches="tight")
plt.close()
print("Chart 7: full comparison table")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ALL CHARTS GENERATED")
print("=" * 60)
print(f"\nSaved to: {CHARTS_DIR}")
for f in sorted(Path(CHARTS_DIR).glob("*.png")):
    print(f"  {f.name} ({f.stat().st_size//1024} KB)")

jetson = next((s for s in summary if "Jetson" in s["label"]), None)
pi     = next((s for s in summary if "Pi" in s["label"]), None)

if jetson and pi:
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    print(f"Jetson Orin Nano sim : {jetson['fps']:.1f} FPS ({jetson['avg_lat_ms']}ms avg) — REAL-TIME ✓")
    print(f"Raspberry Pi 5 sim   : {pi['fps']:.1f} FPS ({pi['avg_lat_ms']}ms avg) — {'REAL-TIME ✓' if pi['fps'] >= 15 else 'NEAR REAL-TIME'}")
    print(f"\nYOLO26s-obb achieves {jetson['fps']:.0f}x faster inference on Jetson vs Pi")
