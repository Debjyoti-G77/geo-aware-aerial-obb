"""
HISTORICAL DOTA BASELINE MATERIAL - not the RarePlanes geospatial workflow.

Inference Results Analysis — YOLO26s-obb

Reads inference_summary.csv and inference_detections.csv
and generates charts for analysis.

Charts generated:
  1. Detections per image (bar chart)
  2. Class distribution — all detections (horizontal bar)
  3. Confidence distribution (histogram)
  4. Latency per image (bar chart)
  5. Average confidence per class (bar chart)
  6. Box area distribution per class (boxplot)
  7. Detections vs image size (scatter)
  8. Summary stats table (saved as PNG)

Usage:
  python analyze_results.py
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RESULTS_DIR = "results/gpu"
CHARTS_DIR  = "results/charts"

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})
COLOR_PRIMARY   = "#378ADD"
COLOR_SECONDARY = "#1D9E75"
COLOR_ACCENT    = "#E24B4A"
COLOR_AMBER     = "#BA7517"

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

summary_path    = Path(RESULTS_DIR) / "inference_summary.csv"
detections_path = Path(RESULTS_DIR) / "inference_detections.csv"

if not summary_path.exists():
    print(f"ERROR: {summary_path} not found. Run inference_gpu.py first.")
    exit(1)

summary    = []
detections = []

with open(summary_path, "r", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))

with open(detections_path, "r", encoding="utf-8") as f:
    detections = list(csv.DictReader(f))

print(f"Loaded {len(summary)} images, {len(detections)} detections")

# Parse types
for r in summary:
    r["latency_ms"] = float(r["latency_ms"])
    r["detections"] = int(r["detections"])
    r["width_px"]   = int(r["width_px"])
    r["height_px"]  = int(r["height_px"])

for d in detections:
    d["confidence"]   = float(d["confidence"])
    d["box_area_px"]  = float(d["box_area_px"])

images     = [r["image"] for r in summary]
latencies  = [r["latency_ms"] for r in summary]
det_counts = [r["detections"] for r in summary]
img_areas  = [r["width_px"] * r["height_px"] for r in summary]

all_classes   = [d["class_name"] for d in detections]
all_confs     = [d["confidence"] for d in detections]
all_areas     = [d["box_area_px"] for d in detections]

# Class counts
class_counts = defaultdict(int)
for c in all_classes:
    class_counts[c] += 1
class_counts = dict(sorted(class_counts.items(), key=lambda x: -x[1]))

# Short image names for x-axis
short_names = [img.replace(".png","").replace(".jpg","") for img in images]

print(f"Classes detected: {list(class_counts.keys())}")
print(f"Total detections: {len(detections)}")

# ---------------------------------------------------------------------------
# CHART 1 — Detections per image
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(short_names, det_counts, color=COLOR_PRIMARY, alpha=0.85)
for bar, val in zip(bars, det_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Image")
ax.set_ylabel("Number of detections")
ax.set_title("Detections per test image")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_detections_per_image.png")
plt.close()
print("Chart 1 saved: detections per image")

# ---------------------------------------------------------------------------
# CHART 2 — Class distribution
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, max(5, len(class_counts) * 0.4)))
classes = list(class_counts.keys())
counts  = list(class_counts.values())
colors  = [COLOR_PRIMARY if c % 2 == 0 else COLOR_SECONDARY for c in range(len(classes))]
bars = ax.barh(classes[::-1], counts[::-1], color=colors[::-1], alpha=0.85)
for bar, val in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9)
ax.set_xlabel("Total detections")
ax.set_title("Class distribution across all test images")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_class_distribution.png")
plt.close()
print("Chart 2 saved: class distribution")

# ---------------------------------------------------------------------------
# CHART 3 — Confidence distribution (histogram)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(all_confs, bins=30, color=COLOR_PRIMARY, alpha=0.8, edgecolor="white")
ax.axvline(np.mean(all_confs), color=COLOR_ACCENT, linestyle="--",
           label=f"Mean: {np.mean(all_confs):.3f}")
ax.axvline(np.median(all_confs), color=COLOR_AMBER, linestyle="--",
           label=f"Median: {np.median(all_confs):.3f}")
ax.set_xlabel("Confidence score")
ax.set_ylabel("Number of detections")
ax.set_title("Confidence score distribution")
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_confidence_distribution.png")
plt.close()
print("Chart 3 saved: confidence distribution")

# ---------------------------------------------------------------------------
# CHART 4 — Latency per image
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(short_names, latencies, color=COLOR_SECONDARY, alpha=0.85)
ax.axhline(np.mean(latencies), color=COLOR_ACCENT, linestyle="--",
           label=f"Mean: {np.mean(latencies):.1f}ms")
for bar, val in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xlabel("Image")
ax.set_ylabel("Latency (ms)")
ax.set_title("Inference latency per image (GPU)")
ax.legend()
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_latency_per_image.png")
plt.close()
print("Chart 4 saved: latency per image")

# ---------------------------------------------------------------------------
# CHART 5 — Average confidence per class
# ---------------------------------------------------------------------------

class_conf = defaultdict(list)
for d in detections:
    class_conf[d["class_name"]].append(d["confidence"])

cls_names   = sorted(class_conf.keys(), key=lambda c: -np.mean(class_conf[c]))
cls_means   = [np.mean(class_conf[c]) for c in cls_names]
cls_stds    = [np.std(class_conf[c]) for c in cls_names]

fig, ax = plt.subplots(figsize=(10, max(5, len(cls_names) * 0.45)))
colors  = [COLOR_SECONDARY if m >= 0.5 else COLOR_AMBER for m in cls_means]
bars = ax.barh(cls_names[::-1], cls_means[::-1], color=colors[::-1],
               xerr=cls_stds[::-1], capsize=3, alpha=0.85)
ax.axvline(0.5, color=COLOR_ACCENT, linestyle="--", alpha=0.6, label="Conf=0.5")
for bar, val in zip(bars, cls_means[::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)
ax.set_xlabel("Mean confidence score")
ax.set_title("Average confidence per detected class")
ax.set_xlim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_avg_confidence_per_class.png")
plt.close()
print("Chart 5 saved: avg confidence per class")

# ---------------------------------------------------------------------------
# CHART 6 — Detection count per class per image (heatmap)
# ---------------------------------------------------------------------------

all_cls_list = sorted(class_counts.keys())
heatmap_data = np.zeros((len(all_cls_list), len(images)))

for d in detections:
    img_idx = images.index(d["image"])
    cls_idx = all_cls_list.index(d["class_name"])
    heatmap_data[cls_idx, img_idx] += 1

fig, ax = plt.subplots(figsize=(max(10, len(images)*0.6), max(6, len(all_cls_list)*0.5)))
im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(short_names)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(all_cls_list)))
ax.set_yticklabels(all_cls_list, fontsize=8)
plt.colorbar(im, ax=ax, label="Detection count")
ax.set_title("Detection heatmap — classes vs images")
# Annotate cells
for i in range(len(all_cls_list)):
    for j in range(len(images)):
        val = int(heatmap_data[i, j])
        if val > 0:
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6,
                    color="white" if val > heatmap_data.max()*0.6 else "black")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_detection_heatmap.png")
plt.close()
print("Chart 6 saved: detection heatmap")

# ---------------------------------------------------------------------------
# CHART 7 — Detections vs image area (scatter)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    [a / 1e6 for a in img_areas], det_counts,
    c=latencies, cmap="YlOrRd", s=80, alpha=0.85, edgecolors="white"
)
plt.colorbar(scatter, ax=ax, label="Latency (ms)")
for i, name in enumerate(short_names):
    ax.annotate(name, (img_areas[i]/1e6, det_counts[i]),
                textcoords="offset points", xytext=(5, 3), fontsize=6)
ax.set_xlabel("Image area (megapixels)")
ax.set_ylabel("Number of detections")
ax.set_title("Detections vs image area (color = latency)")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/07_detections_vs_area.png")
plt.close()
print("Chart 7 saved: detections vs image area")

# ---------------------------------------------------------------------------
# CHART 8 — Summary statistics table
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

stats = [
    ["Total images",               str(len(summary))],
    ["Total detections",           str(len(detections))],
    ["Avg detections/image",       f"{np.mean(det_counts):.1f}"],
    ["Max detections in one image",f"{max(det_counts)}"],
    ["Classes detected",           str(len(class_counts))],
    ["Most common class",          f"{list(class_counts.keys())[0]} ({list(class_counts.values())[0]})"],
    ["Avg GPU latency",            f"{np.mean(latencies):.1f} ms"],
    ["Min GPU latency",            f"{min(latencies):.1f} ms"],
    ["Max GPU latency",            f"{max(latencies):.1f} ms"],
    ["GPU throughput",             f"{1000/np.mean(latencies):.2f} img/sec"],
    ["Avg confidence",             f"{np.mean(all_confs):.3f}"],
    ["Min confidence",             f"{min(all_confs):.3f}"],
    ["Max confidence",             f"{max(all_confs):.3f}"],
    ["Model",                      "YOLO26s-obb (epoch 43)"],
    ["Training dataset",           "DOTA v2.0 (18 classes)"],
    ["Training mAP50",             "0.666"],
]

table = ax.table(
    cellText=stats,
    colLabels=["Metric", "Value"],
    cellLoc="left",
    loc="center",
    colWidths=[0.5, 0.5],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#378ADD")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("#DDDDDD")

ax.set_title("Inference Summary — YOLO26s-obb on DOTA v1.5 test images",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/08_summary_table.png", bbox_inches="tight")
plt.close()
print("Chart 8 saved: summary table")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ALL CHARTS GENERATED")
print("=" * 60)
print(f"Charts saved to: {CHARTS_DIR}")
print("\nFiles:")
for f in sorted(Path(CHARTS_DIR).glob("*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.0f} KB)")
