"""
HISTORICAL DOTA BASELINE MATERIAL - not the RarePlanes ONNX parity workflow.

benchmark_edge.py — ONNX inference-speed study (edge-deployment PROXY)

IMPORTANT: this measures inference speed on the DESKTOP it runs on, using ONNX
Runtime execution providers as stand-ins for edge devices. It does NOT run on a
physical Jetson or Raspberry Pi. These are desktop-only proxy timings, not
device-specific measurements or performance bounds:
  - CUDAExecutionProvider  -> stands in for a Jetson-class GPU
  - CPUExecutionProvider   -> stands in for a Pi-class CPU (4 threads)

Edit the CONFIG paths below, then run:
  python benchmark_edge.py
"""



import onnxruntime as ort
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)

import time
import csv
import gc
import os
import cv2
import numpy as np
import psutil
from pathlib import Path


ONNX_MODEL  = "best.onnx"                  # exported ONNX model
TEST_IMAGES = "test_images"                # folder to benchmark
OUTPUT_DIR  = "results/edge_study"         # result CSVs
GPU_CSV     = "results/gpu/inference_summary.csv"  # from inference_gpu.py

IMGSZ       = 1024
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
NUM_WARMUP  = 5
NUM_IMAGES  = 469

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

img_files = sorted(
    list(Path(TEST_IMAGES).glob("*.png")) +
    list(Path(TEST_IMAGES).glob("*.jpg"))
)[:NUM_IMAGES]

print("=" * 60)
print("EDGE SIMULATION — Raspberry Pi 5 + Jetson Orin Nano")
print("=" * 60)
print(f"ONNX model:  {ONNX_MODEL}")
print(f"Images:      {len(img_files)} test images")
print(f"Image size:  {IMGSZ}px")
print(f"Confidence:  {CONF_THRESH}")
print()

def preprocess(img_path, imgsz):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (imgsz, imgsz))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)


def count_detections(output, conf_thresh):
    if output.ndim == 3:
        preds = output[0]
        if preds.shape[1] >= 6:
            confs = preds[:, -1] if preds.shape[1] == 7 else preds[:, 4]
            return int(np.sum(confs > conf_thresh))
    return 0


def run_onnx(provider_name, scenario_label, scenario_desc, threads=None):
    print(f"\n[{scenario_label}] {scenario_desc}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        sess_opts.intra_op_num_threads = threads
        sess_opts.inter_op_num_threads = 1

    session = ort.InferenceSession(
        ONNX_MODEL,
        sess_options=sess_opts,
        providers=[provider_name]
    )
    input_name = session.get_inputs()[0].name
    print(f"  Provider: {provider_name} ✓")


    dummy = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    for _ in range(NUM_WARMUP):
        session.run(None, {input_name: dummy})

    results = []
    proc = psutil.Process(os.getpid())

    for i, img_path in enumerate(img_files):
        img_input = preprocess(img_path, IMGSZ)
        start = time.perf_counter()
        output = session.run(None, {input_name: img_input})
        latency_ms = (time.perf_counter() - start) * 1000
        ram_mb = proc.memory_info().rss / (1024 * 1024)
        num_det = count_detections(output[0], CONF_THRESH)

        results.append({
            "image":      img_path.name,
            "latency_ms": round(latency_ms, 2),
            "ram_mb":     round(ram_mb, 1),
            "detections": num_det,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(img_files)}] {latency_ms:.1f}ms | RAM: {ram_mb:.0f}MB | Det: {num_det}")

    return results


def summarize(results, label):
    latencies = [r["latency_ms"] for r in results]
    rams      = [r["ram_mb"] for r in results]
    dets      = [r["detections"] for r in results]
    avg_lat   = sum(latencies) / len(latencies)
    return {
        "label":       label,
        "images":      len(results),
        "avg_lat_ms":  round(avg_lat, 1),
        "min_lat_ms":  round(min(latencies), 1),
        "max_lat_ms":  round(max(latencies), 1),
        "fps":         round(1000 / avg_lat, 2),
        "avg_ram_mb":  round(sum(rams) / len(rams), 1),
        "peak_ram_mb": round(max(rams), 1),
        "total_dets":  sum(dets),
        "real_time":   "YES" if (1000 / avg_lat) >= 15 else "NO",
    }



def load_gpu_baseline():
    gpu_latencies = []
    gpu_dets = []
    with open(GPU_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gpu_latencies.append(float(row["latency_ms"]))
            gpu_dets.append(int(row["detections"]))
    avg_lat = sum(gpu_latencies) / len(gpu_latencies)
    return {
        "label":       "PyTorch GPU (RTX 3060, 1024px FP16)",
        "images":      len(gpu_latencies),
        "avg_lat_ms":  round(avg_lat, 1),
        "min_lat_ms":  round(min(gpu_latencies), 1),
        "max_lat_ms":  round(max(gpu_latencies), 1),
        "fps":         round(1000 / avg_lat, 2),
        "avg_ram_mb":  0,
        "peak_ram_mb": 0,
        "total_dets":  sum(gpu_dets),
        "real_time":   "NO",
    }, gpu_latencies


if __name__ == "__main__":
    all_summaries    = []
    all_per_image    = {}


    cpu_results = run_onnx(
        provider_name  = "CPUExecutionProvider",
        scenario_label = "Scenario 1",
        scenario_desc  = "ONNX CPU — Raspberry Pi 5 simulation",
        threads        = 4,
    )
    cpu_summary = summarize(cpu_results, "ONNX CPU (Pi 5 sim, 1024px)")
    all_summaries.append(cpu_summary)
    all_per_image["cpu"] = cpu_results
    gc.collect()


    gpu_results = run_onnx(
        provider_name  = "CUDAExecutionProvider",
        scenario_label = "Scenario 2",
        scenario_desc  = "ONNX GPU — Jetson Orin Nano simulation",
    )
    gpu_summary = summarize(gpu_results, "ONNX GPU (Jetson Orin sim, 1024px)")
    all_summaries.append(gpu_summary)
    all_per_image["jetson_onnx"] = gpu_results
    gc.collect()


    pt_summary, pt_latencies = load_gpu_baseline()
    all_summaries.append(pt_summary)
    all_per_image["pytorch_gpu"] = [
        {"image": f"img_{i}", "latency_ms": l, "ram_mb": 0, "detections": 0}
        for i, l in enumerate(pt_latencies)
    ]
    print(f"\n[Scenario 3] PyTorch GPU baseline loaded from existing results")
    print(f"  {pt_summary['images']} images, avg {pt_summary['avg_lat_ms']}ms, {pt_summary['fps']} FPS")


    for name, results in all_per_image.items():
        csv_path = output_path / f"{name}_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "latency_ms", "ram_mb", "detections"])
            writer.writeheader()
            writer.writerows(results)


    summary_path = output_path / "comparison_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)


    print("\n" + "=" * 80)
    print("EDGE SIMULATION COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Config':<40} {'FPS':>6} {'Avg(ms)':>9} {'Min(ms)':>9} {'Peak RAM':>10} {'RT?':>5}")
    print("-" * 80)
    for s in all_summaries:
        ram = f"{s['peak_ram_mb']:.0f}MB" if s["peak_ram_mb"] > 0 else "N/A"
        print(f"{s['label']:<40} {s['fps']:>6.1f} {s['avg_lat_ms']:>9.1f} {s['min_lat_ms']:>9.1f} {ram:>10} {s['real_time']:>5}")
    print("=" * 80)
