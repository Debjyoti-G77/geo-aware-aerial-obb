# Technical Methodology

This document records the experimental boundaries and engineering decisions
behind the Geo-Aware Aerial Object Detection project. It separates model
quality evaluation from backend parity, and it distinguishes observed results
from claims that have not been tested.

## 1. Scope and provenance

The project uses the MIT-licensed DOTA OBB detector baseline from
[JustinVladut/YOLO26-DOTA-v2](https://github.com/JustinVladut/YOLO26-DOTA-v2)
as its starting point. Its pretrained `best.pt` is an 18-class oriented
detector whose class metadata includes `plane`. This project adds:

- windowed inference over georeferenced RarePlanes COG scenes;
- preservation of OBB vertices and tile provenance;
- source-pixel and geographic coordinate recovery;
- global, class-aware polygon NMS;
- GIS-ready GeoJSON output;
- TRAIN-only RarePlanes dataset preparation and domain adaptation;
- frozen internal validation and PyTorch/ONNX backend parity checks.

This combined source project is distributed under AGPL-3.0. The original
detector baseline remains attributed to JustinVladut under its preserved MIT
notice in `LICENSES/YOLO26-DOTA-v2-MIT.txt`. The geospatial, RarePlanes,
evaluation, reproducibility, and deployment extensions were contributed by
Debjyoti Sengupta, 2026; that statement does not claim ownership of baseline or
third-party material.

Artifact identity is recorded by SHA-256:

| Artifact | SHA-256 |
|---|---|
| Original DOTA `best.pt` | `5102e1493e3d1f2d0bb331d246e060badb1709f405cd17cd4a558cac9faefbef` |
| RarePlanes `rareplanes_finetuned_best.pt` | `4111cdab67bcc3c53623e4bd08aeb5943e4a49225c27c62172036b815991eced` |
| Exported ONNX model | `09ae20574cddb0f69b36bbdbb0c11eaec8d12ed640b3515f65c8f6e726d6331b` |

Weights and datasets are not intended for source control.

## 2. Data boundary and internal split

Only the real-imagery RarePlanes TRAIN split was used for implementation,
diagnosis, fine-tuning, and reported internal validation. The official
RarePlanes TEST split was not used for model selection, threshold tuning,
fine-tuning, or the results in this repository.

The internal validation set was frozen before fine-tuning. Selection was
deterministic, metadata-only, and restricted to locations not previously used
in the earlier 24-scene diagnostic work. The split is location-disjoint:

- training: 80 `loc_id` values, 170 scenes, 9,632 source aircraft;
- validation: 12 `loc_id` values, 17 scenes, 1,268 source aircraft;
- validation IDs: 7, 10, 23, 25, 27, 31, 36, 42, 79, 84, 101, 108;
- training/validation location overlap: 0;
- training/validation scene overlap: 0.

Training may include previously analysed TRAIN locations. Validation does not.
This avoids selecting the new validation set after observing performance on
those locations. The union of the 23 previously analysed TRAIN location IDs is
frozen in `src/prepare_rareplanes_obb.py`. When the original ignored diagnostic
manifests are present they are validated against that frozen union; a clean
clone does not depend on those generated outputs.

## 3. RarePlanes geometry and training labels

An official RarePlanes aircraft geometry is a four-point diamond ordered nose,
left wing, tail, right wing. This shape is preserved in the source annotations
and generated provenance. It is not overwritten.

Ultralytics OBB training represents a four-corner rotated rectangle. For
training, each source diamond is transformed to a Shapely minimum-area rotated
rectangle in source-pixel coordinates. The rectangle corners are normalized
to tile width and height and written as:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are finite, normalized to `[0, 1]`, and retain floating-point
precision. The one training class is `0: plane`.

The preparation pipeline uses deterministic 1024 x 1024 overlapping windows.
Overlap is 0.20 and the integer stride is 819. Each aircraft is assigned to
exactly one eligible tile. Eligibility requires the complete derived OBB to be
visible within the tile; boxes are not clipped into validity. If more than one
tile is eligible, deterministic ownership resolves duplication. Empty chips
are sampled deterministically at a bounded ratio rather than retaining every
background window.

Generated inventory:

| Subset | Tiles | Positive | Background | Assigned instances | Source GT | Omitted |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,751 | 2,201 | 550 | 9,614 | 9,632 | 18 |
| Validation | 348 | 279 | 69 | 1,265 | 1,268 | 3 |

The omitted objects had no tile in which the complete derived rectangle was
visible. Manifests retain source scene, location, tile offsets, source diamond,
derived OBB, visibility, assignment reason, weather, and acquisition
provenance. Generated chips and manifests remain ignored artifacts.

## 4. Full-scene tiling

For dimension `D`, tile size `T`, and overlap `o`, the nominal stride is:

```text
stride = floor(T * (1 - o) + 0.5)
```

For `T=1024` and `o=0.20`, this is 819. Offsets start at zero and advance by
the stride. When a nominal final window would leave an uncovered strip, its
offset is anchored to `D - T`. Duplicate offsets are removed. Rasters smaller
than the tile size retain their actual valid region and may be padded only for
model input. Padding is explicitly represented and is never treated as source
imagery.

Rasterio reads one window at a time. The model sees a contiguous HWC/BGR array
derived from the source CHW/RGB bands. Tiles are processed sequentially on CPU
for the validated local pipeline, and per-tile prediction objects are not
accumulated after their detections have been converted.

## 5. OBB coordinate recovery

Ultralytics OBB detections use the pixel-space `obb.xyxyxyxy` four-corner
property. Coordinates remain floating point.

For a tile whose upper-left source offset is `(col_off, row_off)`, each local
vertex `(x, y)` becomes:

```text
global_col = x + col_off
global_row = y + row_off
```

Model polygon vertices are continuous pixel-space coordinates, so the raster
affine is applied directly. An additional half-pixel shift, appropriate for
the center of an integer pixel index, is not added to a continuous OBB vertex.

The source raster CRS and affine are read from the file. If reprojection is
required, PyProj uses `always_xy=True`. Output is normalized to OGC:CRS84 with
coordinate order `(longitude, latitude)`. RarePlanes PS-RGB COGs used here
report EPSG:4326; their values are numerically compatible with the official
CRS84 annotations when longitude/latitude order is preserved. The explicit
axis-order handling avoids relying on implicit EPSG:4326 axis semantics.

## 6. Boundary validity and cross-tile duplicates

The production boundary policy accepts a detection only when all four local
vertices fall within the tile's real source-data region. It does not clip
invalid OBBs. Rejected candidates retain reconstructable geometry and a reason
for diagnostics, but never enter final predictions or official metrics.

This policy is deliberately conservative. Offline TRAIN-only analysis showed
that some valid rotated OBBs cross internal tile boundaries even when their
centroids are inside. Alternative centroid and visible-area policies were
studied, but an independent location-disjoint TRAIN holdout did not reproduce
the initial F1 improvement. Production therefore retains the original strict
policy.

Remaining plane candidates are reconciled in full-scene pixel space:

1. group by class and sort by descending confidence;
2. keep the highest-confidence polygon;
3. compare only polygons with overlapping spatial extents;
4. suppress same-class polygons with true polygon IoU greater than the global
   threshold (0.50);
5. retain the kept/suppressed IDs, tile IDs, confidences, and IoU.

This is Shapely intersection-over-union on oriented polygons, not axis-aligned
box IoU. Suppressed candidates remain diagnostic records and cannot affect the
final feature collection or evaluation counts.

## 7. DOTA-to-RarePlanes domain diagnosis

The original DOTA model's strongest observed domain failure was associated
with snow. Across the earlier 24-scene TRAIN analysis, snow recall was 0.0346
(19/549) versus 0.6735 (689/1,023) for non-snow aircraft. Size, raster
corruption, band order, CRS/affine metadata, incomplete coverage, off-nadir
association, boundary filtering, and NMS did not explain the main failure.

A bounded low-confidence probe used exactly three TRAIN scenes, confidence
0.05, and all 18 DOTA classes. On snow scene `96_10400100096C2500`:

- GT aircraft: 399;
- raw all-class responses: 1,943;
- deduplicated all-class responses: 1,435;
- raw/deduplicated plane responses: 6/3;
- GT with plane IoU at least 0.50: 2;
- GT with no overlapping any-class response: 385/399.

This supports absent model response as the main observed failure, rather than
merely the production confidence threshold or confusion with another DOTA
class. The probe used TRAIN scenes and is diagnostic, not a generalization
estimate.

## 8. Fine-tuning protocol and provenance

The first adaptation intentionally used a simple natural-distribution
baseline: no snow oversampling, custom normalization, domain-specific
photometric processing, custom loss, or confidence changes.

Recipe:

- source checkpoint: 18-class DOTA `best.pt`;
- target: one-class RarePlanes OBB (`plane`);
- Ultralytics: 8.4.165;
- image size: 1024; batch: 8; maximum epochs: 100;
- optimizer: AdamW; `lr0=0.002`; `lrf=0.01`; momentum 0.9;
- weight decay 0.0005; warmup 3 epochs; cosine LR disabled;
- patience: 20; seed: 20261001; deterministic: true;
- workers: 8; AMP enabled; no frozen layers;
- mosaic 1.0, closed for the final 10 epochs; mixup/cutmix/copy-paste 0;
- HSV 0.015/0.7/0.4, horizontal flip 0.5, vertical flip 0;
- translate 0.1, scale 0.5, degrees/shear/perspective 0.

The real run executed on a Colab NVIDIA Tesla T4. It completed 10 epochs before
a power/runtime interruption. A healthy `last.pt` was recovered with optimizer
state intact. Only `train_args.patience` was changed to 20 in that recovery
checkpoint, and explicit resume continued at epoch 11. Early stopping ended at
epoch 79; epoch 59 was selected as best.

Historical resume metadata is provenance only. The portable training launcher
defaults to `resume: false` and never automatically resumes that historical
run. A new user starts an equivalent fresh run from the source checkpoint.

Ultralytics' best chip-level validation metrics were precision 0.892, recall
0.976, mAP50 0.959, and mAP50-95 0.815.

## 9. Frozen full-scene evaluation

Model quality was evaluated on all 17 frozen, location-disjoint internal
validation scenes. Both models used the same settings:

- tile/image size: 1024; overlap: 0.20; stride: 819;
- model confidence: 0.25; model IoU: 0.45;
- strict all-vertices-inside boundary policy;
- global polygon-NMS IoU: 0.50;
- greedy one-to-one confidence-descending matching at polygon IoU 0.50;
- normalized minimum-area-rectangle GT for official OBB metrics.

Micro metrics aggregate TP, FP, and FN across scenes before computing the
ratios. Macro metrics average scene-level ratios under the explicit
zero-denominator policy in the evaluation code.

| Model | GT | Predictions | TP | FP | FN | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original DOTA | 1,268 | 1,258 | 857 | 401 | 411 | 0.6812 | 0.6759 | 0.6785 | 0.6527 | 0.6371 | 0.6344 |
| Fine-tuned | 1,268 | 1,287 | 1,251 | 36 | 17 | 0.9720 | 0.9866 | 0.9793 | 0.9487 | 0.9856 | 0.9643 |

The validation set contains only one snow scene with 17 aircraft. Its 17/17
fine-tuned result is descriptive only. It is insufficient to claim reliable
snow generalization. Official RarePlanes TEST performance remains unknown.

## 10. ONNX export and parity

The deployment exporter creates a fixed input model with batch 1, three RGB
channels, 1024 x 1024 spatial dimensions, opset 18, static shapes, no embedded
NMS, no quantization, and raw OBB output. Structural validation uses the ONNX
checker and ONNX Runtime CPU.

Single-chip parity on one internal validation chip produced 77 PyTorch and 77
ONNX detections. All 77 matched, with zero unmatched detections, mean confidence
difference `1.413e-7`, and mean polygon IoU `0.999999134`.

A full-scene parity check used frozen validation scene
`7_104001002D43CF00`, chosen solely because its 2994 x 3153 dimensions yielded
the lowest tile count (16). It was not selected by detection performance.

| Stage | PyTorch | ONNX Runtime |
|---|---:|---:|
| Raw detections | 25 | 25 |
| Boundary rejections | 1 | 1 |
| Pre-NMS | 24 | 24 |
| NMS suppressions | 1 | 1 |
| Final | 23 | 23 |
| Runtime | 20.255 s | 16.071 s |

All 23 final polygons matched. Mean/minimum polygon IoU was
`0.999997751` / `0.999969981`; mean/max confidence difference was
`3.162e-7` / `1.848e-6`; and mean/max center difference was
`0.0000122` / `0.0001122` pixels. Both outputs had identical CRS semantics,
finite valid geometries, and no footprint violations.

These parity checks establish numerical and geospatial equivalence for the
tested inputs. They are separate from model accuracy. The observed ONNX CPU
runtime was about 1.26x faster for one scene and is not an optimized benchmark.
TensorRT was not evaluated because the local machine has no dedicated GPU.

## 11. GeoJSON semantics and GIS inspection

The exporter writes an RFC 7946-compatible FeatureCollection with no legacy
GeoJSON `crs` member. Each final geometry is a closed Polygon whose coordinate
order is longitude, latitude in OGC:CRS84.

Feature properties retain:

- detection, scene, class, and source tile identifiers;
- confidence and model name/task;
- tile offsets and tile bounds in source pixels;
- local and full-scene pixel OBB vertices;
- source and output CRS;
- inference confidence, model IoU, tile size/overlap, and global NMS IoU.

Per-scene output also includes an inference summary, duplicate-suppression
diagnostics, and rejected-candidate diagnostics. In QGIS, load
`predictions.geojson` with **Layer > Add Layer > Add Vector Layer**, then add
the source COG as a raster. Polygon orientation, confidence, tile provenance,
scene-edge behavior, and suppression decisions can then be inspected in their
geographic context.

## 12. Reproducibility boundaries

The known-good environment is Python 3.12.14, Ultralytics 8.4.165, PyTorch
2.14.0+cpu, torchvision 0.29.0+cpu, NumPy 2.5.3, Rasterio 1.5.1 with GDAL
3.13.3, PyProj 3.8.0, Shapely 2.1.2, OpenCV 5.0.0.93, ONNX 1.23.1, and ONNX
Runtime 1.30.0.

On Windows, the geospatial native stack is most reliably managed through
conda-forge/Miniforge. Installing a standalone `gdal` wheel through pip is not
part of the validated setup. Repository configuration and commands use paths
relative to the project root; historical Colab and Google Drive paths are not
runtime requirements.

`environment.yml` records the concise known-tested Windows CPU environment.
It pins Python and the native GIS packages through conda-forge and pins the
validated CPU PyTorch, Ultralytics, OpenCV, and ONNX packages through pip. It is
not a machine-specific transitive lock and contains no local prefix. The broad
`requirements.txt` remains available for general environments. Full training
requires a compatible CUDA PyTorch environment; the CPU reference environment
is for preparation, testing, inference, and ONNX Runtime validation.

RarePlanes source data and checkpoints are external ignored assets. The
[official AWS Registry entry](https://registry.opendata.aws/rareplanes/)
identifies `s3://rareplanes-public/` as the public dataset bucket and CC BY-SA
4.0 as the license. Users can inspect it without an AWS account using
`aws s3 ls --no-sign-request s3://rareplanes-public/`. The preparation pipeline
expects official real TRAIN PS-RGB COGs, aircraft GeoJSON, and
`RarePlanes_Public_Metadata.csv` in the repository-relative layout shown in
the README. No unverified sync subpath or TEST acquisition command is given.
Public checkpoint hosting has not yet been assigned and must be supplied
separately from the source repository. The project's AGPL-3.0 source license
does not itself relicense external data, dependencies, or model binaries;
their applicable terms are summarized in `THIRD_PARTY_NOTICES.md`.

The repository also retains historical DOTA baseline evidence. `src/train.py`
and `src/resume.py` are import-safe baseline utilities and are not part of the
RarePlanes fine-tuning path. The older benchmark/chart scripts, `results/`,
`examples/`, and visual assets document the DOTA baseline and desktop
proxy edge study. The current deployment evidence is specifically the
RarePlanes fixed-shape ONNX Runtime CPU parity analysis in Section 10.

The principal remaining validation gap is independent evaluation on untouched
official TEST data (under its permitted evaluation process), especially across
more snow locations. Until then, results should be described as frozen
location-disjoint internal TRAIN validation, not official RarePlanes benchmark
performance.
