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
- frozen internal validation and PyTorch/ONNX backend parity checks;
- metric-CRS multi-temporal spatial-presence analysis;
- a shared FastAPI inference service and non-root CPU container deployment;
- locally validated asset-free CI (hosted Actions execution pending publication).

The YOLO26 detector architecture is not an original contribution of this project.

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

Mean matched rectangular-OBB IoU was 0.5469 for the original checkpoint and
0.8367 after adaptation; micro F1 improved by 0.3007 (computed before rounding).
Mean matched IoU excludes
unmatched objects and is not an all-GT coverage metric. These are internal
validation results: this split also governed training early stopping, so it is
not an independent final generalization test.

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

## 12. Multi-temporal aircraft activity analysis

The optional temporal layer consumes final per-scene prediction GeoJSON for
two or more timestamped acquisitions of the same location. The first frozen
internal-VAL demonstration location is selected without detector results:
eligible inputs must be RarePlanes TRAIN scenes from one frozen validation
location, have authoritative acquisition timestamps and valid georeferencing,
and contain at least two acquisitions. Selection maximizes acquisition count,
then temporal span, with location ID only as a deterministic final tie-break.

Geographic polygons remain in CRS84 for output, but all matching geometry is
projected with `always_xy=True` into the local UTM zone (or a local azimuthal
equidistant fallback outside UTM latitude coverage). Consecutive timestamps
are associated by deterministic global greedy one-to-one assignment. Eligible
pairs must be within a maximum metric distance, and their transparent cost is:

```text
0.70 * (1 - projected polygon IoU)
  + 0.30 * (centroid distance / maximum correspondence distance)
```

Unless explicitly supplied, the maximum distance is the larger of 50 metres
or twice the median detected OBB major axis. The persistent-position boundary
is the larger of 15 metres or half the median major axis, capped by the maximum
distance. These conservative, geometry-scaled values are configurable analysis
parameters; they were not tuned against RarePlanes ground truth and they do
not change detector confidence, model IoU, boundary filtering, or global NMS.

Matched presence is described as `persistent` or `displaced`; unmatched later
and earlier detections are described as `appeared` and `disappeared`
respectively. For three or more acquisitions, a simple 100-metre metric grid
aggregates detection-centroid events for interpretable GIS inspection. This is
descriptive spatial activity analysis, not a statistical hotspot test.

**This workflow measures changes in detected spatial aircraft presence; it
does not establish physical aircraft identity across acquisitions.** Results
inherit detector false positives, false negatives, georegistration error,
viewing-condition changes, and the ambiguity of spatially dense parking areas.

Outputs are RFC 7946 GeoJSON for QGIS: all timestamp detections,
correspondence LineStrings, appeared/disappeared polygons, and an optional
activity grid, plus JSON and CSV summaries. Compatible prior scene inference
is reused only when its summary proves the frozen production policy; missing
selected scenes require the externally supplied checkpoint with its verified
SHA-256 and are run through the unchanged production inference pipeline.

### Frozen internal-validation demonstration

The metadata-only rule selected location 84 from four repeated-location
candidates: locations 10, 23, 84, and 108 had 2, 2, 3, and 2 acquisitions.
Location 84's authoritative timestamps were 2017-01-30 16:12 UTC,
2019-01-14 16:35 UTC, and 2019-03-10 16:11 UTC, spanning 66,441,540 seconds
(about 769 days). All three pre-existing inference outputs were reused after
their summaries and paired-run provenance confirmed the fine-tuned checkpoint,
fixed inference settings, and strict production boundary policy.

The local metric CRS was EPSG:32617. The detected median OBB major axis was
18.418 m, so the untuned geometry-derived floors governed: maximum
correspondence distance 50 m and persistent-position distance 15 m. Counts
were `149 -> 166 -> 188`. The first pair produced 103 correspondences
(74 persistent, 29 displaced), 63 appeared presences, and 46 disappeared
presences; mean/median/max displacement was 12.685/5.728/49.877 m. The second
pair produced 116 correspondences (95 persistent, 21 displaced), 72 appeared,
and 50 disappeared; displacement was 9.937/6.184/47.829 m. Mean projected
polygon IoU was 0.371 and 0.353 respectively. Both pair-wise count-conservation
identities held.

The descriptive 100 m activity grid contained 130 occupied cells. Its highest
observed cell contained 29 detection events across all three timestamps. All
503 detection polygons, 219 correspondence lines, 231 appeared/disappeared
polygons, and 130 grid polygons passed finite-coordinate, geometry-type,
validity, property-provenance, and CRS84 range checks.

These figures are **model-derived temporal spatial analysis, not
identity-ground-truth evaluation**. They do not establish that corresponding
presences are the same physical aircraft or that appeared/disappeared presence
represents a confirmed arrival or departure.

## 13. FastAPI geospatial inference service

The optional service layer in `src/api_service.py` calls the same
`load_obb_model` and `process_raster_scene` functions as the command-line
workflow. The service therefore preserves the validated tile generation,
boundary rejection, full-scene coordinate reconstruction, polygon NMS, CRS84
conversion, and GeoJSON serialization instead of maintaining a second model
pipeline.

Configuration is read at process startup from `GEO_OBB_MODEL_PATH`, optional
`GEO_OBB_MODEL_SHA256`, `GEO_OBB_BACKEND`, `GEO_OBB_DEVICE`, and
`GEO_OBB_MAX_UPLOAD_BYTES`. The model is loaded lazily on the first readiness,
metadata, or inference request and then reused. Initialization and inference
have separate locks; model/GDAL inference is deliberately serialized because
this milestone validates correctness for a local service, not concurrent
throughput. Blocking raster validation and inference run in the ASGI thread
pool, allowing the event loop to continue serving requests. Both `.pt` and
`.onnx` backends are accepted through the existing
Ultralytics loader when their metadata describes a compatible OBB model with
the requested target class.

`/health/live` is independent of model availability. `/health/ready` verifies
the configured file, optional SHA-256, OBB metadata, and target class.
`/model` returns only a filename, digest, backend, class metadata, and frozen
settings; it does not expose the absolute checkpoint path. `/infer` accepts a
multipart GeoTIFF upload and returns scene metadata, the existing inference
summary counts/timing, and the final RFC 7946-style FeatureCollection.

Uploads are restricted to `.tif`/`.tiff` and known TIFF media types, streamed
in bounded chunks under a server-generated temporary filename, capped at 512
MiB by default, validated as three-band uint8 GeoTIFFs with a non-identity
affine transform and CRS, and removed with the entire temporary workspace when
the request finishes. Client filenames are used only as sanitized response
metadata. Expected failures are mapped to classified HTTP errors without
tracebacks or internal filesystem paths.

The upload limit is enforced while copying Starlette's parsed `UploadFile`
into the request workspace. Multipart parsing can spool input before this
application-level check; production deployment also needs an ingress body
limit and temporary-storage quota. This limit alone does not bound raster
dimensions or inference duration.

### Local service parity check

One frozen internal TRAIN-derived validation scene,
`108_1040010049B46C00` (4364 x 3360), was processed through both the existing
CLI and the loopback API using the verified fine-tuned checkpoint and identical
frozen settings. Both processed 24 windows and returned 171 raw plane
candidates, 5 boundary rejections, 166 pre-NMS candidates, 34 suppressions, and
132 final polygons. Polygon coordinates and confidences matched exactly, and
all inference counts matched. All 132 API geometries passed the shared
finite-coordinate, validity, CRS84 range, confidence, and footprint checks.
The API deliberately uses a server-controlled `input` scene stem, so scene,
detection, and source-tile identifiers differ from the CLI's original scene
stem; the sanitized client filename is retained separately. Other feature
properties matched. API scene processing took 23.26 seconds after readiness
had loaded the model; the CLI's 74.07-second total included model loading and
had different initialization/cache conditions. These timings are smoke-test
observations, not a service-versus-CLI performance claim. No accuracy tuning
or official RarePlanes TEST access was involved.

The API is intentionally scoped to loopback/local portfolio deployment. A
public production deployment still needs authentication and authorization,
TLS termination, rate limiting, request/process isolation, resource limits,
observability, health supervision, and a deliberate multi-worker model-memory
strategy. The temporal analysis module remains an offline workflow; a future
`POST /temporal-analysis` could wrap it after defining its multi-file upload
and long-running job semantics.

## 14. Container deployment and CI

The Dockerfile uses a digest-pinned `mambaorg/micromamba:2.9.0` Debian 13 base.
`deploy/environment.container.yml` keeps Python 3.12 and GDAL/Rasterio,
PyProj, Shapely, and NumPy together under conda-forge. Explicit CPU PyTorch
and the application dependencies are recorded separately in
`deploy/requirements.container.txt`. These are direct reference pins, not a
complete transitive lock. The existing `environment.yml` remains the tested
Windows reference. Micromamba's inherited entrypoint activates the environment
for commands; build-time activation is explicitly enabled as described in the
[Micromamba container documentation](https://micromamba-docker.readthedocs.io/en/stable/quick_start.html).

The runtime runs as `mambauser`, uses one Uvicorn worker, and listens on
`0.0.0.0:${PORT}` inside the container. `deploy/serve.py` validates the numeric
port without shell interpolation. The stdlib-only Docker healthcheck probes
`/health/live` independently of checkpoint readiness. Models are never loaded
at application import, so a model-free container can start normally and return
HTTP 503 from `/health/ready`.

The strict `.dockerignore` removes Git state, credentials/configuration,
models, original/generated datasets, outputs, caches, environments, and
historical visual artifacts from the build context. Explicit `COPY` statements
provide an additional inclusion boundary. The runtime image contains source,
configuration, documentation, and dependency libraries; tests are included
only in the separate `test` target. No checkpoint is downloaded by the build.

The Docker-reported runtime size was about 3.44 GB before final hardening and
about 3.20 GB after safe package-cache cleanup. `micromamba clean --all` had
retained roughly 551 MB of logical extracted-cache files, many hardlinked to
the installed environment. The build now verifies that no environment symlink
depends on `/opt/conda/pkgs`, then removes that cache in the installation layer
with `--force-pkgs-dirs`. The resulting image has no retained conda package
cache; pip already uses `--no-cache-dir`. Logical cache bytes are not one-for-one
image savings because of hardlinks and layer representation.

The remaining footprint is principally the scientific stack: the installed
Torch package is about 766 MB, Polars' runtime about 180 MB, OpenCV's Python
module plus bundled libraries about 196 MB, and non-site-package conda libraries
about 344 MB. System OpenGL/GLib dependencies also contribute. Project files
are only about 1.7 MB and runtime tests are not copied. The small TestClient
dependency is intentionally shared with the test target; removing required
libraries or replacing the validated native stack solely for size was avoided.
Core GIS/ML versions remained identical after cleanup; imports, unit tests,
and no-model startup checks validate the rebuilt environment without repeating
the expensive already-validated scene inference.

`compose.yaml` mounts only the user-specified model file read-only, rejects a
missing host file rather than creating a directory, publishes the API on host
loopback, drops Linux capabilities, enables `no-new-privileges`, and uses a
read-only root filesystem with a bounded writable `/tmp`. The host imagery
is submitted as an HTTP upload and is not mounted into the service. Larger
uploads may need a reviewed temporary-memory quota; the multipart parser and
application workspace can both consume temporary storage.

GitHub Actions runs on pull requests and pushes to `main`. It builds the
`test` image, checks compilation/imports and CPU availability, runs API,
geospatial inference, temporal, and full unit suites, then builds the runtime
target and verifies model-free liveness/readiness. It uses read-only repository
permissions and does not persist checkout credentials. The workflow requires
package/base-image downloads but no dataset, checkpoint, model download, GPU,
or private secret. It neither trains nor runs real model inference. The workflow
is locally validated through its Docker/test commands; a hosted Actions run
requires a later publication of the workflow.

Local container validation used the external read-only fine-tuned PT model and
TRAIN-derived frozen-VAL scene `108_1040010049B46C00`: 24 windows, 171 raw plane
candidates, 5 boundary rejections, 166 pre-NMS candidates, 34 suppressions, and
132 final features, exactly matching the saved native API/CLI counts and
non-runtime summary fields. All 132 polygons were valid and within the raster
footprint. Native API versus Linux container differences were bounded by
`2.0605739337042905e-10` degrees per coordinate, `6.103515625e-5` pixels per
vertex component, and `4.172325134277344e-7` confidence; minimum corresponding
polygon IoU was `0.9999932959531665`. This is numerical parity, not bitwise
identity or a speed benchmark. The local native and asset-free container suites
each ran 194 tests with one intentional skip. The optional ONNX container
smoke was not performed because the existing external ONNX artifact was not
readable under the current filesystem permissions; no permissions were changed.

## 15. Reproducibility boundaries

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
