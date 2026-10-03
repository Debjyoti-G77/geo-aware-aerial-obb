# Third-party notices

This file identifies material third-party components and external assets used
by the project. It is a factual attribution record, not a replacement for the
applicable license texts or legal advice.

## Project distribution

The source code for this combined project distribution is released under the
[GNU Affero General Public License, version 3](LICENSE). The geospatial,
RarePlanes, evaluation, reproducibility, and deployment extensions were
contributed in this project by Debjyoti Sengupta, 2026. This project-level
statement does not claim ownership of the detector baseline, external data,
installed dependencies, or model binaries.

## Detector baseline: YOLO26-DOTA-v2

The pretrained DOTA oriented-object detector and the historical baseline
utilities/results originated from
[JustinVladut/YOLO26-DOTA-v2](https://github.com/JustinVladut/YOLO26-DOTA-v2).
That baseline provided the detector starting point; this project adds the
geospatial, RarePlanes domain-adaptation, evaluation, GIS-output, and ONNX
deployment workflow.

The baseline was distributed under the MIT License. Its original notice,
`Copyright (c) 2026 JustinVladut`, and unmodified permission text are preserved
in [LICENSES/YOLO26-DOTA-v2-MIT.txt](LICENSES/YOLO26-DOTA-v2-MIT.txt). The
project-level AGPL-3.0 license does not mischaracterize the baseline's original
license or remove its attribution.

## Ultralytics

The project depends on the separately installed
[Ultralytics](https://github.com/ultralytics/ultralytics) Python package for
YOLO OBB loading, training, inference, and export. The locally inspected
Ultralytics 8.4.165 distribution metadata declares `License: AGPL-3.0` and
bundles the GNU Affero General Public License, version 3, in its installed
distribution metadata.

Ultralytics is a separately licensed dependency and is not owned by this
project. The open-source project distribution uses AGPL-3.0 consistently with
that dependency. Ultralytics may also make separate commercial terms
available; no such terms are granted by this repository. No Ultralytics
package source or license text is vendored here.

## RarePlanes data

RarePlanes imagery, annotations, and metadata are external assets and are not
redistributed in this source repository. The authoritative acquisition source
is the [RarePlanes entry in the Registry of Open Data on
AWS](https://registry.opendata.aws/rareplanes/), which identifies the public
bucket `s3://rareplanes-public/` and the dataset license as CC BY-SA 4.0.

This project documents and uses only RarePlanes TRAIN data. Official
RarePlanes TEST data is not part of the documented workflow or reported
evaluation.

## Model and deployment binaries

The DOTA `best.pt`, the RarePlanes fine-tuned
`rareplanes_finetuned_best.pt`, and generated ONNX binaries are intentionally
excluded from source control. Any future release hosting or redistribution of
those binaries requires a separate review of the licenses and terms applicable
to the model, training inputs, and producing software.

## Other installed dependencies

Other dependencies, including PyTorch, torchvision, ONNX, ONNX Runtime,
Rasterio, GDAL, Shapely, and PyProj, are installed separately and retain their
own licenses and bundled third-party notices. Package versions are documented
in `environment.yml`; consult each distribution's metadata and license files
when redistributing an environment or packaged application.

The project's AGPL-3.0 source license does not relicense or grant additional
rights to third-party packages, external datasets, or model binaries beyond
their own applicable terms.
