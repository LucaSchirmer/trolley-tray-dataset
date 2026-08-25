# Automatic masks for consumed fish-and-rice trays

This directory contains a standalone pipeline that applies the custom YOLO segmentation checkpoint to the selected
consumed fish-and-rice photographs. It does not modify the source annotations, images, dataset scripts, or checkpoint.

The production input is `annotations/pairs_fish_rice_labelstudio.json`. Its 129 rows resolve to 123 unique consumed
images. The pipeline also discovers the 8 physical fish-and-rice images omitted by that export, so all 131 images are
annotated. Duplicate rows are audited and then deduplicated by consumed-image path. The first occurrence is retained
when duplicate task metadata conflicts; omitted images receive the four core-food classes as their allowed taxonomy.

## What the pipeline does

For every physical consumed fish-and-rice image, the pipeline:

1. Runs `segmentation_model/best_salmon_finetuned.pt` at 1024 pixels with `conf=0.20` and `iou=0.70`. Direct
   `fish_salmon` predictions use the reference-calibrated threshold `0.10`; the audited chicken-to-salmon fish-domain
   override uses `0.08`.
2. Maps the checkpoint's embedded class names to the Label Studio taxonomy.
3. Rejects classes not listed in that task's `possibleElements`.
4. Preserves every valid instance polygon (for example, disconnected rice clusters and carrot pieces); model NMS
   removes duplicate detections.
5. Uses the known fish-only task taxonomy to map otherwise-forbidden `chicken` predictions to `fish_salmon`. Every
   such auditable override is recorded in `reports/rejected_polygons.json` with reason `class_override`.
6. Remaps checkpoint IDs to `yolo_dataset_salmon_veg_rice_15/classes.txt` before writing YOLO labels.
7. Writes a YOLO polygon label file, including an empty file when no mask survives.
8. Produces Label Studio predictions, confidence data, rejection logs, and visual previews.

Inference is performed only on consumed images. No consumption percentage, drink state, flag, note, or unconsumed
image annotation is generated.

## Repository paths

Run all commands from the repository root—the directory containing `automatic_masks`, `annotations`, and
`segmentation_model`.

| Purpose | Path |
| --- | --- |
| Pipeline | `automatic_masks/mask_pipeline.py` |
| Production tasks | `annotations/pairs_fish_rice_labelstudio.json` |
| Consumed images | `images_cropped/consumed/fish_rice/` |
| Production checkpoint | `segmentation_model/best_salmon_finetuned.pt` |
| Original rollback checkpoint | `segmentation_model/best.pt` |
| Label Studio config | `annotations/galleyeye_label_studio_config_v2.xml` |
| Generated output | `automatic_masks/outputs/` |

## Prerequisites

- Python 3.10 or newer with the `venv` module
- An NVIDIA GPU for the production run
- A working NVIDIA driver visible inside the shell/WSL environment
- Enough disk space for PyTorch, Ultralytics, previews, and contact sheets

Confirm GPU access before installing the environment:

```bash
nvidia-smi
```

Do not start the GPU run if this command reports an NVML or driver error. On WSL, update/restart the Windows NVIDIA
driver and WSL, then retry `nvidia-smi` inside WSL.

## One-time environment setup

The environment is intentionally located below `automatic_masks/` and ignored by Git.

The repository-level `requirements.txt` contains the dependencies for Label Studio and the automatic-mask tools. Use the automated
setup script with Python 3.10 or newer:

```bash
automatic_masks/setup_venv.sh
```

The setup script defaults to `python3.10`. Select another compatible interpreter or PyTorch wheel index with
environment variables:

```bash
PYTHON_BIN=python3.11 automatic_masks/setup_venv.sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 automatic_masks/setup_venv.sh
```

The equivalent manual setup is:

```bash
python3.10 -m venv automatic_masks/.venv
automatic_masks/.venv/bin/python -m pip install --upgrade pip
automatic_masks/.venv/bin/pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
automatic_masks/.venv/bin/pip install -r requirements.txt
```

The separate PyTorch command selects CUDA 12.8 wheels. If the machine requires a different supported CUDA wheel,
change only the PyTorch index URL. The following command should print `True` and identify the GPU:

```bash
automatic_masks/.venv/bin/python -c \
  "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Run the unit tests after installation:

```bash
automatic_masks/.venv/bin/python -m pytest automatic_masks/tests
```

## Recommended production run

### One-command launcher

The recommended entry point runs the tests, input validation, device check, inference, and final 131-output acceptance
checks in sequence:

```bash
automatic_masks/run_pipeline.sh
```

It uses the currently activated virtual environment when one exists. Otherwise, it uses
`automatic_masks/.venv/bin/python`. Optional inference arguments are forwarded to the pipeline:

```bash
# Use GPU 1
automatic_masks/run_pipeline.sh --device 1

# Generate an API payload for existing Label Studio tasks
automatic_masks/run_pipeline.sh \
  --native-export /absolute/path/to/native-label-studio-export.json

# Explicit CPU fallback; substantially slower
automatic_masks/run_pipeline.sh --device cpu
```

The launcher stops immediately if tests, validation, CUDA detection, inference, or an acceptance check fails.

### Individual commands

First validate the task selection without loading PyTorch or the checkpoint:

```bash
automatic_masks/.venv/bin/python automatic_masks/mask_pipeline.py validate
```

The expected result is:

```text
Validated 131 unique images; 6 duplicate rows; 9 invalid-reference tasks; 8 physical images outside selection
```

Review `automatic_masks/outputs/reports/input_validation.json`. In particular, inspect
`conflicting_reference_assignments`; inference deterministically uses the first task row for those duplicates.

Then run the complete production job on GPU 0:

```bash
automatic_masks/.venv/bin/python automatic_masks/mask_pipeline.py run
```

`run` performs the input validation again, validates the checkpoint taxonomy, and generates all outputs. Its production
defaults are equivalent to:

```bash
automatic_masks/.venv/bin/python automatic_masks/mask_pipeline.py infer \
  --device 0 \
  --imgsz 1024 \
  --conf 0.20 \
  --fish-conf 0.10 \
  --iou 0.70
```

To use another GPU, pass its index with `--device`, for example `--device 1`. CPU inference is available with
`--device cpu`, but processing all 131 images will be substantially slower.

Do not use `--allow-nonstandard-count` for production. It exists only for tests and small development fixtures.

## Generated outputs

Everything generated by the pipeline stays below `automatic_masks/outputs/` and is ignored by Git.

| Output | Description |
| --- | --- |
| `labels/*.txt` | YOLO segmentation labels; exactly one file for every physical image |
| `label_studio/preannotated_tasks.json` | Import file for a new Label Studio project |
| `label_studio/api_predictions.json` | Optional prediction payload for tasks that already have Label Studio IDs |
| `previews/all_classes/*.jpg` | Individual overlays for every retained class, using distinct colors and readable labels |
| `previews/core_food/*.jpg` | Individual overlays limited to fish, rice, broccoli, and carrots |
| `contact_sheets/all_classes/*.jpg` | Review sheets containing every retained class |
| `contact_sheets/core_food/*.jpg` | Food-only review sheets without package and drink clutter |
| `contact_sheets/low_confidence/*.jpg` | Images containing at least one retained prediction below `0.40` |
| `contact_sheets/nearly_empty/*.jpg` | Images where core-food polygons cover less than 1% of the image |
| `reports/input_validation.json` | Duplicate, conflict, invalid-reference, missing-file, and outside-selection audit |
| `reports/model_validation.json` | Embedded checkpoint names and their canonical mapping |
| `reports/confidences.csv` | Confidence for every retained image/class polygon |
| `reports/review_priorities.csv` | Per-image core-food count, confidence, area, and priority flags |
| `reports/rejected_polygons.json` | Rejected predictions and rejection reasons |
| `reports/run_summary.json` | Production settings and final output counts |
| `reports/evaluation.json` | Core-class mask IoU measured against the 15 hand-labeled references |
| `reports/evaluation.csv` | Per-image/class IoU, precision, recall, and pixel counts |

Re-run the objective comparison after inference with:

```bash
automatic_masks/.venv/bin/python automatic_masks/evaluate_masks.py
```

This evaluator unions disconnected YOLO polygons per class before raster IoU calculation, matching the semantic-mask
meaning of the annotations. Treat these scores as a quality gate and still review contact sheets: automatic masks are
pre-annotations, not a substitute for human acceptance on safety- or research-critical ground truth.

An empty YOLO text file is valid: it means the image was processed but no acceptable polygon survived filtering.

## Triage for manual correction

Create three deterministic review queues after inference:

```bash
automatic_masks/.venv/bin/python automatic_masks/active_learning.py triage
```

The command writes `label_studio/triage/triage.csv`, a count summary, and three Label Studio JSON files:

- `no_rework.json`: the 15 images that already have human labels in `yolo_dataset_salmon_veg_rice_15`; use those
  human labels rather than accepting the generated masks.
- `full_redo.json`: near-empty images, excluding the human references. Predictions are deliberately removed because
  drawing the few genuine remnants from a blank task is safer than deleting many model speckles.
- `minor_rework.json`: all other images with model predictions retained for boundary, missed-region, and false-positive
  corrections.

This is a workload triage, not a claim that confidence proves annotation correctness. The `triage.csv` reason and
metrics make every assignment auditable. To triage a preserved candidate run, pass its preannotations, priorities,
and an output directory explicitly.

To review the queues in Label Studio:

1. Start the repository's image server with `python scripts/serve_with_cors.py --port 8000`.
2. Create a project using `annotations/galleyeye_label_studio_config_v2.xml`.
3. Enable **Settings → Annotation → Use predictions to pre-label tasks**.
4. Import `minor_rework.json` and `full_redo.json` from the triage directory. Do not import `no_rework.json` for
   correction; its images already have human YOLO labels in `yolo_dataset_salmon_veg_rice_15`.

Label Studio predictions are read-only until copied into an annotation. With prediction pre-labeling enabled, opening
a task creates the editable annotation from its prediction; otherwise use the editor's copy-prediction action. Correct
or delete polygons and submit every reviewed task before exporting native Label Studio JSON.

## Active learning on corrected consumed trays

The inference pipeline does not train the model. To make corrections improve later predictions, use the active-learning
workflow below. It deliberately trains only the four core food classes (`fish_salmon`, `rice`, `broccoli`, and
`carrots`) and preserves completely consumed images as empty negative examples.

After running the mask pipeline, select a deterministic 40-image correction batch balanced across nearly-empty,
low-core-confidence, and regular trays:

```bash
automatic_masks/.venv/bin/python automatic_masks/active_learning.py select
```

Import `automatic_masks/outputs/label_studio/active_learning_batch.json` into a new Label Studio project using
`annotations/galleyeye_label_studio_config_v2.xml`. Correct the core-food polygons according to the annotation
guidelines. Delete a food polygon when that food is completely gone. Keep separate polygons when visible pieces of the
same class are disconnected.
Complete every selected task, then export the tasks as native Label Studio JSON including annotations.

Convert that corrected export into a leakage-safe YOLO dataset. Images sharing the same unconsumed reference remain in
the same split:

```bash
automatic_masks/.venv/bin/python automatic_masks/active_learning.py prepare \
  --export /absolute/path/to/corrected-label-studio-export.json
```

Review `automatic_masks/training_data/consumed_core_v1/summary.json` and `manifest.csv`. The summary should include
both training and validation images and should report some `empty_core_labels` when the batch contains completely
consumed trays.

Start fine-tuning only after reviewing that dataset:

```bash
automatic_masks/.venv/bin/python automatic_masks/active_learning.py train
```

Training starts from `segmentation_model/best_salmon_finetuned.pt`, uses 1024-pixel images for 50 epochs by default, and writes runs
below `automatic_masks/training_runs/`. Keep the original checkpoint. Validate the new run's `weights/best.pt` on a
fixed corrected holdout before using it for production inference.

## Rebuilding the sample-derived production checkpoint

The existing 15-image YOLO sample was used with a leakage-visible 12/3 train/holdout split. Preserve the checkpoint's
24-class head when fine-tuning: replacing it with a new four-class head produced zero held-out salmon recall and was
rejected. The accepted conservative run transferred all pretrained weights, froze backbone layers 0–9, and used a
`0.0001` learning rate:

```bash
automatic_masks/.venv/bin/python automatic_masks/active_learning.py prepare-checkpoint-sample
automatic_masks/.venv/bin/python automatic_masks/active_learning.py train \
  --data automatic_masks/training_data/sample_checkpoint_v1/dataset.yaml \
  --model segmentation_model/best.pt \
  --epochs 30 --imgsz 1024 --batch 2 --device 0 \
  --lr0 0.0001 --freeze 10 --name sample-checkpoint-v1
```

The promoted checkpoint is `segmentation_model/best_salmon_finetuned.pt`; the original remains available for rollback.
On the three images excluded from training, mean core semantic IoU improved from `0.648` to `0.675`. On all 15 reference
images, the promoted production pipeline measures `0.779` mean IoU: broccoli `0.919`, carrots `0.844`, rice `0.749`,
and salmon `0.693`. Reproduce the current measurements with `automatic_masks/evaluate_masks.py`.

## Acceptance checks

After `run` completes, execute:

```bash
find automatic_masks/outputs/labels -maxdepth 1 -type f -name '*.txt' | wc -l
automatic_masks/.venv/bin/python -c \
  "import json; r=json.load(open('automatic_masks/outputs/reports/run_summary.json')); print(json.dumps(r, indent=2)); assert r['processed_unique_images']==131; assert r['yolo_label_files']==131; assert r['preannotated_tasks']==131"
```

The first command and all three asserted counts must equal `131`. Also confirm that
`reports/model_validation.json` contains `"valid": true` and maps `fish_salmon`, `rice`, `broccoli`, and `carrots`.

Before importing anything into Label Studio:

1. Review every image in `contact_sheets/`.
2. Inspect the normal, nearly empty, and invalid-reference trays especially closely.
3. Check low-confidence and rejected cases in the reports.
4. Stop and correct the pipeline/model if masks are systematically misplaced.

## Label Studio: import as a new project

1. Create a Label Studio project.
2. Use `annotations/galleyeye_label_studio_config_v2.xml` as its labeling configuration.
3. Ensure the image server represented by the task URLs is running and reachable from the browser. The current tasks
   use `http://localhost:8000/...`; the repository's existing server can be started with:

   ```bash
   automatic_masks/.venv/bin/python scripts/serve_with_cors.py
   ```

4. Import `automatic_masks/outputs/label_studio/preannotated_tasks.json`.
5. Open several tasks and verify that polygons appear on the consumed image under
   `masks_consumed -> img_consumed`.
6. Manually review and correct every generated mask before accepting annotations.

## Label Studio: attach predictions to existing tasks

An existing project requires a native Label Studio JSON export containing task IDs and each task's `data.consumed`
URL. Create the API prediction payload directly during inference:

```bash
automatic_masks/.venv/bin/python automatic_masks/mask_pipeline.py run \
  --native-export /absolute/path/to/native-label-studio-export.json
```

Or convert already-generated pre-annotations without rerunning inference:

```bash
automatic_masks/.venv/bin/python automatic_masks/mask_pipeline.py convert \
  --native-export /absolute/path/to/native-label-studio-export.json
```

The result is `automatic_masks/outputs/label_studio/api_predictions.json`. This is a Label Studio API payload, not a
fresh-project task import. The native export must contain exactly matching consumed-image URLs and unique task IDs.

## Command reference

```text
validate  Audit and deduplicate task input; no model dependencies required
validate-model  Load a checkpoint and validate its embedded class taxonomy without inference
infer     Validate input and run inference
run       End-to-end production alias for validation plus inference
convert   Match existing pre-annotations to native Label Studio task IDs
```

Show all options for a command with:

```bash
automatic_masks/.venv/bin/python automatic_masks/mask_pipeline.py run --help
```

Custom inputs and output locations can be supplied with `--tasks`, `--images`, `--model`, and `--output`. Relative
image paths stored inside the task JSON are always resolved from the repository root.

## Troubleshooting

### `CUDA was requested but torch.cuda.is_available() is false`

Run `nvidia-smi`, then run the Python CUDA check from the setup section. If `nvidia-smi` fails, fix host/WSL GPU
access first. If only Python fails, reinstall a PyTorch build compatible with the system's NVIDIA driver.

### `Failed to initialize NVML`

The operating system is preventing GPU access. Under WSL, shut down WSL from Windows with `wsl --shutdown`, update or
restart the NVIDIA driver if necessary, reopen the distribution, and verify `nvidia-smi` before retrying.

### Checkpoint taxonomy mismatch

Inference stops before processing images and records the embedded names and error in
`outputs/reports/model_validation.json`. Do not bypass this check. Correct the class aliases or supply the intended
checkpoint, then rerun.

### Missing images or a count other than 131

Inspect `outputs/reports/input_validation.json`. Confirm that commands are running from the correct repository and
that the production task JSON and cropped-image directory have not changed.

### Label Studio shows broken images

Start the local image server, keep it running during annotation, and verify one task URL in a browser. If Label Studio
runs in Docker or on another host, `localhost` refers to that environment rather than the image-server machine; adjust
the task URLs or networking accordingly.

### Label Studio loads tasks but no polygons

Confirm that the project uses `galleyeye_label_studio_config_v2.xml`, and inspect a generated result for
`"from_name": "masks_consumed"`, `"to_name": "img_consumed"`, and `"type": "polygonlabels"`.

## Scope and follow-up

This workflow deliberately does not retrain the model. If consumed salmon masks are inadequate, correct predictions
in Label Studio first and use the reviewed consumed annotations in a separate fine-tuning phase.
