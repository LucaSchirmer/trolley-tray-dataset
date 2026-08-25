#!/usr/bin/env bash
# Run and verify the complete automatic-mask pipeline.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOCAL_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: automatic_masks/run_pipeline.sh [PIPELINE OPTIONS]

Runs tests, input validation, checkpoint validation, device validation, mask inference, and output
acceptance checks. Options are forwarded to `mask_pipeline.py run`.

Examples:
  automatic_masks/run_pipeline.sh
  automatic_masks/run_pipeline.sh --device 1
  automatic_masks/run_pipeline.sh --device cpu
  automatic_masks/run_pipeline.sh --native-export /path/to/export.json

See `automatic_masks/mask_pipeline.py run --help` for all pipeline options.
EOF
  exit 0
fi

if [[ -x "${LOCAL_PYTHON}" ]]; then
  PYTHON="${LOCAL_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
else
  echo "Error: no virtual environment was found." >&2
  echo "Activate your venv or create automatic_masks/.venv as documented in automatic_masks/README.md." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Using Python: ${PYTHON}"
"${PYTHON}" -c "import sys; assert sys.version_info >= (3, 10), f'Python 3.10+ is required, found {sys.version.split()[0]}'" || {
  echo "Error: the selected venv uses an unsupported Python version." >&2
  echo "Run automatic_masks/setup_venv.sh with a Python 3.10+ interpreter." >&2
  exit 1
}
"${PYTHON}" -c "import torch, ultralytics, cv2; print(f'PyTorch: {torch.__version__}'); print(f'Ultralytics: {ultralytics.__version__}'); print(f'OpenCV: {cv2.__version__}')" || {
  echo "Error: required runtime dependencies are missing from the selected venv." >&2
  echo "Follow the environment setup in automatic_masks/README.md." >&2
  exit 1
}

echo
echo "[1/6] Running unit tests"
"${PYTHON}" -m pytest automatic_masks/tests

echo
echo "[2/6] Validating the selected input tasks"
"${PYTHON}" automatic_masks/mask_pipeline.py validate

# The pipeline defaults to GPU 0. Skip this early check only when the caller
# explicitly requests CPU inference.
USE_CPU=false
MODEL_PATH="segmentation_model/best_salmon_finetuned.pt"
PREVIOUS=""
for ARG in "$@"; do
  if [[ "${PREVIOUS}" == "--device" && "${ARG}" == "cpu" ]]; then
    USE_CPU=true
  elif [[ "${PREVIOUS}" == "--model" ]]; then
    MODEL_PATH="${ARG}"
  elif [[ "${ARG}" == "--device=cpu" ]]; then
    USE_CPU=true
  elif [[ "${ARG}" == --model=* ]]; then
    MODEL_PATH="${ARG#--model=}"
  fi
  PREVIOUS="${ARG}"
done

echo
echo "[3/6] Validating checkpoint taxonomy"
echo "Checkpoint: ${MODEL_PATH}"
"${PYTHON}" automatic_masks/mask_pipeline.py validate-model --model "${MODEL_PATH}"

echo
echo "[4/6] Checking inference device"
if [[ "${USE_CPU}" == true ]]; then
  echo "CPU inference explicitly requested. This may take a long time."
else
  "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))" || {
    echo "Error: CUDA is unavailable. Run nvidia-smi and follow the GPU troubleshooting section in automatic_masks/README.md." >&2
    echo "For a slow CPU run, invoke this script with: --device cpu" >&2
    exit 1
  }
fi

echo
echo "[5/6] Generating automatic masks"
"${PYTHON}" automatic_masks/mask_pipeline.py run "$@"

echo
echo "[6/6] Verifying production output"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path

summary_path = Path("automatic_masks/outputs/reports/run_summary.json")
if not summary_path.is_file():
    raise SystemExit(f"Missing run summary: {summary_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected = 131
checks = {
    "processed unique images": summary.get("processed_unique_images"),
    "YOLO label files": summary.get("yolo_label_files"),
    "Label Studio tasks": summary.get("preannotated_tasks"),
}
for label, actual in checks.items():
    if actual != expected:
        raise SystemExit(f"Acceptance check failed: {label} is {actual}, expected {expected}")

labels = list(Path("automatic_masks/outputs/labels").glob("*.txt"))
if len(labels) != expected:
    raise SystemExit(f"Acceptance check failed: found {len(labels)} physical label files, expected {expected}")

print(json.dumps(summary, indent=2))
print("All acceptance checks passed.")
PY

echo
echo "Pipeline complete."
echo "Review automatic_masks/outputs/contact_sheets/ before importing predictions."
echo "Label Studio import: automatic_masks/outputs/label_studio/preannotated_tasks.json"
