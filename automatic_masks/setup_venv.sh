#!/usr/bin/env bash
# Create the dedicated automatic-mask environment and install its dependencies.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "${SCRIPT_DIR}")"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found." >&2
  echo "Install Python 3.10 or newer, then rerun this script." >&2
  echo "You may select another interpreter with, for example:" >&2
  echo "  PYTHON_BIN=python3.11 automatic_masks/setup_venv.sh" >&2
  exit 1
fi

"${PYTHON_BIN}" -c "import sys; assert sys.version_info >= (3, 10), f'Python 3.10+ is required, found {sys.version.split()[0]}'"

if [[ -e "${VENV_DIR}" ]]; then
  echo "Error: ${VENV_DIR} already exists." >&2
  echo "Remove it only if you intentionally want to recreate the dedicated mask environment." >&2
  exit 1
fi

echo "Creating ${VENV_DIR} with $(${PYTHON_BIN} --version)"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
  uv pip install --link-mode copy --python "${VENV_DIR}/bin/python" \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url "${TORCH_INDEX_URL}"
  uv pip install --link-mode copy --python "${VENV_DIR}/bin/python" -r "${REPO_ROOT}/requirements.txt"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/pip" install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url "${TORCH_INDEX_URL}"
  "${VENV_DIR}/bin/pip" install -r "${REPO_ROOT}/requirements.txt"
fi

"${VENV_DIR}/bin/python" -c \
  "import sys, torch, ultralytics, cv2; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__)"

echo
echo "Environment setup complete. Run:"
echo "  automatic_masks/run_pipeline.sh"
