#!/usr/bin/env bash
#
# SoccerNet Game State Reconstruction - evaluation runner (Lightning Studio).
# Self-healing: patches a known dataset-task bug, downloads only the needed
# split(s), installs deps into a local .venv, and runs single-worker to avoid
# the cuDNN/torch_shm_manager issue on the Studio.

set -euo pipefail

SPLITS="${SPLITS:-test}"
NVID="${NVID:--1}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
# Default config = BoT-SORT · SOF + GTA-Link tracker (soccernet_botsort).
# Set CONFIG_NAME=soccernet for the original StrongSORT baseline.
CONFIG_NAME="${CONFIG_NAME:-soccernet_botsort}"
VENV="${VENV:-.venv}"
DATA_DIR="data/SoccerNetGS"

echo "=================================================================="
echo " SoccerNet GSR evaluation | config=${CONFIG_NAME} splits=${SPLITS} nvid=${NVID}"
echo "=================================================================="

# 1. Dependencies into a local .venv (Python 3.9)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv venv --python 3.9 "${VENV}"
uv pip install --python "${VENV}" -e .
uv run --python "${VENV}" mim install mmcv==2.0.1

# 2. Patch the gamestate-2025 -> 2024 bug in the installed TrackLab
SNGS=$(uv run --python "${VENV}" python -c "import tracklab,os;print(os.path.join(os.path.dirname(tracklab.__file__),'wrappers','dataset','soccernet','soccernet_game_state.py'))")
sed -i 's/gamestate-2025/gamestate-2024/g' "${SNGS}"
echo "==> Patched dataset task name in ${SNGS}"

# 3. Download only the needed split(s) and unzip into place
mkdir -p "${DATA_DIR}"
for split in ${SPLITS}; do
  if [ ! -d "${DATA_DIR}/${split}" ]; then
    echo "==> Downloading SoccerNetGS split: ${split}"
    uv run --python "${VENV}" python -c "
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory='${DATA_DIR}')
d.downloadDataTask(task='gamestate-2024', split=['${split}'])
"
    unzip -o "${DATA_DIR}/gamestate-2024/${split}.zip" -d "${DATA_DIR}/${split}"
  else
    echo "==> Split '${split}' already present, skipping download."
  fi
done

# 4. Run evaluation per split (single-worker; data already present)
mkdir -p "${RESULTS_DIR}"
for split in ${SPLITS}; do
  echo "==> Evaluating split: ${split} (nvid=${NVID})"
  echo "n" | uv run --python "${VENV}" tracklab -cn "${CONFIG_NAME}" \
      dataset.eval_set="${split}" \
      dataset.nvid="${NVID}" \
      num_cores=0 \
      2>&1 | tee "${RESULTS_DIR}/eval_${split}.log"
done

echo "==> Done. Logs (with metric tables) in ${RESULTS_DIR}/eval_<split>.log"