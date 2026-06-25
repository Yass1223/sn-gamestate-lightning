#!/usr/bin/env bash
#
# SoccerNet Game State Reconstruction - evaluation runner.
#
# This script runs ON the Lightning AI machine (it is launched by
# .github/scripts/run_lightning_eval.py). It:
#   1. Installs all dependencies with uv (Python 3.9), including mmcv via mim.
#   2. Runs the TrackLab GSR pipeline on each requested split.
#   3. Collects the metric outputs (GS-HOTA + full TrackEval suite) into RESULTS_DIR.
#
# The SoccerNetGS dataset and all model weights are downloaded automatically by
# TrackLab on the first run if they are not already present on the machine.
#
# Configuration via environment variables (with defaults):
#   SPLITS       splits to evaluate, space separated   (default: "valid test")
#   NVID         number of videos, -1 means all        (default: "-1")
#   RESULTS_DIR  where metric summaries are copied      (default: "eval_results")
#   CONFIG_NAME  tracklab config name                   (default: "soccernet")

set -euo pipefail

SPLITS="${SPLITS:-valid test}"
NVID="${NVID:--1}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
CONFIG_NAME="${CONFIG_NAME:-soccernet}"

echo "=================================================================="
echo " SoccerNet GSR evaluation"
echo "   splits      : ${SPLITS}"
echo "   nvid        : ${NVID}"
echo "   config      : ${CONFIG_NAME}"
echo "   results dir : ${RESULTS_DIR}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. Dependencies (idempotent: uv only reinstalls what changed)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> Creating the Python 3.9 environment and installing dependencies"
uv venv --python 3.9
uv pip install -e .
uv run mim install mmcv==2.0.1

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# 2. Run evaluation on each split
#    Dataset + model weights auto-download on first use.
# ---------------------------------------------------------------------------
for split in ${SPLITS}; do
  echo "------------------------------------------------------------------"
  echo "==> Evaluating split: ${split}  (nvid=${NVID})"
  echo "------------------------------------------------------------------"
  # Metrics are printed to stdout AND saved by TrackEval under outputs/.
  # `tee` keeps a copy of the full log (which contains the printed metric tables).
  uv run tracklab -cn "${CONFIG_NAME}" \
      dataset.eval_set="${split}" \
      dataset.nvid="${NVID}" \
      2>&1 | tee "${RESULTS_DIR}/eval_${split}.log"
done

# ---------------------------------------------------------------------------
# 3. Gather metric summary files produced by TrackEval
#    (exact filenames depend on the TrackEval/TrackLab version; we copy broadly
#     and the per-split logs above always contain the printed metric tables).
# ---------------------------------------------------------------------------
echo "==> Collecting metric summary files into ${RESULTS_DIR}/"
if [ -d outputs ]; then
  find outputs -type f \
      \( -iname "*summary*.txt" -o -iname "*detailed*.csv" -o -iname "*.json" \) \
      -exec cp --parents {} "${RESULTS_DIR}/" \; 2>/dev/null || true
fi

echo "=================================================================="
echo " Done."
echo "   Per-split logs (with printed metrics): ${RESULTS_DIR}/eval_<split>.log"
echo "   Copied summary files                 : ${RESULTS_DIR}/outputs/..."
echo "   Full run outputs                     : ./outputs/"
echo "=================================================================="
