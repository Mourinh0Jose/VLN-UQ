#!/usr/bin/env bash
# Run a small VLFM Habitat evaluation with the local GPU-compatible settings.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VLFM_PYTHON="${VLFM_PYTHON:-/home/jose/anaconda3/envs/VLFM_gpu/bin/python}"
SPLIT="${SPLIT:-val_mini}"
EPISODES="${EPISODES:-1}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

exec "${VLFM_PYTHON}" -m vlfm.run \
  "habitat_baselines.test_episode_count=${EPISODES}" \
  "habitat_baselines.eval.video_option=[]" \
  "habitat_baselines.eval.split=${SPLIT}" \
  "$@"

