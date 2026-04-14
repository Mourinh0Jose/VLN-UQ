#!/usr/bin/env bash
# Download HM3D scene assets needed for VLFM evaluation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VLFM_PYTHON="${VLFM_PYTHON:-/home/jose/anaconda3/envs/VLFM_gpu/bin/python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
HM3D_UIDS="${HM3D_UIDS:-hm3d_val_v0.2}"
SAFE_DATA_LINK="${SAFE_DATA_LINK:-/tmp/vlfm_data_download}"

MATTERPORT_TOKEN_ID="${MATTERPORT_TOKEN_ID:-${MATTERPORT_USERNAME:-}}"
MATTERPORT_TOKEN_SECRET="${MATTERPORT_TOKEN_SECRET:-${MATTERPORT_PASSWORD:-}}"

if [[ -z "${MATTERPORT_TOKEN_ID}" || -z "${MATTERPORT_TOKEN_SECRET}" ]]; then
  cat >&2 <<'EOF'
Missing Matterport credentials.

Set these first:
  export MATTERPORT_TOKEN_ID="..."
  export MATTERPORT_TOKEN_SECRET="..."

Then run:
  scripts/download_hm3d_local.sh

For a larger download, override HM3D_UIDS, for example:
  HM3D_UIDS="hm3d_val_v0.2 hm3d_train_v0.2" scripts/download_hm3d_local.sh
EOF
  exit 2
fi

mkdir -p "${DATA_DIR}"

# habitat_sim's downloader shell-outs to curl without robust quoting for
# output paths, so repositories under paths with spaces need a safe alias.
DOWNLOAD_DATA_DIR="${DATA_DIR}"
if [[ "${DATA_DIR}" == *" "* ]]; then
  ln -sfn "${DATA_DIR}" "${SAFE_DATA_LINK}"
  DOWNLOAD_DATA_DIR="${SAFE_DATA_LINK}"
fi

"${VLFM_PYTHON}" -m habitat_sim.utils.datasets_download \
  --username "${MATTERPORT_TOKEN_ID}" \
  --password "${MATTERPORT_TOKEN_SECRET}" \
  --uids ${HM3D_UIDS} \
  --data-path "${DOWNLOAD_DATA_DIR}"

echo
echo "HM3D download finished under: ${DATA_DIR}/scene_datasets/hm3d"
