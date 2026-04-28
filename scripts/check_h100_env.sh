#!/usr/bin/env bash
# Read-only H100/PACE environment diagnostics for VLFM.

set -uo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

VLFM_PYTHON="${VLFM_PYTHON:-$(command -v python || true)}"

echo "=== VLFM H100 environment check ==="
date
echo "host: $(hostname)"
echo "cwd: $(pwd)"
echo "user: ${USER:-unknown}"
echo "python: ${VLFM_PYTHON}"
echo

echo "=== GPU / modules ==="
if command -v module >/dev/null 2>&1; then
  module list 2>&1 || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo

echo "=== Required files ==="
required_files=(
  "data/dummy_policy.pth"
  "data/pointnav_weights.pth"
  "data/mobile_sam.pt"
  "data/groundingdino_swint_ogc.pth"
  "data/yolov7-e6e.pt"
)
for path in "${required_files[@]}"; do
  if [[ -f "${path}" ]]; then
    size=$(du -h "${path}" | awk '{print $1}')
    echo "OK ${path} (${size})"
  else
    echo "MISSING ${path}"
  fi
done
echo

echo "=== HM3D dataset layout ==="
for split in val val_mini; do
  episode_dir="data/datasets/objectnav/hm3d/v1/${split}/content"
  scene_dir="data/scene_datasets/hm3d/val"
  if [[ -d "${episode_dir}" ]]; then
    echo "${split} episode files: $(find "${episode_dir}" -maxdepth 1 -name '*.json.gz' | wc -l)"
  else
    echo "MISSING ${episode_dir}"
  fi
  if [[ -f "data/datasets/objectnav/hm3d/v1/${split}/${split}.json.gz" ]]; then
    echo "OK data/datasets/objectnav/hm3d/v1/${split}/${split}.json.gz"
  else
    echo "MISSING data/datasets/objectnav/hm3d/v1/${split}/${split}.json.gz"
  fi
  if [[ -d "${scene_dir}" ]]; then
    echo "HM3D val scene glbs: $(find "${scene_dir}" \( -name '*.basis.glb' -o -name '*.glb' \) | wc -l)"
  else
    echo "MISSING ${scene_dir}"
  fi
done
echo

echo "=== Python imports and real CUDA op ==="
"${VLFM_PYTHON}" - <<'PY'
import importlib
import os
import pathlib
import sys

print("executable:", sys.executable)
print("version:", sys.version.replace("\n", " "))

mods = [
    "torch",
    "torchvision",
    "habitat",
    "habitat_baselines",
    "cv2",
    "numpy",
    "flask",
    "groundingdino",
    "mobile_sam",
    "frontier_exploration",
    "depth_camera_filtering",
    "vlfm",
]
for name in mods:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        path = getattr(mod, "__file__", "built-in")
        print(f"IMPORT OK {name}: version={version} path={path}")
    except Exception as exc:
        print(f"IMPORT FAIL {name}: {type(exc).__name__}: {exc}")

try:
    import groundingdino
    cfg = pathlib.Path(groundingdino.__file__).resolve().parent / "config" / "GroundingDINO_SwinT_OGC.py"
    print("groundingdino config:", cfg, "exists=", cfg.is_file())
except Exception as exc:
    print("groundingdino config lookup failed:", repr(exc))

try:
    import torch
    print("torch cuda_available:", torch.cuda.is_available())
    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("device name:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("arch list:", getattr(torch.cuda, "get_arch_list", lambda: "n/a")())
        x = torch.randn((1024, 1024), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("CUDA MATMUL OK:", float(y[0, 0].detach().cpu()))
except Exception as exc:
    print(f"CUDA CHECK FAIL {type(exc).__name__}: {exc}")

print("__EGL_VENDOR_LIBRARY_FILENAMES:", os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"))
PY
echo

echo "=== Hydra config dry load ==="
HYDRA_FULL_ERROR=1 "${VLFM_PYTHON}" - <<'PY'
from habitat import get_config

import vlfm.run  # noqa: F401 - registers this repo's Hydra config search path

cfg = get_config("config/experiments/vlfm_objectnav_hm3d.yaml")
print("trainer:", cfg.habitat_baselines.trainer_name)
print("eval split:", cfg.habitat_baselines.eval.split)
print("dataset data_path:", cfg.habitat.dataset.data_path)
print("scene dataset:", cfg.habitat.simulator.scene_dataset)
print("policy name:", cfg.habitat_baselines.rl.policy.name)
print("frontier selector:", cfg.habitat_baselines.rl.policy.frontier_selector)
PY
echo

echo "=== Listening VLM ports ==="
for port in 12181 12182 12183 12184; do
  code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" "http://localhost:${port}/" || echo "DOWN")
  echo "port ${port}: HTTP ${code}"
done

echo
echo "=== Done ==="
