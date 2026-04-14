#!/usr/bin/env bash
# Start VLFM model servers without requiring tmux.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VLFM_PYTHON="${VLFM_PYTHON:-$(command -v python)}"
LOG_DIR="${VLFM_LOG_DIR:-logs/vlm_servers}"
PID_DIR="${LOG_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}" lockfiles

export MOBILE_SAM_CHECKPOINT="${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}"
export GROUNDING_DINO_WEIGHTS="${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}"
export GROUNDING_DINO_CONFIG="${GROUNDING_DINO_CONFIG:-$("${VLFM_PYTHON}" -c 'import groundingdino, pathlib; print(pathlib.Path(groundingdino.__file__).resolve().parent / "config" / "GroundingDINO_SwinT_OGC.py")')}"
export CLASSES_PATH="${CLASSES_PATH:-vlfm/vlm/classes.txt}"
export GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
export BLIP2ITM_PORT="${BLIP2ITM_PORT:-12182}"
export SAM_PORT="${SAM_PORT:-12183}"
export YOLOV7_PORT="${YOLOV7_PORT:-12184}"

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

check_file "${MOBILE_SAM_CHECKPOINT}"
check_file "${GROUNDING_DINO_WEIGHTS}"
check_file "${GROUNDING_DINO_CONFIG}"
check_file "data/yolov7-e6e.pt"

start_server() {
  local name="$1"
  local port="$2"
  shift 2
  local pid_file="${PID_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "${name} already running on port ${port} with pid $(cat "${pid_file}")"
    return
  fi

  echo "Starting ${name} on port ${port}; log: ${log_file}"
  setsid nohup "$@" >"${log_file}" 2>&1 </dev/null &
  echo "$!" >"${pid_file}"
}

start_server grounding_dino "${GROUNDING_DINO_PORT}" "${VLFM_PYTHON}" -m vlfm.vlm.grounding_dino --port "${GROUNDING_DINO_PORT}"
start_server blip2itm "${BLIP2ITM_PORT}" "${VLFM_PYTHON}" -m vlfm.vlm.blip2itm --port "${BLIP2ITM_PORT}"
start_server mobile_sam "${SAM_PORT}" "${VLFM_PYTHON}" -m vlfm.vlm.sam --port "${SAM_PORT}"
start_server yolov7 "${YOLOV7_PORT}" "${VLFM_PYTHON}" -m vlfm.vlm.yolov7 --port "${YOLOV7_PORT}"

echo
echo "VLFM servers are starting. First load can take a minute or two."
echo "Logs: ${LOG_DIR}"
echo "PIDs: ${PID_DIR}"
echo "Stop them with: scripts/stop_vlm_servers_local.sh"
