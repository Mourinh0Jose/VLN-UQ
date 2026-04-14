#!/usr/bin/env bash
# Stop servers started by launch_vlm_servers_local.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="${ROOT_DIR}/logs/vlm_servers/pids"

if [[ ! -d "${PID_DIR}" ]]; then
  echo "No PID directory found: ${PID_DIR}"
  exit 0
fi

for pid_file in "${PID_DIR}"/*.pid; do
  [[ -e "${pid_file}" ]] || continue
  name="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${name} (${pid})"
    kill "${pid}"
  else
    echo "${name} was not running (${pid})"
  fi
  rm -f "${pid_file}"
done
