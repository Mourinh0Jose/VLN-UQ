#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RESULT_DIR="${RESULT_DIR:-results/val5scene90}"
SPLIT="${SPLIT:-val_5scene_90}"
EPISODES="${EPISODES:-90}"

ORIG_JSON_DIR="${RESULT_DIR}/vlfm_original_jsons"
BASE_JSON_DIR="${RESULT_DIR}/greedy_frontier_jsons"
ORIG_LOG="${RESULT_DIR}/vlfm_original_run.log"
BASE_LOG="${RESULT_DIR}/greedy_frontier_run.log"
ALIGN_LOG="${RESULT_DIR}/episode_alignment_check.txt"
SUMMARY_LOG="${RESULT_DIR}/summary.txt"

mkdir -p "${RESULT_DIR}" "${ORIG_JSON_DIR}" "${BASE_JSON_DIR}"

echo "[step8.6] run original semantic policy"
/usr/bin/time -f "wall_time_seconds=%e" bash -lc \
  "ZSOS_LOG_DIR='${ORIG_JSON_DIR}' SPLIT='${SPLIT}' EPISODES='${EPISODES}' scripts/run_vlfm_episode_local.sh habitat_baselines.rl.policy.frontier_selector=semantic" \
  2>&1 | tee "${ORIG_LOG}"

echo "[step8.6] run geometry baseline"
/usr/bin/time -f "wall_time_seconds=%e" bash -lc \
  "ZSOS_LOG_DIR='${BASE_JSON_DIR}' SPLIT='${SPLIT}' EPISODES='${EPISODES}' scripts/run_vlfm_episode_local.sh habitat_baselines.rl.policy.frontier_selector=cheapest" \
  2>&1 | tee "${BASE_LOG}"

echo "[step8.6] check alignment"
python scripts/check_episode_alignment.py \
  --vlfm-dir "${ORIG_JSON_DIR}" \
  --baseline-dir "${BASE_JSON_DIR}" | tee "${ALIGN_LOG}"

echo "[step8.6] write summary"
{
  echo "=== VLFM Original ==="
  python scripts/parse_jsons.py "${ORIG_JSON_DIR}"
  echo
  echo "=== Greedy Frontier Baseline ==="
  python scripts/parse_jsons.py "${BASE_JSON_DIR}"
  echo
  python scripts/summarize_paired.py \
    --vlfm-dir "${ORIG_JSON_DIR}" \
    --baseline-dir "${BASE_JSON_DIR}"
} | tee "${SUMMARY_LOG}"
