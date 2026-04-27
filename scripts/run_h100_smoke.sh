#!/usr/bin/env bash
# H100 smoke test: 4 methods x 5 episodes x val_mini split (~10-20 min total).
# Verifies VLM servers + 4 frontier selectors + Habitat pipeline on H100.
# REQUIRES: VLM servers already started via scripts/launch_vlm_servers_local.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# H100 conda env path (override via env if needed)
export VLFM_PYTHON="${VLFM_PYTHON:-/home/hice1/jli3610/.conda/envs/VLFM_gpu/bin/python}"

SMOKE_DIR="${SMOKE_DIR:-results/h100_smoke_$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-val_mini}"
EPISODES="${EPISODES:-5}"

mkdir -p "${SMOKE_DIR}"

echo "=== H100 smoke test ==="
echo "Output dir: ${SMOKE_DIR}"
echo "Split: ${SPLIT}, Episodes per method: ${EPISODES}"
date

# 4 methods: name | selector | mc_lambda | mc_seed_master
for cfg in \
    "semantic|semantic|1.0|12345" \
    "cheapest|cheapest|1.0|12345" \
    "mc_mean|mc_risk_aware|0.0|42" \
    "mc_risk|mc_risk_aware|1.0|42"; do
    NAME="$(echo "${cfg}" | cut -d'|' -f1)"
    SELECTOR="$(echo "${cfg}" | cut -d'|' -f2)"
    LAMBDA="$(echo "${cfg}" | cut -d'|' -f3)"
    SEED="$(echo "${cfg}" | cut -d'|' -f4)"

    OUT_JSON="${SMOKE_DIR}/${NAME}_jsons"
    OUT_LOG="${SMOKE_DIR}/${NAME}_run.log"
    mkdir -p "${OUT_JSON}"

    echo ""
    echo "--- [smoke] ${NAME} (selector=${SELECTOR}, mc_lambda=${LAMBDA}, mc_seed=${SEED}) ---"

    /usr/bin/time -f "wall_time_seconds=%e" bash -lc \
        "ZSOS_LOG_DIR='${OUT_JSON}' SPLIT='${SPLIT}' EPISODES='${EPISODES}' \
         scripts/run_vlfm_episode_local.sh \
           habitat_baselines.rl.policy.frontier_selector=${SELECTOR} \
           habitat_baselines.rl.policy.mc_lambda=${LAMBDA} \
           habitat_baselines.rl.policy.mc_seed_master=${SEED}" \
        2>&1 | tee "${OUT_LOG}" || { echo "[smoke] ${NAME} FAILED — see ${OUT_LOG}"; exit 1; }

    NUM_JSON=$(ls "${OUT_JSON}"/*.json 2>/dev/null | wc -l)
    echo "[smoke] ${NAME}: ${NUM_JSON} JSONs (expected ${EPISODES})"
done

echo ""
echo "=== Smoke test complete ==="
date
ls -la "${SMOKE_DIR}/"
