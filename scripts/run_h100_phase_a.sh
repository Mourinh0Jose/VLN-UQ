#!/usr/bin/env bash
# H100 Phase A: 4 methods x 1000 episodes paired = 4000 total, 4 parallel processes.
# Phase A composition:
#   - semantic            (VLFM original)              N=1000
#   - cheapest            (geometry baseline)          N=1000
#   - mc_risk seed=42     (MC risk-aware, lambda=1.0)  N=1000
#   - mc_risk seed=123    (MC risk-aware, lambda=1.0)  N=1000
#
# Phase B (next H100 allocation): mc_mean x 3 seeds + mc_risk seed=2024 = 4000 more.
# REQUIRES: VLM servers already running on ports 12181-12184.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export VLFM_PYTHON="${VLFM_PYTHON:-/home/hice1/jli3610/.conda/envs/VLFM_gpu/bin/python}"

DATE="${DATE:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-results/h100_phase_a_${DATE}}"
SPLIT="${SPLIT:-val}"
EPISODES="${EPISODES:-1000}"

SEMANTIC_DIR="${RESULT_DIR}/semantic_jsons"
CHEAPEST_DIR="${RESULT_DIR}/cheapest_jsons"
MC42_DIR="${RESULT_DIR}/mc_risk_seed42_jsons"
MC123_DIR="${RESULT_DIR}/mc_risk_seed123_jsons"

mkdir -p "${SEMANTIC_DIR}" "${CHEAPEST_DIR}" "${MC42_DIR}" "${MC123_DIR}"

run_one() {
    local NAME="$1"
    local SELECTOR="$2"
    local LAMBDA="$3"
    local SEED="$4"
    local OUT_JSON="$5"
    local OUT_LOG="$6"

    ZSOS_LOG_DIR="${OUT_JSON}" SPLIT="${SPLIT}" EPISODES="${EPISODES}" \
        /usr/bin/time -f "wall_time_seconds=%e" bash -lc \
        "scripts/run_vlfm_episode_local.sh \
           habitat_baselines.rl.policy.frontier_selector=${SELECTOR} \
           habitat_baselines.rl.policy.mc_lambda=${LAMBDA} \
           habitat_baselines.rl.policy.mc_seed_master=${SEED}" \
        > "${OUT_LOG}" 2>&1
}

echo "=== Phase A: 4 parallel processes, ${EPISODES} eps each, split=${SPLIT} ==="
echo "Result dir: ${RESULT_DIR}"
date

run_one "semantic"       "semantic"      "1.0" "12345" "${SEMANTIC_DIR}" "${RESULT_DIR}/semantic.log" &
PID_SEM=$!
run_one "cheapest"       "cheapest"      "1.0" "12345" "${CHEAPEST_DIR}" "${RESULT_DIR}/cheapest.log" &
PID_CHEAP=$!
run_one "mc_risk_seed42"  "mc_risk_aware" "1.0" "42"   "${MC42_DIR}"     "${RESULT_DIR}/mc_risk_seed42.log" &
PID_MC42=$!
run_one "mc_risk_seed123" "mc_risk_aware" "1.0" "123"  "${MC123_DIR}"    "${RESULT_DIR}/mc_risk_seed123.log" &
PID_MC123=$!

echo "PIDs: semantic=${PID_SEM} cheapest=${PID_CHEAP} mc_risk42=${PID_MC42} mc_risk123=${PID_MC123}"

EXIT_SEM=0;   wait ${PID_SEM}    || EXIT_SEM=$?
EXIT_CHEAP=0; wait ${PID_CHEAP}  || EXIT_CHEAP=$?
EXIT_MC42=0;  wait ${PID_MC42}   || EXIT_MC42=$?
EXIT_MC123=0; wait ${PID_MC123}  || EXIT_MC123=$?

echo ""
echo "=== Phase A finished ==="
date
echo "Exit codes: semantic=${EXIT_SEM} cheapest=${EXIT_CHEAP} mc_risk42=${EXIT_MC42} mc_risk123=${EXIT_MC123}"

for d in "${SEMANTIC_DIR}" "${CHEAPEST_DIR}" "${MC42_DIR}" "${MC123_DIR}"; do
    n=$(ls "${d}"/*.json 2>/dev/null | wc -l)
    echo "  ${d}: ${n} JSONs"
done

# Alignment + summary only if all 4 processes succeeded
if [[ ${EXIT_SEM} -eq 0 && ${EXIT_CHEAP} -eq 0 && ${EXIT_MC42} -eq 0 && ${EXIT_MC123} -eq 0 ]]; then
    echo ""
    echo "=== Episode alignment: semantic vs cheapest ==="
    python scripts/check_episode_alignment.py \
        --vlfm-dir "${SEMANTIC_DIR}" \
        --baseline-dir "${CHEAPEST_DIR}" 2>&1 | tee "${RESULT_DIR}/alignment_semantic_cheapest.txt"

    echo ""
    echo "=== Episode alignment: semantic vs mc_risk_seed42 ==="
    python scripts/check_episode_alignment.py \
        --vlfm-dir "${SEMANTIC_DIR}" \
        --baseline-dir "${MC42_DIR}" 2>&1 | tee "${RESULT_DIR}/alignment_semantic_mc42.txt"

    echo ""
    echo "=== Episode alignment: mc_risk_seed42 vs mc_risk_seed123 ==="
    python scripts/check_episode_alignment.py \
        --vlfm-dir "${MC42_DIR}" \
        --baseline-dir "${MC123_DIR}" 2>&1 | tee "${RESULT_DIR}/alignment_mc42_mc123.txt"

    echo ""
    echo "=== Per-method aggregate metrics ==="
    {
        for entry in "semantic|${SEMANTIC_DIR}" "cheapest|${CHEAPEST_DIR}" "mc_risk_seed42|${MC42_DIR}" "mc_risk_seed123|${MC123_DIR}"; do
            NAME="$(echo "${entry}" | cut -d'|' -f1)"
            DIR="$(echo "${entry}" | cut -d'|' -f2)"
            echo "--- ${NAME} ---"
            python scripts/parse_jsons.py "${DIR}"
            echo
        done
    } | tee "${RESULT_DIR}/summary.txt"

    echo ""
    echo "Phase A complete. Results: ${RESULT_DIR}/"
    exit 0
else
    echo ""
    echo "WARNING: at least one process failed. Skipping alignment + summary."
    echo "Inspect logs in ${RESULT_DIR}/*.log to diagnose."
    exit 1
fi
