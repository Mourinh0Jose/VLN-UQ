#!/usr/bin/env bash
#SBATCH --job-name=vlfm_phase_a
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sbatch/vlfm_phase_a_%j.out
#SBATCH --error=logs/sbatch/vlfm_phase_a_%j.err
#
# Submit:    sbatch scripts/h100_phase_a.sbatch
# Watch:     tail -f logs/sbatch/vlfm_phase_a_*.out
# Status:    squeue -u $USER
# Cancel:    scancel <jobid>
#
# Note: if --gres=gpu:H100:1 is rejected by the partition, try --gres=gpu:1
# without the constraint, or ask PACE admins about the correct constraint name.

set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/hice1/jli3610/projects/VLN-UQ}"
cd "${ROOT_DIR}"

mkdir -p logs/sbatch logs/vlm_servers/pids

# 1. Activate conda env
echo "=== [sbatch] activate conda ==="
module load anaconda3/2023.03
source activate VLFM_gpu
which python
python -c "import habitat, torch; print('habitat', habitat.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 2. Launch VLM servers (Flask, ports 12181-12184)
echo ""
echo "=== [sbatch] launch VLM servers ==="
bash scripts/launch_vlm_servers_local.sh
echo "Waiting 90s for VLM models to load..."
sleep 90

# 3. Health check VLM servers (HTTP 404 on / is expected; means Flask is up)
echo ""
echo "=== [sbatch] health-check VLM servers ==="
ALL_UP=1
for port in 12181 12182 12183 12184; do
    code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://localhost:${port}/" || echo "DOWN")
    echo "  port ${port}: HTTP ${code}"
    if [[ "${code}" == "DOWN" || "${code}" == "000" ]]; then
        ALL_UP=0
    fi
done
if [[ ${ALL_UP} -ne 1 ]]; then
    echo "FATAL: at least one VLM server failed to start. Inspect logs/vlm_servers/*.log"
    bash scripts/stop_vlm_servers_local.sh || true
    exit 1
fi

# 4. Run Phase A
echo ""
echo "=== [sbatch] run Phase A (4 processes x 1000 eps) ==="
date
DATE="$(date +%Y%m%d_%H%M%S)"
export DATE
bash scripts/run_h100_phase_a.sh
EXIT=$?

# 5. Stop VLM servers
echo ""
echo "=== [sbatch] stop VLM servers ==="
bash scripts/stop_vlm_servers_local.sh || true

date
echo "=== [sbatch] job complete (exit=${EXIT}) ==="
exit ${EXIT}
