# H100 Phase A Run Guide

H100 sweep on PACE-ICE: 4 methods × 1000 episodes paired = **4000 eps total**, 4 parallel processes.

## Phase A composition (this run)

| method        | frontier_selector | mc_lambda | mc_seed_master | N    |
|---------------|-------------------|-----------|----------------|------|
| semantic      | `semantic`        | 1.0       | 12345          | 1000 |
| cheapest      | `cheapest`        | 1.0       | 12345          | 1000 |
| mc_risk_seed42  | `mc_risk_aware` | 1.0       | 42             | 1000 |
| mc_risk_seed123 | `mc_risk_aware` | 1.0       | 123            | 1000 |

## Phase B (next H100 allocation, 4000 more eps)

- mc_mean × 3 seeds (42 / 123 / 2024) at `mc_lambda=0.0`
- mc_risk seed=2024 at `mc_lambda=1.0`

## Prerequisites

- H100 node (interactive job or sbatch on `ice-gpu` / `coc-gpu` partition)
- conda env `VLFM_gpu` at `~/.conda/envs/VLFM_gpu/`
- HM3D val data: 20 scenes in `data/scene_datasets/hm3d/val/` and 20 episode files in `data/datasets/objectnav/hm3d/v1/val/content/`

## Step 1: pull latest code on H100

```bash
cd /home/hice1/jli3610/projects/VLN-UQ
git fetch origin
git pull origin codex/mc-risk-aware-frontier-reranking
chmod +x scripts/run_h100_smoke.sh scripts/run_h100_phase_a.sh
```

## Step 2: smoke test (10–20 min, interactive)

Confirms 4 methods × 5 episodes × val_mini all succeed.

```bash
module load anaconda3/2023.03
source activate VLFM_gpu

bash scripts/launch_vlm_servers_local.sh
sleep 90

bash scripts/run_h100_smoke.sh

ls results/h100_smoke_*/
cat results/h100_smoke_*/summary* 2>/dev/null
```

If any method's `<method>_jsons/` has fewer than 5 JSONs or the corresponding `*_run.log` shows a stack trace, fix before submitting Phase A.

Stop VLM servers after smoke test (sbatch will start its own):

```bash
bash scripts/stop_vlm_servers_local.sh
```

## Step 3: submit Phase A sbatch (4000 eps, ~2–4h)

```bash
mkdir -p logs/sbatch
sbatch scripts/h100_phase_a.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f logs/sbatch/vlfm_phase_a_*.out
```

Cancel if needed: `scancel <jobid>`

## Step 4: inspect results

When complete:

```bash
ls results/h100_phase_a_*/
cat results/h100_phase_a_*/summary.txt
cat results/h100_phase_a_*/alignment_*.txt
```

Per-method outputs:

- `results/h100_phase_a_<DATE>/semantic_jsons/`        (1000 JSONs)
- `results/h100_phase_a_<DATE>/cheapest_jsons/`        (1000 JSONs)
- `results/h100_phase_a_<DATE>/mc_risk_seed42_jsons/`  (1000 JSONs)
- `results/h100_phase_a_<DATE>/mc_risk_seed123_jsons/` (1000 JSONs)

Plus:

- `summary.txt` — per-method aggregate metrics (SR, SPL, etc.)
- `alignment_*.txt` — paired episode alignment verification (must show "Episode IDs match exactly")
- `<method>.log` — per-method full run log with `wall_time_seconds=...`

## Notes

- **GPU resource sharing**: VLM servers (~9 GB GPU mem) + 4 Habitat processes share one H100 (80 GB). Per-process Habitat: ~3 GB.
- **Bottleneck**: 4 Habitat processes hit the same 4 Flask servers concurrently. BLIP2-ITM is slowest; if throughput is below expectations, consider reducing parallelism to 2 in `run_h100_phase_a.sh`.
- **MC seed semantics**: `mc_seed_master` controls the master seed for `make_mc_sample_seeds(...)` in `vlfm/policy/itm_policy.py`. Different seeds → different MC sample sequences → different `mc_risk` decisions on the same episode.
- **Paired alignment**: all 4 processes evaluate the same first 1000 episodes (Habitat ObjectNav iterates deterministically given the same split + episode count). `scripts/check_episode_alignment.py` verifies via `scene_id:episode_id` keys.
- **`cheapest` fallback**: see `vlfm/policy/itm_policy.py:234-237` — falls back to nearest-frontier if grid-cost computation fails on a given step.

## sbatch troubleshooting

- If `--gres=gpu:H100:1` is rejected: try `--gres=gpu:1` and add `--constraint=H100` (or whatever the cluster uses).
- If the partition rejects the job: check `sinfo -p ice-gpu` for available time limits; this script asks for 6h.
- If conda activation fails: `which conda` should resolve after `module load anaconda3/2023.03`. If not, check `module avail | grep anaconda`.

## Phase B (later)

Phase B will be a sibling sbatch script (`scripts/h100_phase_b.sbatch` and `scripts/run_h100_phase_b.sh`, to be added when the next H100 allocation is available).
