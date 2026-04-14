# VLN-UQ: Uncertainty-Aware Zero-Shot ObjectNav Check-in

This repository is a course check-in workspace built on top of the official
[VLFM](https://github.com/bdaiinstitute/vlfm) codebase:
Vision-Language Frontier Maps for Zero-Shot Semantic Navigation
([paper](https://arxiv.org/abs/2312.03275)).

The current goal is modest and concrete: keep a working VLFM baseline,
add a geometry-only frontier baseline, and compare them on matched HM3D
ObjectNav episodes before moving on to uncertainty-aware methods.

## What This Repo Adds

- A `frontier_selector` switch with `semantic`, `nearest`, and `cheapest`
  options
- A geometry-only cheapest-frontier baseline for paired comparison
- Resume-safe single-environment evaluation with `ZSOS_LOG_DIR`
- Local experiment scripts for alignment checks and paired summaries
- Repro artifacts for the smoke test, pilot run, and extended 90-episode run

## Current Status

- Official VLFM baseline runs locally in Habitat
- Local VLM server launch flow is available via
  `scripts/launch_vlm_servers_local.sh`
- HM3D `val_mini` pilot comparison is complete
- A fixed 5-scene / 90-episode extended comparison is complete
- Failure-case notes are recorded for the first check-in pass

## Main Results

### Pilot Comparison: 30 Episodes on `val_mini`

| Method | Episodes | Success | SPL | SoftSPL |
|---|---:|---:|---:|---:|
| VLFM Original | 30 | 53.33% | 32.12% | 33.82% |
| Greedy Frontier (`cheapest`) | 30 | 53.33% | 30.20% | 35.64% |

- Paired delta success: `+0.000`
- Paired delta SPL: `+0.019`
- Success flips: `4`

### Extended Comparison: 90 Episodes on `val_5scene_90`

| Method | Episodes | Success | SPL | SoftSPL |
|---|---:|---:|---:|---:|
| VLFM Original | 90 | 51.11% | 30.04% | 36.79% |
| Greedy Frontier (`cheapest`) | 90 | 50.00% | 29.88% | 36.58% |

- Paired delta success: `+0.011`
- Paired delta SPL: `+0.002`
- Paired delta SoftSPL: `+0.002`
- Success flips: `9`
- Wall time:
  - VLFM Original: `4663.42s`
  - Greedy Frontier: `3368.81s`

The main check-in takeaway is that the semantic policy is only slightly better
than the geometry-only baseline on the 90-episode matched split, while the
baseline runs noticeably faster. That leaves a useful opening for uncertainty-
aware ranking or stop-gating to earn its keep.

## Important Paths

- Pilot summary:
  - `results/pilot/summary.txt`
- Pilot failure cases:
  - `results/pilot/failure_cases/failure_analysis.md`
- Extended run summary:
  - `results/val5scene90/summary.txt`
- Extended run alignment check:
  - `results/val5scene90/episode_alignment_check.txt`
- Extended experiment record:
  - `repro/step8_6_experiment_log.md`
- Component audit:
  - `repro/component_audit.md`

## Local Environment Notes

Two conda environments were used during setup:

- `VLFM`
  - closest to the original dependency set
  - good for compatibility and CPU-side checks
- `VLFM_gpu`
  - practical local environment for the RTX 5060 Ti
  - recommended for actual evaluation runs in this workspace

## Quick Start

### 1. Activate the GPU-ready environment

```bash
conda activate VLFM_gpu
cd /path/to/vlfm-main
```

### 2. Start the local VLM servers

```bash
bash scripts/launch_vlm_servers_local.sh
```

When done:

```bash
bash scripts/stop_vlm_servers_local.sh
```

### 3. Smoke test

```bash
EPISODES=1 scripts/run_vlfm_episode_local.sh
```

### 4. Pilot comparison

```bash
ZSOS_LOG_DIR=results/pilot/vlfm_original_jsons \
EPISODES=30 scripts/run_vlfm_episode_local.sh \
  habitat_baselines.rl.policy.frontier_selector=semantic

ZSOS_LOG_DIR=results/pilot/greedy_frontier_jsons \
EPISODES=30 scripts/run_vlfm_episode_local.sh \
  habitat_baselines.rl.policy.frontier_selector=cheapest
```

### 5. Extended 90-episode comparison

```bash
bash scripts/run_step8_6_90.sh
```

### 6. Summarize results

```bash
python scripts/parse_jsons.py results/pilot/vlfm_original_jsons
python scripts/parse_jsons.py results/pilot/greedy_frontier_jsons
python scripts/summarize_paired.py \
  --vlfm-dir results/pilot/vlfm_original_jsons \
  --baseline-dir results/pilot/greedy_frontier_jsons

python scripts/parse_jsons.py results/val5scene90/vlfm_original_jsons
python scripts/parse_jsons.py results/val5scene90/greedy_frontier_jsons
python scripts/summarize_paired.py \
  --vlfm-dir results/val5scene90/vlfm_original_jsons \
  --baseline-dir results/val5scene90/greedy_frontier_jsons
```

## Data and Weights

This repository does not store HM3D scenes, dataset archives, or large model
weights. Download them separately into `data/` as described by the original
VLFM / Habitat setup flow.

## Repository Layout

```text
configs/                 Habitat and policy configs
eval/                    Episode manifests and protocols
repro/                   Reproducibility notes and experiment records
results/pilot/           30-episode pilot outputs
results/val5scene90/     90-episode extended outputs
scripts/                 Run, analysis, and helper scripts
vlfm/                    Main policy and trainer code
```

## Upstream Attribution

This repository started from the official VLFM implementation by Naoki
Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, and Bernadette Bucher.
Please cite the original VLFM paper if you use the underlying method.
