# STEP 8.6: Extended Evaluation Record

This note records the two evaluation scales used after STEP 8/8.5:

1. the original 30-episode pilot on `val_mini`
2. the extended 90-episode run on a fixed 5-scene subset of `val`

## A. 30-Episode Pilot

- Split: `val_mini`
- Scenes: `TEEsavR23oF`, `wcojb4TFT35`
- Episodes: `30`
- Alignment check: PASS
  - see `results/pilot/episode_alignment_check.txt`

### Original VLFM

- JSON dir: `results/pilot/vlfm_original_jsons`
- Run log: `results/pilot/vlfm_original_run.log`
- Success: `53.33% (16/30)`
- SPL: `32.12%`
- Soft SPL: `33.82%`
- Wall time: `1354.40s`

### Geometry Baseline

- JSON dir: `results/pilot/greedy_frontier_jsons`
- Run log: `results/pilot/greedy_frontier_run.log`
- Frontier selector: `cheapest`
- Success: `53.33% (16/30)`
- SPL: `30.20%`
- Soft SPL: `35.64%`
- Wall time: `943.72s`

### Paired Comparison

- Mean Delta Success (VLFM - Baseline): `+0.000`
- Mean Delta SPL (VLFM - Baseline): `+0.019`
- Mean Delta SoftSPL (VLFM - Baseline): `-0.018`
- Success flips: `4`

## B. 90-Episode / 5-Scene Extended Run

- Source split: `val`
- Derived split: `val_5scene_90`
- Episodes: `90`
- Scenes: `5`
- Sampling rule: `18` episodes per scene
- Sampling mode: `random`
- Seed: `20260413`
- Scene list:
  - `TEEsavR23oF`
  - `wcojb4TFT35`
  - `4ok3usBNeis`
  - `5cdEh9F2hJL`
  - `6s7QHgap2fW`
- Split manifest:
  - `data/datasets/objectnav/hm3d/v1/val_5scene_90/manifest.json`

### Run Outputs

- Original VLFM JSON dir: `results/val5scene90/vlfm_original_jsons`
- Original VLFM log: `results/val5scene90/vlfm_original_run.log`
- Geometry baseline JSON dir: `results/val5scene90/greedy_frontier_jsons`
- Geometry baseline log: `results/val5scene90/greedy_frontier_run.log`
- Alignment check:
  - `results/val5scene90/episode_alignment_check.txt`
- Summary:
  - `results/val5scene90/summary.txt`

### Final Results

Alignment check: `PASS`

- `scene_id + episode_id` matched exactly
- Count: `90`

### Original VLFM (`semantic`)

- Success: `51.11% (46/90)`
- SPL: `30.04%`
- Soft SPL: `36.79%`
- Wall time: `4663.42s`

### Geometry Baseline (`cheapest`)

- Success: `50.00% (45/90)`
- SPL: `29.88%`
- Soft SPL: `36.58%`
- Wall time: `3368.81s`

### Paired Comparison

- Mean Delta Success (VLFM - Baseline): `+0.011`
- Mean Delta SPL (VLFM - Baseline): `+0.002`
- Mean Delta SoftSPL (VLFM - Baseline): `+0.002`
- Success flips: `9`

### Resume Notes

To make long runs resumable, the following resume-safe patches were added:

- `vlfm/utils/vlfm_trainer.py`
  - skip already-evaluated episodes when `ZSOS_LOG_DIR` is set and `num_envs == 1`
- `vlfm/utils/log_saver.py`
  - normalize `scene_id` consistently in both `log_episode()` and `is_evaluated()`

This allows the 90-episode run to be restarted without recomputing already-logged
episodes.
