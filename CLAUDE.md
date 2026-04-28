# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VLFM (Vision-Language Frontier Maps) — zero-shot ObjectGoal navigation. A Python 3.9 / PyTorch 1.12+cu113 research codebase evaluated in Habitat sim (HM3D/MP3D/Gibson) and deployed on Boston Dynamics Spot. See [README.md](README.md) for the paper and install flow.

## Common commands

Environment: conda env `vlfm` (or `VLFM_gpu` on this host). Install modes from [pyproject.toml](pyproject.toml): `pip install -e .[habitat]` for simulation, `.[reality]` for Spot, `.[dev]` for pre-commit/pytest.

- **Launch VLM servers** (required before any eval — starts GroundingDINO/BLIP2‑ITM/MobileSAM/YOLOv7 Flask servers on ports 12181–12184):
  - tmux version: [scripts/launch_vlm_servers.sh](scripts/launch_vlm_servers.sh)
  - local no-tmux version used on this host: [scripts/launch_vlm_servers_local.sh](scripts/launch_vlm_servers_local.sh) (PIDs in `logs/vlm_servers/pids/`, stop via [scripts/stop_vlm_servers_local.sh](scripts/stop_vlm_servers_local.sh))
- **Generate required dummy policy weights** (one-time; `vlfm.run` hard-fails without `data/dummy_policy.pth`): `python -m vlfm.utils.generate_dummy_policy`
- **Run HM3D ObjectNav eval**: `python -m vlfm.run` (entry point [vlfm/run.py](vlfm/run.py), Hydra config `config/experiments/vlfm_objectnav_hm3d`)
- **MP3D eval**: `python -m vlfm.run habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz`
- **Single-episode smoke test** used on this host: [scripts/run_vlfm_episode_local.sh](scripts/run_vlfm_episode_local.sh) (`SPLIT=val_mini EPISODES=1`, any extra args forwarded to Hydra)
- **Tests**: `pytest test/` — run one with `pytest test/test_setup.py::<name>`
- **Lint/format/type**: `pre-commit run --all-files` (ruff `--fix` + black + mypy strict, configured in [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml)). Line length 120. `check-added-large-files` caps at 200 KB — do not commit model weights, videos, or the teaser image beyond the explicit excludes.
- **Hydra overrides**: all run-time configuration is Hydra-based; pass `key=value` on the CLI (e.g. `habitat_baselines.eval.split=val_50`, `habitat_baselines.eval.video_option='["disk"]'`). See [scripts/eval_itm_policy.sh](scripts/eval_itm_policy.sh) for common overrides.

## Architecture

The pipeline is **observation → maps → value map → frontier selection → PointNav action**. Policy code does not train; it wraps a pretrained PointNav ResNet policy with VLM-driven waypoint selection. Key cross-cutting concerns:

**Entry & config wiring ([vlfm/run.py](vlfm/run.py))**. Imports `vlfm.policy.habitat_policies`, `vlfm.measurements.traveled_stairs`, `vlfm.obs_transformers.resize`, `vlfm.policy.action_replay_policy`, `vlfm.utils.vlfm_trainer` purely for their registration side effects (baseline_registry / Hydra ConfigStore). The experiment YAML [config/experiments/vlfm_objectnav_hm3d.yaml](config/experiments/vlfm_objectnav_hm3d.yaml) composes Habitat's `objectnav_hm3d` benchmark with extra `base_explorer` / `frontier_sensor` lab sensors and the `frontier_exploration_map` / `traveled_stairs` measurements. `habitat_baselines.rl.policy.name=HabitatITMPolicyV2` picks the concrete policy class. Habitat requires a checkpoint — `data/dummy_policy.pth` satisfies this without carrying real weights.

**Policy layering ([vlfm/policy/](vlfm/policy/))**. `BasePolicy` → `BaseObjectNavPolicy` (manages obstacle/object maps, wraps PointNav, runs VLM detectors) → `BaseITMPolicy` (adds `FrontierMap` + `ValueMap` + `AcyclicEnforcer` for frontier selection) → `HabitatITMPolicy[V2]` / reality policies. `BaseObjectNavPolicy` guards the habitat import with a `try/except` and falls back to a stub `BasePolicy` so the module still imports in Spot-only environments — preserve that pattern when adding shared code.

**Mapping ([vlfm/mapping/](vlfm/mapping/))**. `ObstacleMap` and `ObjectPointCloudMap` are built from depth + pose. `FrontierMap` extracts frontier waypoints from the obstacle map (via the external `frontier_exploration` package). `ValueMap` stores per-cell BLIP‑2 ITM cosine-similarity values scored against the target-object prompt; the policy picks the frontier with the highest value along the ray from the agent.

**VLM client/server split ([vlfm/vlm/](vlfm/vlm/))**. Each model has a `__main__`-runnable Flask server (GroundingDINO, BLIP2‑ITM, MobileSAM, YOLOv7) and a thin client used by the policy. This isolates heavy CUDA model loads from the Habitat process and lets them be shared across evaluator workers. When adding a new VLM, follow the [server_wrapper.py](vlfm/vlm/server_wrapper.py) pattern.

**Reality deployment ([vlfm/reality/](vlfm/reality/))**. Mirrors the Habitat stack but on Spot via `spot_wrapper` / `bosdyn`. Shares mapping + VLM code; only the env/robot I/O and `reality_policies.py` differ. Config: [config/experiments/reality.yaml](config/experiments/reality.yaml).

**Ports & paths convention**. VLM server ports are read from env vars (`GROUNDING_DINO_PORT` 12181, `BLIP2ITM_PORT` 12182, `SAM_PORT` 12183, `YOLOV7_PORT` 12184). Model weights live under `data/` (`mobile_sam.pt`, `groundingdino_swint_ogc.pth`, `yolov7-e6e.pt`, `pointnav_weights.pth`). HM3D scenes/episodes expected under `data/scene_datasets/hm3d` and `data/datasets/objectnav/hm3d`. `vlfm.run` asserts `data/` exists relative to cwd — run from the repo root.

## Check-in workflow (CSE 8803 UQ mid-project)

The active task is the mid-project check-in plan in [docs/checkin_codex.md](docs/checkin_codex.md) — a 12-step minimum deliverable loop: freeze original VLFM → audit components → implement a geometry-only frontier baseline → paired-episode comparison on HM3D `val_mini`. Organized into 5 layers:

- **L0 Baseline freeze** (STEP 0–1) ✅ done — tag `freeze-vlfm-smoketest-valmini1`, commit `41434c5`; single-episode wall-clock 30.47s → `N_EPISODES=30` ([repro/timing_baseline.txt](repro/timing_baseline.txt), [repro/smoke_test_command.txt](repro/smoke_test_command.txt)).
- **L1 Code audit** (STEP 2–4) — docs only: [repro/component_audit.md](repro/component_audit.md) ✅ done (all VLFM components ENABLED, value-map fusion = `weighted_avg` since `use_max_confidence=False`); `repro/baseline_plan.md`, `repro/logging_audit.md` pending.
- **L2 Baseline impl** (STEP 5) — add `frontier_selector: semantic|nearest|cheapest` Hydra switch. **Hard stop-loss: ≤3 files changed** (planned: `VLFMConfig` in [vlfm/policy/base_objectnav_policy.py](vlfm/policy/base_objectnav_policy.py), new branch in `_get_best_frontier` at [vlfm/policy/itm_policy.py:76](vlfm/policy/itm_policy.py#L76), default in [config/experiments/vlfm_objectnav_hm3d.yaml](config/experiments/vlfm_objectnav_hm3d.yaml)), else fall back to nearest-frontier.
- **L3 Paired experiments** (STEP 6–8.5) — fix `eval/pilot_episode_ids.txt` (30 eps), run both methods, `scripts/check_episode_alignment.py` must PASS before L4.
- **L4 Analysis & delivery** (STEP 9–12) — `results/pilot/summary.txt`, failure analysis, README, report table.

**Experiment-record rule**: every run must produce (1) exact command, (2) full `run.log` with `/usr/bin/time -f "wall_time_seconds=%e"`, (3) structured csv/md. End each layer with a 2–5 line `repro/stepN_journal.md`.

**Logging convention**: per-episode metrics come from the existing `ZSOS_LOG_DIR` JSON logger ([vlfm/utils/episode_stats_logger.py](vlfm/utils/episode_stats_logger.py) + [vlfm/utils/log_saver.py](vlfm/utils/log_saver.py)); aggregate with [scripts/parse_jsons.py](scripts/parse_jsons.py). Do not add a parallel CSV logger — reuse this pipeline for both VLFM and the baseline runs.

**Baseline scope discipline**: only touch the frontier selector. Mapping, detector switch, and low-level PointNav must stay identical across VLFM and baseline for a fair comparison.

**Note**: [docs/checkin_codex.md](docs/checkin_codex.md) is listed in `.git/info/exclude` — edits to that plan file will not show in `git status`. Intentional (keeps personal planning out of the repo history); do not `git add -f` it without asking.

## Gotchas

- The BLIP‑2 / LAVIS stack is pinned to `transformers==4.26.0`; bumping it silently breaks ITM scoring.
- `force_torch_single_threaded: True` in the eval config is intentional (perf + determinism with Habitat) — do not remove.
- The yolov7 source tree must be cloned as a sibling directory (`git clone git@github.com:WongKinYiu/yolov7.git` inside the repo) for the import in `vlfm/vlm/yolov7.py` to resolve.
