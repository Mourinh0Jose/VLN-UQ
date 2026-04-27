# VLFM MC Risk-Aware Frontier Re-Ranking — Final Execution Plan v4

## 0. Purpose

This document is the final implementation plan for adding a lightweight Monte Carlo (MC) uncertainty layer to the existing VLFM ObjectNav pipeline.

The goal is **not** to redesign VLFM, retrain a VLM, or change the benchmark. The goal is to keep the validated VLFM baseline and modify only the frontier ranking / selection stage:

```text
VLFM baseline frontier scoring
→ keep top-K frontier candidates
→ estimate local score uncertainty by Monte Carlo perturbation
→ re-rank by risk-aware score J = μ - λσ
→ pass the re-ranked frontier list to the existing cyclic / stick-to-last logic
```

The implementation should be small, testable, and compatible with the current `HabitatITMPolicyV2` path.

---

## 1. Current Codebase Facts

Based on repo inspection:

```text
Policy path: HabitatITMPolicyV2
Frontier type: np.ndarray, shape [N, 2]
Frontier coordinates: world / episodic (x, y), not grid row/col
Value map: np.ndarray, shape [H, W, C], usually around [1000, 1000, value_channels]
Baseline frontier scoring: ValueMap.sort_waypoints()
Final selection: BaseITMPolicy._get_best_frontier()
Baseline local aggregation: pixel_value_within_radius(..., reduction="median")
```

Important consequences:

1. Do **not** treat a frontier `(x, y)` as a map array index.
2. Convert world `(x, y)` to value-map pixel coordinates using the same logic as `ValueMap.sort_waypoints()`.
3. Match the baseline scoring semantics: circular local region → positive values only → median.
4. Do **not** assume the BLIP2-ITM / value-map score is a calibrated probability or bounded in `[0, 1]`.
5. First implementation assumes a single semantic value channel, or explicitly uses `channel_idx = 0`.

---

## 2. Minimal Data and Runtime Setup

For the first validation stage, use HM3D validation scenes and HM3D ObjectNav validation episodes.

Required data:

```text
HM3D val scenes:
data/scene_datasets/hm3d/val/

HM3D ObjectNav val episodes:
data/datasets/objectnav/hm3d/v1/val/
```

First debug target:

```text
Run original VLFM baseline on 10 validation episodes.
Confirm frontier generation, scoring, selection, and metric logging.
```

Do not start with the full 2000-episode validation benchmark. Start with 10–50 episodes, then scale up.

---

## 3. Mathematical Definition

At a decision step, let the candidate frontier set be:

\[
\mathcal{F} = \{f_1, f_2, \dots, f_M\}.
\]

The original VLFM semantic selector gives each frontier a deterministic baseline score:

\[
s_i = S(f_i; V),
\]

where `V` is the current value map and `S` is the local median scoring rule.

We first keep only the top-K frontiers according to the baseline score:

\[
\mathcal{F}_K = \text{TopK}_{f_i \in \mathcal{F}}\ S(f_i; V).
\]

For each frontier \(f_i \in \mathcal{F}_K\), define a perturbed score sample:

\[
s_i^{(n)} = S(f_i; \widetilde{V}^{(n)}), \quad n=1,\dots,N.
\]

In this implementation, the perturbation is not applied to raw RGB and not to a full rollout. It is applied to the **local positive value evidence set** used by the baseline median scorer.

Estimate:

\[
\hat{\mu}_i = \frac{1}{N}\sum_{n=1}^{N}s_i^{(n)},
\]

\[
\hat{\sigma}_i = \sqrt{\frac{1}{N-1}\sum_{n=1}^{N}\left(s_i^{(n)} - \hat{\mu}_i\right)^2}.
\]

Risk-aware score:

\[
J_i = \hat{\mu}_i - \lambda \hat{\sigma}_i.
\]

Final MC re-ranking:

\[
f^* = \arg\max_{f_i \in \mathcal{F}_K} J_i.
\]

Interpretation:

```text
μ: average semantic value under local evidence perturbation
σ: instability of that semantic value
J = μ - λσ: downside-aware frontier score
```

When `λ = 0`, the method becomes MC mean-only selection. When `λ > 0`, unstable frontiers are penalized.

---

## 4. Default Hyperparameters

Use these for the first implementation:

```text
top_k = 3
n_samples = 5
keep_prob = 0.8
noise_rel = 0.1
lambda = 1.0
radius_m = 0.5
channel_idx = 0
master_seed = 12345
min_values_for_mc = 3
```

Ablations later:

```text
lambda ∈ {0, 0.5, 1.0, 2.0}
n_samples ∈ {5, 10}
top_k ∈ {3, 5}
noise_rel ∈ {0.0, 0.1, 0.2}
```

---

## 5. Strict Implementation Order

Do **not** implement the whole method at once. Use this order:

```text
1. Add ValueMap._xy_to_value_map_px()
2. Add ValueMap._positive_values_within_radius()
3. Run helper consistency test against ValueMap.sort_waypoints()
4. Only after the consistency test passes, add ValueMap.mc_samples_at_waypoint()
5. Add ValueMap.mc_summary_at_waypoint()
6. Add MC RNG / SeedSequence helper in policy
7. Add config variables
8. Add selector name "mc_risk_aware"
9. Add BaseITMPolicy._get_best_frontier() branch
10. Add HabitatITMPolicyV2._sort_frontiers_mc_risk_aware()
11. Run one-episode debug with print logs
12. Add structured logging
13. Run small evaluation: 10–50 episodes
```

The consistency test is mandatory. If it fails, do not continue to MC sampling.

---

## 6. ValueMap Helper Functions

Add these methods inside the `ValueMap` class, likely in `vlfm/mapping/value_map.py`.

### 6.1 Convert world `(x, y)` to value-map pixel

This must match the coordinate transform used inside `ValueMap.sort_waypoints()`.

```python
from typing import Tuple
import numpy as np


def _xy_to_value_map_px(self, point: np.ndarray) -> Tuple[int, int]:
    """Convert a world/episodic (x, y) waypoint to value-map pixel coordinates.

    This must match the transform used in ValueMap.sort_waypoints().
    """
    x, y = point

    px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
    py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]

    point_px = (self._value_map.shape[0] - px, py)
    return int(point_px[0]), int(point_px[1])
```

### 6.2 Extract positive values within the baseline radius

This helper should return the same evidence set used by `pixel_value_within_radius()` before median reduction.

```python
def _positive_values_within_radius(
    self,
    image: np.ndarray,
    pixel_location: Tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """Extract positive local values inside a circular radius.

    This mirrors the evidence extraction used by pixel_value_within_radius():
    circular crop, then keep only values > 0.
    """
    row, col = int(pixel_location[0]), int(pixel_location[1])
    h, w = image.shape[:2]
    r = int(radius_px)

    row_min = max(row - r, 0)
    row_max = min(row + r + 1, h)
    col_min = max(col - r, 0)
    col_max = min(col + r + 1, w)

    cropped = image[row_min:row_max, col_min:col_max]

    yy, xx = np.ogrid[row_min:row_max, col_min:col_max]
    circle_mask = (yy - row) ** 2 + (xx - col) ** 2 <= r**2

    overlap_values = cropped[circle_mask]
    overlap_values = overlap_values[overlap_values > 0]

    return overlap_values.astype(np.float32)
```

If the consistency test below fails, copy the exact crop / mask logic from the repo's `pixel_value_within_radius()` implementation and adapt it to return `overlap_values` instead of the reduced scalar.

---

## 7. Mandatory Consistency Test Before MC

Before implementing MC, verify that the helper path gives exactly the same value as the baseline `sort_waypoints()` path.

Add this temporary debug method to `ValueMap` or use it as a standalone debug function.

```python
def debug_check_value_map_helper_consistency(
    self,
    point: np.ndarray,
    radius_m: float = 0.5,
    channel_idx: int = 0,
    atol: float = 1e-5,
) -> None:
    """Check helper-based median equals baseline sort_waypoints value."""

    # Baseline path.
    _, baseline_values = self.sort_waypoints(np.array([point]), radius_m)
    baseline_val = baseline_values[0]

    # First implementation handles one channel explicitly.
    if isinstance(baseline_val, tuple):
        baseline_val = baseline_val[channel_idx]

    # Helper path.
    radius_px = int(radius_m * self.pixels_per_meter)
    point_px = self._xy_to_value_map_px(point)
    values = self._positive_values_within_radius(
        self._value_map[..., channel_idx],
        point_px,
        radius_px,
    )

    if values.size == 0:
        helper_val = -1.0
    else:
        helper_val = float(np.median(values))

    print(
        "[MC helper check]",
        f"point={point}",
        f"point_px={point_px}",
        f"baseline={float(baseline_val):.6f}",
        f"helper={helper_val:.6f}",
        f"num_values={values.size}",
    )

    assert abs(float(baseline_val) - float(helper_val)) < atol, (
        f"ValueMap helper mismatch: baseline={baseline_val}, "
        f"helper={helper_val}, point={point}, point_px={point_px}"
    )
```

Temporary use inside `_sort_frontiers_by_value()` or near the frontier scoring call:

```python
if len(frontiers) > 0:
    for point in frontiers[:3]:
        self._value_map.debug_check_value_map_helper_consistency(point)
```

Only continue when the printout repeatedly shows:

```text
baseline == helper
```

within tolerance.

---

## 8. MC Sampling Helpers in ValueMap

After the consistency test passes, add MC sampling methods.

### 8.1 MC samples at one waypoint

Perturbation is applied to the **positive evidence list**, not to the full 2D spatial patch.

```python
from typing import Iterable, List, Optional
import numpy as np


def mc_samples_at_waypoint(
    self,
    point: np.ndarray,
    radius_m: float = 0.5,
    n_samples: int = 5,
    keep_prob: float = 0.8,
    noise_rel: float = 0.1,
    sample_seeds: Optional[Iterable[int]] = None,
    channel_idx: int = 0,
    min_values_for_mc: int = 3,
) -> List[float]:
    """Return MC-perturbed local median scores at a waypoint.

    The baseline score is median(positive values in radius).
    MC samples are generated by Bernoulli subsampling that positive-value list
    and adding relative Gaussian noise scaled by the local value std.
    """
    radius_px = int(radius_m * self.pixels_per_meter)
    point_px = self._xy_to_value_map_px(point)

    values = self._positive_values_within_radius(
        self._value_map[..., channel_idx],
        point_px,
        radius_px,
    )

    # Case 1: no semantic evidence. Match baseline behavior.
    if values.size == 0:
        return [-1.0] * n_samples

    baseline_val = float(np.median(values))

    # Case 2: too few positive values for meaningful subsampling.
    # Fall back to baseline samples to avoid fake uncertainty.
    if values.size < min_values_for_mc:
        return [baseline_val] * n_samples

    local_std = float(np.std(values))
    noise_std = noise_rel * max(local_std, 1e-6)

    if sample_seeds is None:
        sample_seeds = list(range(n_samples))
    else:
        sample_seeds = list(sample_seeds)

    if len(sample_seeds) != n_samples:
        raise ValueError(
            f"sample_seeds length {len(sample_seeds)} != n_samples {n_samples}"
        )

    samples: List[float] = []

    for seed in sample_seeds:
        rng = np.random.default_rng(int(seed))

        keep_mask = rng.random(values.shape[0]) < keep_prob
        sampled_values = values[keep_mask]

        # Case 3: subsampling happened to remove all values.
        # Do not append -1.0 here; that would create artificial risk.
        if sampled_values.size == 0:
            samples.append(baseline_val)
            continue

        if noise_rel > 0.0:
            sampled_values = sampled_values + rng.normal(
                loc=0.0,
                scale=noise_std,
                size=sampled_values.shape,
            )

        # Keep baseline semantics: use positive evidence only.
        sampled_values = sampled_values[sampled_values > 0]

        if sampled_values.size == 0:
            samples.append(baseline_val)
        else:
            samples.append(float(np.median(sampled_values)))

    return samples
```

### 8.2 MC summary at one waypoint

```python
from typing import Dict, List


def mc_summary_at_waypoint(
    self,
    point: np.ndarray,
    radius_m: float = 0.5,
    n_samples: int = 5,
    keep_prob: float = 0.8,
    noise_rel: float = 0.1,
    lambda_risk: float = 1.0,
    sample_seeds: Optional[Iterable[int]] = None,
    channel_idx: int = 0,
    min_values_for_mc: int = 3,
) -> Dict[str, object]:
    """Compute MC mean, std, and risk-aware score at a waypoint."""
    samples = self.mc_samples_at_waypoint(
        point=point,
        radius_m=radius_m,
        n_samples=n_samples,
        keep_prob=keep_prob,
        noise_rel=noise_rel,
        sample_seeds=sample_seeds,
        channel_idx=channel_idx,
        min_values_for_mc=min_values_for_mc,
    )

    arr = np.asarray(samples, dtype=np.float32)
    mu = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    risk_score = float(mu - lambda_risk * std)

    return {
        "samples": [float(x) for x in samples],
        "mean": mu,
        "std": std,
        "risk_score": risk_score,
    }
```

---

## 9. Reproducible RNG Design

Use `numpy.random.SeedSequence` to generate stable per-decision MC seeds.

Requirements:

```text
Same episode + same decision step → same MC seeds
Same decision step → all top-K frontiers share the same seeds
Different decision steps → different seeds
Single-episode reruns should not depend on whether previous episodes were run
```

Avoid Python's built-in `hash()` because it is not stable across sessions.

Add this helper near the policy class or in a small utility file:

```python
import zlib
from typing import List
import numpy as np


def _stable_int_from_episode_id(episode_id: object) -> int:
    """Convert an episode id into a stable uint32 seed component."""
    return zlib.crc32(str(episode_id).encode("utf-8"))


def make_mc_sample_seeds(
    master_seed: int,
    episode_id: object,
    decision_counter: int,
    n_samples: int,
) -> List[int]:
    """Generate reproducible MC seeds for one decision step.

    All candidate frontiers at this decision step should use this same seed list
    following the common-random-numbers principle.
    """
    episode_seed = _stable_int_from_episode_id(episode_id)
    seed_seq = np.random.SeedSequence(
        [int(master_seed), int(episode_seed), int(decision_counter)]
    )
    return seed_seq.generate_state(n_samples, dtype=np.uint32).astype(int).tolist()
```

Inside the policy, initialize:

```python
self._mc_seed_master = 12345
self._mc_episode_id = "unknown"
self._mc_decision_counter = 0
```

At episode reset, call:

```python
def reset_mc_for_episode(self, episode_id: object) -> None:
    """Reset MC counter for reproducibility at the start of each episode."""
    self._mc_episode_id = episode_id
    self._mc_decision_counter = 0
```

If the true Habitat episode id is not immediately available, use a temporary monotonically increasing episode counter for debugging. Before final evaluation, connect this to the actual episode id if possible.

Per decision step:

```python
sample_seeds = make_mc_sample_seeds(
    master_seed=self._mc_seed_master,
    episode_id=self._mc_episode_id,
    decision_counter=self._mc_decision_counter,
    n_samples=self._mc_n_samples,
)
self._mc_decision_counter += 1
```

This gives **common random numbers** across frontier candidates within a decision step while preserving reproducibility across episodes.

---

## 10. Config Additions

Add config fields or class attributes:

```python
self._mc_top_k = 3
self._mc_n_samples = 5
self._mc_keep_prob = 0.8
self._mc_noise_rel = 0.1
self._mc_lambda = 1.0
self._mc_radius_m = 0.5
self._mc_channel_idx = 0
self._mc_min_values_for_mc = 3
self._mc_seed_master = 12345
self._mc_debug = False
```

For the first implementation, do not over-parameterize. Hard-coded defaults are acceptable during debug; move to config after the method works.

---

## 11. Selector Branch

### 11.1 Allow selector name

Find `_get_frontier_selector()` or the equivalent selector validation code.

Current allowed set likely resembles:

```python
{"semantic", "nearest", "cheapest"}
```

Change to:

```python
{"semantic", "nearest", "cheapest", "mc_risk_aware"}
```

### 11.2 Add branch in `BaseITMPolicy._get_best_frontier()`

Modify the selection branch:

```python
if selector == "semantic":
    sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
elif selector == "mc_risk_aware":
    sorted_pts, sorted_values = self._sort_frontiers_mc_risk_aware(observations, frontiers)
else:
    sorted_pts, sorted_values = self._sort_frontiers_geometric(frontiers, robot_xy, selector)
```

Keep the existing cyclic suppression and stick-to-last-frontier logic unchanged.

---

## 12. MC Risk-Aware Re-Ranking in HabitatITMPolicyV2

Add this method to `HabitatITMPolicyV2`.

```python
from typing import List, Tuple
import numpy as np


def _sort_frontiers_mc_risk_aware(
    self,
    observations: "TensorDict",
    frontiers: np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """Sort frontiers by MC risk-aware score.

    Step 1: use baseline VLFM scoring to sort all frontiers.
    Step 2: apply MC re-ranking only to top-K.
    Step 3: return a full sorted list so downstream cyclic logic remains intact.
    """
    if len(frontiers) == 0:
        return frontiers, []

    # Baseline semantic ranking.
    baseline_pts, baseline_values = self._value_map.sort_waypoints(frontiers, 0.5)

    # If baseline uses multi-channel values, first implementation only handles scalar.
    # Use channel 0 if needed.
    def to_scalar(v):
        return float(v[0]) if isinstance(v, tuple) else float(v)

    baseline_values_scalar = [to_scalar(v) for v in baseline_values]

    top_k = min(self._mc_top_k, len(baseline_pts))
    top_pts = baseline_pts[:top_k]
    remaining_pts = baseline_pts[top_k:]
    remaining_baseline_values = baseline_values_scalar[top_k:]

    sample_seeds = make_mc_sample_seeds(
        master_seed=self._mc_seed_master,
        episode_id=getattr(self, "_mc_episode_id", "unknown"),
        decision_counter=getattr(self, "_mc_decision_counter", 0),
        n_samples=self._mc_n_samples,
    )
    self._mc_decision_counter = getattr(self, "_mc_decision_counter", 0) + 1

    top_records = []
    for rank, point in enumerate(top_pts):
        summary = self._value_map.mc_summary_at_waypoint(
            point=point,
            radius_m=self._mc_radius_m,
            n_samples=self._mc_n_samples,
            keep_prob=self._mc_keep_prob,
            noise_rel=self._mc_noise_rel,
            lambda_risk=self._mc_lambda,
            sample_seeds=sample_seeds,
            channel_idx=self._mc_channel_idx,
            min_values_for_mc=self._mc_min_values_for_mc,
        )

        record = {
            "point": point,
            "baseline_rank": rank,
            "baseline_value": baseline_values_scalar[rank],
            "mc_samples": summary["samples"],
            "mc_mean": summary["mean"],
            "mc_std": summary["std"],
            "risk_score": summary["risk_score"],
        }
        top_records.append(record)

    # Re-rank top-K by risk score.
    top_records = sorted(top_records, key=lambda r: r["risk_score"], reverse=True)

    reranked_top_pts = [r["point"] for r in top_records]
    reranked_top_values = [float(r["risk_score"]) for r in top_records]

    # Safer default: avoid mixing risk-score scale and baseline-score scale.
    # Keep all remaining frontiers below the risk-re-ranked top-K while preserving their order.
    if len(remaining_pts) > 0:
        if len(reranked_top_values) > 0:
            min_top_risk = min(reranked_top_values)
        else:
            min_top_risk = 0.0
        remaining_values = [
            float(min_top_risk - 0.01 - 0.001 * i)
            for i in range(len(remaining_baseline_values))
        ]
    else:
        remaining_values = []

    sorted_pts = np.array(reranked_top_pts + list(remaining_pts))
    sorted_values = reranked_top_values + remaining_values

    if getattr(self, "_mc_debug", False):
        print("[MC risk-aware] seeds:", sample_seeds)
        for i, r in enumerate(top_records):
            print(
                f"  rank={i}",
                f"baseline_rank={r['baseline_rank']}",
                f"baseline={r['baseline_value']:.4f}",
                f"mu={r['mc_mean']:.4f}",
                f"std={r['mc_std']:.4f}",
                f"risk={r['risk_score']:.4f}",
                f"samples={r['mc_samples']}",
            )

    # Optional: store top_records for logging by the caller.
    self._last_mc_frontier_records = top_records

    return sorted_pts, sorted_values
```

Notes:

1. This method uses CRN by passing the same `sample_seeds` to every top-K frontier at one decision step.
2. Remaining frontiers are placed below the re-ranked top-K in a consistent risk-score scale.
3. If cyclic suppression behaves strangely, log `top_two_values` and compare baseline vs MC method.

---

## 13. Debug Protocol

### 13.1 One-step print debug

For the first run, use:

```text
top_k = 3
n_samples = 5
lambda = 1
noise_rel = 0.1
mc_debug = True
```

Expected print per decision:

```text
[MC risk-aware] seeds: [ ... ]
  rank=0 baseline_rank=1 baseline=0.3481 mu=0.3402 std=0.0121 risk=0.3281 samples=[...]
  rank=1 baseline_rank=0 baseline=0.3520 mu=0.3300 std=0.0400 risk=0.2900 samples=[...]
  rank=2 baseline_rank=2 baseline=0.3005 mu=0.3020 std=0.0060 risk=0.2960 samples=[...]
```

You want to see:

```text
MC samples are close to baseline scale
std is not always zero
risk ranking sometimes differs from baseline ranking
no NaN / inf values
```

### 13.2 Sanity checks

Run these checks on a few decision steps:

```text
1. Helper consistency test passes.
2. If noise_rel = 0 and keep_prob = 1, then MC samples equal baseline value.
3. If lambda = 0, ranking uses MC mean only.
4. If all local evidence is sparse values.size < 3, MC samples collapse to baseline value.
5. If all local evidence is empty, score remains -1 like baseline.
6. Cyclic suppression frequency is not wildly different from baseline.
```

---

## 14. Logging Schema

Log enough information to answer: Did MC change the selected frontier? Was the changed decision helpful?

### 14.1 Per decision step

```json
{
  "episode_id": "...",
  "decision_step": 12,
  "selector": "mc_risk_aware",
  "num_frontiers": 8,
  "mc_top_k": 3,
  "mc_n_samples": 5,
  "mc_keep_prob": 0.8,
  "mc_noise_rel": 0.1,
  "mc_lambda": 1.0,
  "sample_seeds": [123, 456, 789, 101112, 131415],
  "baseline_top_frontier_xy": [1.2, -0.4],
  "mc_top_frontier_xy": [0.9, -0.2],
  "final_selected_frontier_xy": [0.9, -0.2],
  "changed_from_baseline": true,
  "cyclic_suppressed": false,
  "top_two_values": [0.3281, 0.2960]
}
```

### 14.2 Per top-K frontier record

```json
{
  "episode_id": "...",
  "decision_step": 12,
  "frontier_rank_baseline": 0,
  "frontier_xy": [1.2, -0.4],
  "baseline_value": 0.3520,
  "mc_samples": [0.35, 0.31, 0.36, 0.29, 0.34],
  "mc_mean": 0.3300,
  "mc_std": 0.0400,
  "risk_score": 0.2900,
  "selected_by_baseline": true,
  "selected_by_mc_risk": false
}
```

### 14.3 Per episode summary

```json
{
  "episode_id": "...",
  "method": "mc_risk_aware",
  "success": true,
  "spl": 0.42,
  "path_length": 18.3,
  "collision_count": 3,
  "timeout": false,
  "num_decisions": 21,
  "num_changed_decisions": 5,
  "avg_selected_mc_std": 0.018,
  "avg_all_topk_mc_std": 0.023,
  "cyclic_suppression_count": 2
}
```

---

## 15. Experimental Plan

### 15.1 Methods

Run the same episode subset for each method.

```text
A. VLFM baseline
   selector = semantic

B. MC mean only
   selector = mc_risk_aware
   lambda = 0

C. MC risk-aware
   selector = mc_risk_aware
   lambda = 1

D. Optional risk sweep
   lambda ∈ {0.5, 2.0}
```

### 15.2 First scale

```text
Debug: 1 episode
Small run: 10 episodes
Initial result: 50 episodes
Course report result: 100–200 episodes if time allows
```

Do not run full validation until the method is stable.

### 15.3 Metrics

```text
Navigation:
- success rate
- SPL
- path length
- collision count/rate
- timeout rate

UQ / decision:
- average selected mc_std
- average top-K mc_std
- number of changed decisions
- changed-decision success/failure breakdown
- uncertainty–failure correlation

Compute:
- average decision time
- average episode wall-clock time
- overhead relative to baseline
```

---

## 16. UQ Diagnostics

### 16.1 MC std histogram

Plot distribution of `mc_std` over all top-K frontiers.

Questions:

```text
Is std always near zero?
Is std sometimes extremely large?
Are failed episodes associated with higher std?
```

### 16.2 Changed-decision analysis

For steps where MC risk-aware chooses a different frontier from baseline:

```text
baseline frontier: high μ but high σ?
risk-aware frontier: slightly lower μ but much lower σ?
```

This directly supports the project story.

### 16.3 Uncertainty–failure correlation diagnostic

For each episode:

```text
avg_selected_std = mean mc_std of the frontier selected at each decision
outcome = 1 if episode succeeds else 0
```

Then:

```text
1. Bin episodes by avg_selected_std into 4–5 bins.
2. Compute success rate per bin.
3. Plot success rate vs avg_selected_std.
```

Hypothesis:

```text
Higher avg_selected_std should correspond to lower success rate.
```

If true, this suggests MC std is a meaningful uncertainty signal. If false, report it honestly: the perturbation-based std captures local score instability but may not predict full-episode navigation failure.

### 16.4 Cyclic suppression diagnostic

Compare baseline and MC method:

```text
cyclic_suppression_count per episode
frequency of suppressed frontier
whether MC method triggers abnormal cyclic suppression
```

If abnormal, inspect `top_two_values` and consider adjusting the returned `sorted_values` scale.

---

## 17. Expected Outcomes

Expected positive pattern:

```text
- MC risk-aware changes some but not all decisions.
- Changed decisions often avoid high-variance frontier choices.
- Success rate or timeout/collision rate improves slightly.
- SPL may improve, stay similar, or mildly decrease depending on conservatism.
```

Expected negative pattern:

```text
- MC std has no relation to failure.
- Risk-aware selection changes decisions but does not improve metrics.
- Compute overhead is too large for small gains.
```

Both are reportable. The key is to show a principled UQ diagnostic rather than only reporting SR/SPL.

---

## 18. Report Method Text

You can reuse this in the final report:

> We introduce a lightweight Monte Carlo uncertainty estimator for VLFM-style frontier selection. For each decision step, the original VLFM value map first ranks all frontier candidates using its standard local median scoring rule. We retain only the top-K candidates and estimate the stability of each candidate's score by perturbing the same positive local value-map evidence used by the baseline scorer. Specifically, we generate Monte Carlo samples by Bernoulli subsampling this local evidence set and adding relative Gaussian noise scaled by the local score standard deviation. The resulting sample mean and standard deviation define a downside-aware score \(J = \mu - \lambda\sigma\), where \(\lambda\) controls risk sensitivity. Following the common-random-numbers principle, all top-K candidates at a given decision step share the same perturbation seeds, ensuring that comparison reflects genuine differences in frontier stability rather than perturbation noise.

IUQ framing:

> The proposed perturbation-based estimator can be viewed as a non-parametric Monte Carlo approximation to the variance of a noisy, non-differentiable downstream functional: the median value-map score. While it is not a fully Bayesian posterior, it provides a tractable local variance signal at the decision boundary. We use this signal to construct a downside-aware acquisition rule \(J = \mu - \lambda\sigma\), connecting uncertainty propagation to risk-sensitive decision making.

---

## 19. Limitations

State these clearly:

1. The MC perturbation is not a calibrated posterior over VLM predictions.
2. The method estimates local score instability, not full trajectory uncertainty.
3. The first version handles a single value-map channel or channel `0` only.
4. The risk-aware score can be overly conservative for large `λ`.
5. MC overhead grows with `top_k × n_samples × number_of_decisions`.
6. If `mc_std` does not correlate with failure, more principled uncertainty models may be needed.

---

## 20. Immediate Next Actions

Do only this first:

```text
1. Implement ValueMap._xy_to_value_map_px()
2. Implement ValueMap._positive_values_within_radius()
3. Run the consistency test on a few real frontier points
4. Confirm baseline median == helper median
```

Only after that:

```text
5. Implement MC samples
6. Implement MC summary
7. Add selector branch
8. Run one episode with debug prints
```

---

## 21. One-Sentence Summary

In each VLFM frontier decision, keep the baseline top-3 candidates, use MC perturbation of the same positive local value-map evidence used by the baseline median scorer to estimate each candidate's mean and standard deviation, and re-rank by \(J=\mu-\lambda\sigma\) before passing the result to the existing cyclic frontier selection logic.
