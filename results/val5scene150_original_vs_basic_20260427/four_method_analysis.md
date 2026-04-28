# Four-Method Pilot Analysis on `val_5scene_150`

## Setup

This pilot compares four frontier-selection variants on the same fixed split:

- Split: `val_5scene_150`
- Episodes: 150 total, 5 scenes x 30 episodes
- Alignment: all compared runs match exactly by `scene_id + episode_id`
- Output root: `results/val5scene150_original_vs_basic_20260427/`

Methods:

| Method | Selector | Key Setting | Description |
|---|---|---:|---|
| Original VLFM | `semantic` | n/a | Original semantic frontier selection |
| Basic / Cheapest Frontier | `cheapest` | n/a | Nearest / cheapest frontier baseline |
| MC Mean | `mc_risk_aware` | `mc_lambda=0.0` | Rerank top frontiers by MC mean only |
| MC Risk | `mc_risk_aware` | `mc_lambda=1.0` | Rerank by `mean - std` risk-aware score |

## Main Results

| Method | Success | Successful Episodes | SPL | SoftSPL |
|---|---:|---:|---:|---:|
| Original VLFM | **52.00%** | **78 / 150** | 29.84% | 37.04% |
| Basic / Cheapest Frontier | 49.33% | 74 / 150 | **29.96%** | **37.78%** |
| MC Mean | 50.00% | 75 / 150 | 29.01% | 35.84% |
| MC Risk | 48.67% | 73 / 150 | 28.46% | 35.74% |

## High-Level Takeaways

Original VLFM has the best success rate on this split, with 78 successful episodes. This suggests that the original semantic frontier scoring is still the strongest default among the four tested variants.

Basic / Cheapest Frontier is surprisingly competitive. It has lower success than Original VLFM, but the highest SPL and SoftSPL. This means that when it works, it can be efficient, and on this 150-episode pilot the simple geometric baseline is not weak enough to be dismissed.

MC Mean is close to the Basic baseline. It improves success by one episode over Basic / Cheapest Frontier, but its SPL and SoftSPL are lower. This suggests that MC mean reranking changes some decisions in useful ways, but does not yet produce a clear navigation-efficiency benefit.

MC Risk with `lambda=1.0` is the weakest of the four on aggregate metrics. However, the drop is small: it is two successful episodes behind MC Mean and five behind Original VLFM. This is a pilot-scale result, not strong evidence that risk-aware scoring is generally bad.

## Pairwise Interpretation

Original VLFM vs Basic / Cheapest Frontier:

- Original VLFM wins by 4 successful episodes: 78 vs 74.
- Basic has slightly higher SPL and SoftSPL despite lower success.
- Paired success flips: 12 episodes.
- Interpretation: semantic frontier scoring helps success, but the basic frontier policy can still be efficient on the episodes where it succeeds.

Original VLFM vs MC Mean:

- Original VLFM wins by 3 successful episodes: 78 vs 75.
- Original also has higher SPL and SoftSPL.
- Paired success flips: 11 episodes.
- Interpretation: MC mean changes a small number of decisions, but does not beat the original semantic ranking on this split.

MC Mean vs MC Risk:

- MC Mean wins by 2 successful episodes: 75 vs 73.
- Paired success flips: 4 episodes.
- MC Mean wins 3 of those flipped episodes, while MC Risk wins 1.
- Interpretation: the difference between `lambda=0.0` and `lambda=1.0` is small but negative here. The risk penalty appears too conservative for this pilot.

## Failure-Cause Summary

| Method | False Positive | Never Saw Target, Likely Infeasible | Never Saw Target, Feasible | False Negative | Bad Stop True Positive |
|---|---:|---:|---:|---:|---:|
| Original VLFM | 40 | 19 | 8 | 1 | 3 |
| Basic / Cheapest Frontier | 39 | 19 | 11 | 1 | 4 |
| MC Mean | 41 | 17 | 9 | 4 | 3 |
| MC Risk | 42 | 17 | 10 | 4 | 3 |

The dominant failure mode is `false_positive` for all four methods. This is important because frontier selection does not directly fix false-positive stopping behavior. The MC variants do not reduce false positives; in fact, MC Mean has 41 and MC Risk has 42, compared with 40 for Original VLFM.

The MC variants slightly reduce `never_saw_target_did_not_travel_stairs_likely_infeasible` compared with Original VLFM and Basic, from 19 to 17. But this small gain is offset by more false positives and false negatives.

## Category-Level Observations

| Category | Original Failure Rate | Basic Failure Rate | MC Mean Failure Rate | MC Risk Failure Rate |
|---|---:|---:|---:|---:|
| tv | 80.95% | 85.71% | **76.19%** | 80.95% |
| couch | 60.00% | 60.00% | 60.00% | 60.00% |
| bed | **38.46%** | **38.46%** | 50.00% | 50.00% |
| toilet | 58.33% | 58.33% | **54.17%** | **54.17%** |
| potted plant | 85.71% | 100.00% | **78.57%** | 85.71% |
| chair | **10.00%** | 12.50% | 17.50% | 17.50% |

MC Mean appears helpful for `tv`, `toilet`, and `potted plant`, but hurts `bed` and `chair` on this split. MC Risk loses the `tv` and `potted plant` gains that MC Mean had, while keeping the worse `bed` and `chair` failure rates.

Because the category counts are small, these should be treated as diagnostic signals rather than final claims.

## Why `MC Risk` Can Be Worse Than `MC Mean`

The risk-aware score is:

```text
risk_score = mean - lambda * std
```

For MC Mean, `lambda=0.0`, so the score is just:

```text
risk_score = mean
```

For MC Risk in this run, `lambda=1.0`, so the score becomes:

```text
risk_score = mean - std
```

This penalizes frontiers with higher MC uncertainty. That can be useful if uncertainty mostly means noisy or unreliable semantic evidence. But in ObjectNav, uncertainty can also mean an informative frontier where the target may be partially visible or where the map evidence is still sparse. Penalizing those frontiers too strongly can make the agent more conservative and cause it to choose a safer-looking but less informative direction.

This matches the observed result: MC Risk is only slightly worse than MC Mean, but the risk penalty does not improve the dominant failure modes and loses two net successful episodes.

## Current Conclusion

The best current method on this pilot is Original VLFM by success rate. Basic / Cheapest Frontier is a strong simple baseline because it has the best SPL and SoftSPL. MC Mean is competitive but not better than Original VLFM. MC Risk with `lambda=1.0` does not help and is probably too conservative.

The main scientific takeaway should not be "MC risk is bad." A better conclusion is:

> On this 150-episode pilot, adding MC uncertainty changes frontier decisions but does not yet improve ObjectNav success. A strong risk penalty (`lambda=1.0`) slightly hurts performance relative to MC Mean, suggesting that the risk coefficient needs tuning.

## Recommended Next Experiments

Run a small lambda sweep on the same fixed 150 episodes:

| Variant | `mc_lambda` | Purpose |
|---|---:|---|
| MC Mean | 0.0 | Already completed; no risk penalty |
| MC Risk Mild | 0.25 | Test light uncertainty penalty |
| MC Risk Medium | 0.5 | Test moderate uncertainty penalty |
| MC Risk Strong | 1.0 | Already completed; likely too conservative |

Use the same fixed controls:

- `val_5scene_150`
- same episode files
- same `mc_top_k`
- same `mc_n_samples`
- same `mc_keep_prob`
- same `mc_noise_rel`
- same `mc_seed_master`

The next two most useful runs are `mc_lambda=0.25` and `mc_lambda=0.5`. If either improves over MC Mean, the story becomes a calibrated-risk result: light uncertainty penalty helps, but strong uncertainty penalty overcorrects.

If neither improves over MC Mean, the result is still useful: the MC uncertainty signal is diagnostic, but the current `mean - lambda * std` reranking rule is not enough to improve navigation performance.

