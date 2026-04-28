# Cluster 30-Episode Pilot Analysis

## Setup

This pilot evaluates four frontier-selection variants on the same 30 HM3D ObjectNav `val_mini` episodes.

| Method | Selector | Description |
|---|---|---|
| Semantic VLFM | `semantic` | Original VLFM semantic frontier ranking using VLM value estimates. |
| Greedy Frontier | `cheapest` | Geometry-only baseline that prefers the lowest path-cost frontier. |
| MC Mean | `mc_risk_aware`, `mc_lambda=0.0` | MC reranking of top semantic frontiers using the MC mean only. |
| MC Risk | `mc_risk_aware`, `mc_lambda=1.0` | MC reranking using `mean - lambda * std`, penalizing uncertain frontiers. |

The runs were executed on the cluster under:

```text
results/cluster_pilot_semantic
results/cluster_pilot_cheapest
results/cluster_pilot_mc_mean
results/cluster_pilot_mc_risk
```

## Headline Results

| Method | Episodes | Success | SPL | SoftSPL | Dist-to-goal | Target detected | Stop called |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic VLFM | 30 | 53.33% | 32.53% | 34.24% | 6.725 | 86.67% | 86.67% |
| Greedy Frontier | 30 | **56.67%** | 31.62% | **36.26%** | **5.850** | **93.33%** | **93.33%** |
| MC Risk | 30 | 50.00% | 31.80% | 34.11% | 7.004 | 86.67% | 86.67% |
| MC Mean | 30 | 46.67% | 29.84% | 32.18% | 7.199 | 83.33% | 83.33% |

The geometry-only greedy baseline is the strongest method by success rate and SoftSPL on this pilot split. Semantic VLFM has the best SPL, but only narrowly. The MC variants do not improve over the original semantic policy; however, MC Risk clearly improves over MC Mean, suggesting that the uncertainty penalty is directionally useful even though the current MC signal is not strong enough to improve the aggregate navigation metrics.

## Paired Comparisons

Because all methods ran on the same 30 episodes, paired comparison is more informative than averages alone.

| Pair | Delta Success | Delta SPL | Delta SoftSPL | Success Flips | Interpretation |
|---|---:|---:|---:|---:|---|
| Semantic - Greedy | -0.033 | +0.009 | -0.020 | 3 | Greedy wins slightly on success and SoftSPL; semantic is slightly better on SPL. |
| Semantic - MC Risk | +0.033 | +0.007 | +0.001 | 3 | Semantic remains slightly better than MC Risk. |
| MC Risk - MC Mean | +0.033 | +0.020 | +0.019 | 1 | Risk penalty improves over pure MC mean. |

Notable success flips:

| Pair | Episode | Scene | Better method | Notes |
|---|---|---|---|---|
| Semantic vs Greedy | 10 | `TEEsavR23oF` | Greedy | Semantic failed as false negative; Greedy succeeded. |
| Semantic vs Greedy | 10 | `wcojb4TFT35` | Semantic | Greedy failed as false positive; Semantic succeeded. |
| Semantic vs Greedy | 2 | `wcojb4TFT35` | Greedy | Semantic failed as false positive; Greedy succeeded. |
| Semantic vs MC Risk | 14 | `TEEsavR23oF` | MC Risk | Semantic failed as false negative; MC Risk succeeded. |
| Semantic vs MC Risk | 10 | `wcojb4TFT35` | Semantic | MC Risk failed as false positive; Semantic succeeded. |
| Semantic vs MC Risk | 8 | `wcojb4TFT35` | Semantic | MC Risk failed as false negative; Semantic succeeded. |
| MC Risk vs MC Mean | 14 | `TEEsavR23oF` | MC Risk | MC Mean failed as false negative; MC Risk succeeded. |

## Failure Modes

| Failure Cause | Semantic | Greedy | MC Risk | MC Mean |
|---|---:|---:|---:|---:|
| Succeeded | 16 | **17** | 15 | 14 |
| False positive | 9 | 9 | 10 | 10 |
| False negative | 2 | 0 | 2 | 3 |
| Bad stop true positive | 1 | 2 | 1 | 1 |
| Never saw target, likely infeasible | 2 | 2 | 2 | 2 |

False positives are the dominant failure mode across all methods. MC methods do not reduce false positives; in fact, both MC variants have 10 false positives versus 9 for semantic and greedy. This suggests that the current uncertainty reranking does not address the main failure source in the pilot.

Category-level failures:

| Category | Semantic | Greedy | MC Risk | MC Mean |
|---|---:|---:|---:|---:|
| Potted plant | 100.00% (5/5) | 100.00% (5/5) | 100.00% (5/5) | 100.00% (5/5) |
| Toilet | 60.00% (3/5) | 60.00% (3/5) | 60.00% (3/5) | 60.00% (3/5) |
| Bed | 42.86% (3/7) | **28.57% (2/7)** | **28.57% (2/7)** | 42.86% (3/7) |
| TV | 66.67% (2/3) | 66.67% (2/3) | 100.00% (3/3) | 100.00% (3/3) |
| Chair | **14.29% (1/7)** | **14.29% (1/7)** | 28.57% (2/7) | 28.57% (2/7) |
| Couch | **0.00% (0/3)** | **0.00% (0/3)** | **0.00% (0/3)** | **0.00% (0/3)** |

Potted plant is consistently hard for every method. TV performance gets worse for the MC variants, while bed improves for Greedy and MC Risk relative to Semantic and MC Mean.

## MC Diagnostics

| Method | Decisions / episode | Changed decisions / episode | MC changed top decisions / episode | Avg selected MC std | Avg all top-k MC std |
|---|---:|---:|---:|---:|---:|
| MC Mean | 110.30 | 23.17 | 3.23 | 0.0002 | 0.0002 |
| MC Risk | 96.53 | 22.20 | 2.80 | 0.0002 | 0.0002 |

The MC mechanism changes some frontier choices, but the top-choice change rate is modest. MC Risk changes the top frontier about 2.8 times per episode on average.

Uncertainty diagnostics:

| Method | Success avg selected std | Failure avg selected std | Changed decisions | Changed baseline std | Changed selected std | Selected lower-std rate |
|---|---:|---:|---:|---:|---:|---:|
| MC Mean | 0.000213 | 0.000244 | 695 | 0.000439 | 0.000220 | 0.604 |
| MC Risk | 0.000226 | 0.000245 | 666 | 0.000397 | 0.000206 | 0.593 |

Failures have only slightly higher selected uncertainty than successes. The gap is very small, so the current MC uncertainty estimate is weakly discriminative. MC Risk does select lower-std frontiers in roughly 59% of changed decisions, but this is not enough to improve over the original semantic policy.

## Interpretation

The key result is negative but useful: MC risk-aware reranking does not outperform the original semantic VLFM policy on the 30-episode `val_mini` pilot. The current ranking signal appears too small to change outcomes consistently. At the same time, MC Risk is better than MC Mean, which supports the idea that uncertainty penalty is preferable to simply averaging noisy samples.

The Greedy Frontier baseline remains very competitive. It achieves the best success rate, best SoftSPL, lowest distance-to-goal, and highest target-detected/stop-called rates. This suggests that on this small split, frontier geometry is already a strong cue, and semantic reranking does not dominate simple navigation-cost minimization.

For the paper/report, the most defensible claim is:

> On a 30-episode HM3D `val_mini` pilot, uncertainty-aware MC frontier reranking did not improve aggregate ObjectNav metrics over the original semantic VLFM policy. However, the risk-aware variant outperformed the MC-mean ablation, indicating that penalizing uncertainty is directionally helpful. The main remaining failure mode is false-positive stopping, which the current MC uncertainty estimate does not reduce.

## Limitations

- The pilot uses only 30 episodes from `val_mini`, so differences of one episode correspond to 3.33 percentage points.
- The split contains two scenes, so category and scene effects may dominate.
- The MC uncertainty values are very small and close between successes and failures.
- The analysis does not yet include visual qualitative evidence from the flip episodes.

## Recommended Next Steps

1. Use this 30-episode pilot as the check-in result.
2. For a stronger final claim, later run a larger split such as 90 episodes.
3. Inspect the flip episodes qualitatively, especially:
   - `TEEsavR23oF`, episode `14`
   - `wcojb4TFT35`, episode `10`
   - `wcojb4TFT35`, episode `8`
   - `wcojb4TFT35`, episode `2`
4. Consider revising the MC uncertainty design so that it targets false-positive stopping, since false positives dominate the failure profile.

## Extra Cloud Data Needed

The summary above is complete for a numeric 30-episode analysis. For deeper qualitative analysis, pull or inspect these cloud files:

```text
results/cluster_pilot_analysis/analysis.md
results/cluster_pilot_semantic/{10_TEEsavR23oF,10_wcojb4TFT35,8_wcojb4TFT35,14_TEEsavR23oF,2_wcojb4TFT35}.json
results/cluster_pilot_cheapest/{10_TEEsavR23oF,10_wcojb4TFT35,2_wcojb4TFT35}.json
results/cluster_pilot_mc_risk/{14_TEEsavR23oF,10_wcojb4TFT35,8_wcojb4TFT35}.json
results/cluster_pilot_mc_mean/14_TEEsavR23oF.json
```

Also verify that each result directory contains 30 JSON files:

```bash
find results/cluster_pilot_semantic -name "*.json" | wc -l
find results/cluster_pilot_cheapest -name "*.json" | wc -l
find results/cluster_pilot_mc_risk -name "*.json" | wc -l
find results/cluster_pilot_mc_mean -name "*.json" | wc -l
```
