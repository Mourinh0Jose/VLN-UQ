# MC Risk-Aware Analysis: results/mc_risk_aware_eval10_20260426

## Headline Metrics

| Method | N | Success | SPL | SoftSPL | Dist-to-goal | Target detected | Stop called |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic | 10 | 0.300 +/- 0.153 | 0.175 +/- 0.103 | 0.175 +/- 0.102 | 12.545 +/- 3.417 | 0.700 +/- 0.153 | 0.700 +/- 0.153 |
| mc_mean | 10 | 0.300 +/- 0.153 | 0.206 +/- 0.108 | 0.206 +/- 0.107 | 12.565 +/- 3.427 | 0.700 +/- 0.153 | 0.700 +/- 0.153 |
| mc_risk | 10 | 0.400 +/- 0.163 | 0.265 +/- 0.112 | 0.264 +/- 0.110 | 11.982 +/- 3.598 | 0.800 +/- 0.133 | 0.800 +/- 0.133 |


## Paired Against Semantic

| Method | N paired | Both succeed | Both fail | Method only | Semantic only | Delta Success | Delta SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| mc_mean | 10 | 3 | 7 | 0 | 0 | +0.000 | +0.030 |
| mc_risk | 10 | 3 | 6 | 1 | 0 | +0.100 | +0.089 |


## Outcome Counts

| Outcome | semantic | mc_mean | mc_risk |
|---|---:|---:|---:|
| did_not_fail | 3 | 3 | 4 |
| false_negative | 2 | 2 | 1 |
| false_positive | 4 | 4 | 4 |
| never_saw_target_did_not_travel_stairs_likely_infeasible | 1 | 1 | 1 |


## MC Episode Diagnostics

| Method | num_decisions | num_changed_decisions | num_mc_changed_top_decisions | avg_selected_mc_std | avg_all_topk_mc_std | cyclic_suppression_count |
|---|---:|---:|---:|---:|---:|---:|
| mc_mean | 143.3000 +/- 51.7088 | 20.5000 +/- 5.8542 | 4.6000 +/- 1.5434 | 0.0003 +/- 0.0001 | 0.0002 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mc_risk | 102.0000 +/- 34.8763 | 17.7000 +/- 4.5535 | 2.7000 +/- 0.8035 | 0.0003 +/- 0.0001 | 0.0003 +/- 0.0000 | 0.0000 +/- 0.0000 |


## UQ Diagnostics

| Method | Success avg selected std | Failure avg selected std | Changed decisions | Changed baseline std | Changed selected std | Selected lower-std rate |
|---|---:|---:|---:|---:|---:|---:|
| mc_mean | 0.000224 | 0.000344 | 205 | 0.000722 | 0.000245 | 0.662 |
| mc_risk | 0.000270 | 0.000356 | 177 | 0.000567 | 0.000185 | 0.600 |


## MC Std Histogram

| Method | Bin low | Bin high | Count |
|---|---:|---:|---:|
| mc_mean | 0.000000 | 0.002670 | 3202 |
| mc_mean | 0.002670 | 0.005341 | 47 |
| mc_mean | 0.005341 | 0.008011 | 7 |
| mc_mean | 0.008011 | 0.010681 | 1 |
| mc_mean | 0.010681 | 0.013352 | 1 |
| mc_risk | 0.000000 | 0.002670 | 2514 |
| mc_risk | 0.002670 | 0.005341 | 43 |
| mc_risk | 0.005341 | 0.008011 | 6 |
| mc_risk | 0.008011 | 0.010681 | 1 |
| mc_risk | 0.010681 | 0.013352 | 1 |


## Episode Uncertainty Bins

| Method | Bin | N | Avg selected std range | Success rate | Avg SPL |
|---|---:|---:|---|---:|---:|
| mc_mean | 1 | 2 | 0.000000 - 0.000065 | 0.500 | 0.416 |
| mc_mean | 2 | 2 | 0.000183 - 0.000185 | 0.000 | 0.000 |
| mc_mean | 3 | 2 | 0.000224 - 0.000331 | 0.500 | 0.369 |
| mc_mean | 4 | 2 | 0.000341 - 0.000480 | 0.500 | 0.244 |
| mc_mean | 5 | 2 | 0.000621 - 0.000650 | 0.000 | 0.000 |
| mc_risk | 1 | 2 | 0.000000 - 0.000065 | 0.500 | 0.416 |
| mc_risk | 2 | 2 | 0.000138 - 0.000183 | 0.000 | 0.000 |
| mc_risk | 3 | 2 | 0.000331 - 0.000341 | 1.000 | 0.613 |
| mc_risk | 4 | 2 | 0.000408 - 0.000480 | 0.500 | 0.293 |
| mc_risk | 5 | 2 | 0.000621 - 0.000649 | 0.000 | 0.000 |


## Success Flips

### mc_mean vs semantic

No success flips.

### mc_risk vs semantic

- episode=14 scene=TEEsavR23oF: mc_risk=1 (did_not_fail) semantic=0 (false_negative)

