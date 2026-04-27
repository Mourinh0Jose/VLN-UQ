## Headline metrics (mean ± stderr)

| Setting | Method | N | Success Rate | Avg SPL | Avg SoftSPL | Avg dist-to-goal (m) | target_detected | stop_called |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pilot (30) | VLFM Original | 30 | 53.3% ± 9.3% | 0.321 ± 0.065 | 0.338 ± 0.063 | 6.73 ± 1.71 | 86.7% ± 6.3% | 86.7% ± 6.3% |
| Pilot (30) | Greedy Frontier | 30 | 53.3% ± 9.3% | 0.302 ± 0.059 | 0.356 ± 0.052 | 5.86 ± 1.64 | 90.0% ± 5.6% | 90.0% ± 5.6% |
| Extended (90) | VLFM Original | 90 | 51.1% ± 5.3% | 0.300 ± 0.037 | 0.368 ± 0.035 | 5.82 ± 0.88 | 76.7% ± 4.5% | 74.4% ± 4.6% |
| Extended (90) | Greedy Frontier | 90 | 50.0% ± 5.3% | 0.299 ± 0.037 | 0.366 ± 0.035 | 5.77 ± 0.83 | 77.8% ± 4.4% | 75.6% ± 4.6% |


## Paired comparison (same episodes, VLFM vs Greedy)

| Setting | N paired | Both succeed | Both fail | VLFM only | Greedy only | ΔSR (VLFM−Greedy) | ΔSPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pilot (30) | 30 | 14 | 12 | 2 | 2 | +0.0% | +0.019 |
| Extended (90) | 90 | 41 | 40 | 5 | 4 | +1.1% | +0.002 |


## Outcome breakdown (counts)

### Pilot (30)

| outcome | VLFM | Greedy |
|---|---:|---:|
| bad_stop_true_positive | 1 | 2 |
| did_not_fail | 16 | 16 |
| false_negative | 2 | 1 |
| false_positive | 9 | 9 |
| never_saw_target_did_not_travel_stairs_likely_infeasible | 2 | 2 |

### Extended (90)

| outcome | VLFM | Greedy |
|---|---:|---:|
| bad_stop_true_positive | 1 | 1 |
| did_not_fail | 46 | 45 |
| false_negative | 0 | 1 |
| false_positive | 24 | 24 |
| never_saw_target_did_not_travel_stairs_feasible | 7 | 8 |
| never_saw_target_did_not_travel_stairs_likely_infeasible | 12 | 11 |

