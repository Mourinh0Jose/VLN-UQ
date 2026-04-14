# Baseline Implementation Plan

## Frontier selector location
- Core entry: `vlfm/policy/itm_policy.py` (`BaseITMPolicy._get_best_frontier`)
- Active scoring hook for the current experiment: `vlfm/policy/itm_policy.py` (`ITMPolicyV2._sort_frontiers_by_value`, used by `HabitatITMPolicyV2`)

## Current selection logic (semantic)
- Semantic mode still follows the original path: `_get_best_frontier()` calls `_sort_frontiers_by_value(...)`, gets `(sorted_pts, sorted_values)`, then applies:
  - stick-to-last frontier
  - `_acyclic_enforcer` loop suppression
  - closest-frontier fallback if every candidate is cyclic

## Baseline modification plan
- Add config switch `frontier_selector` with three modes:
  - `semantic`: keep original VLFM behavior
  - `nearest`: rank frontiers by Euclidean distance to `robot_xy`
  - `cheapest`: rank frontiers by grid BFS cost on the obstacle map's navigable mask
- Keep all non-selector behavior unchanged:
  - occupancy / obstacle map
  - frontier extraction
  - detector switch (YOLOv7 + GroundingDINO + MobileSAM)
  - low-level PointNav navigation
- Preserve the existing stick-to-last and anti-cyclic logic after ranking, so the baseline swaps the ranking signal, not the whole exploration loop.

## Files touched
1. `vlfm/policy/base_objectnav_policy.py`
2. `vlfm/policy/itm_policy.py`

## Cheapest reachable frontier logic
- Convert `robot_xy` and frontier points to obstacle-map pixels.
- Build a traversable grid from:
  - `ObstacleMap._navigable_map`
  - dilated `ObstacleMap.explored_area`
- Run 4-neighbor BFS from the robot cell.
- Use BFS distance (converted back to meters by `pixels_per_meter`) as frontier cost.
- Sort ascending by cost, then feed the sorted frontiers back into the original `_get_best_frontier()` post-processing.

## Fallback
- If `cheapest` cannot compute any finite grid costs, fall back to `nearest` and print a warning.
- This keeps the 3-file budget intact and prevents the check-in from stalling on planner internals.
