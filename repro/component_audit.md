# VLFM Component Audit

| Component | Status | Evidence | File:Line |
|---|---|---|---|
| Occupancy map | ENABLED | Current Habitat policy constructs an `ObstacleMap` and updates it from depth observations each step; this is the active geometric map used for explored/navigable area tracking. | `vlfm/policy/base_objectnav_policy.py:86`, `vlfm/policy/habitat_policies.py:193` |
| Frontier extraction | ENABLED | Frontier extraction is implemented inside `ObstacleMap._get_frontiers()` using `detect_frontier_waypoints`, and the resulting frontiers are stored in `self.frontiers`. | `vlfm/mapping/obstacle_map.py:155`, `vlfm/mapping/obstacle_map.py:164` |
| Value map (VLM projection) | ENABLED | The current experiment config uses `HabitatITMPolicyV2`; that policy calls `_update_value_map()` every step, computes BLIP2-ITM cosine scores, and projects them into `ValueMap.update_map(...)`. | `config/experiments/vlfm_objectnav_hm3d.yaml:56`, `vlfm/policy/itm_policy.py:250`, `vlfm/policy/itm_policy.py:260`, `vlfm/policy/itm_policy.py:206` |
| Value map update rule | weighted_avg | `ValueMap._fuse_new_data()` supports ablations (`replace`, `equal_weighting`) plus two main modes: max-confidence replacement and weighted averaging. The current VLFM config sets `use_max_confidence: False`, so the weighted-average branch is active. | `vlfm/policy/base_objectnav_policy.py:381`, `vlfm/mapping/value_map.py:377`, `vlfm/mapping/value_map.py:401`, `vlfm/mapping/value_map.py:409` |
| Detector goal switch | ENABLED | Object detections are produced by YOLOv7 for COCO classes and GroundingDINO otherwise, then refined with MobileSAM masks and fused into the object map. Once the target object exists in the object map, policy mode switches from `explore` to `navigate` and calls `_pointnav(..., stop=True)`. | `vlfm/policy/base_objectnav_policy.py:64`, `vlfm/policy/base_objectnav_policy.py:133`, `vlfm/policy/base_objectnav_policy.py:171`, `vlfm/policy/base_objectnav_policy.py:221`, `vlfm/policy/base_objectnav_policy.py:311` |
| Low-level navigation | ENABLED | The object-nav policy instantiates `WrappedPointNavResNetPolicy`, which wraps a Habitat PointNav ResNet policy and is used as the low-level action generator. | `vlfm/policy/base_objectnav_policy.py:70`, `vlfm/policy/utils/pointnav_policy.py:51` |

## Value Map Update Rule

Current setting: `weighted_avg`

Evidence:
- `VLFMConfig.use_max_confidence` defaults to `False`, so the current
  original VLFM config does not use the max-confidence overwrite branch.
  See `vlfm/policy/base_objectnav_policy.py:381`
- In `ValueMap._fuse_new_data()`:
  - `replace` is an ablation branch. See `vlfm/mapping/value_map.py:377`
  - `equal_weighting` is a non-confidence averaging ablation.
    See `vlfm/mapping/value_map.py:386`
  - `self._use_max_confidence=True` enables max-confidence replacement.
    See `vlfm/mapping/value_map.py:401`
  - Otherwise the map uses confidence-weighted averaging between old and new
    values. See `vlfm/mapping/value_map.py:409`

## Notes

- The current official code path is not "GroundingDINO + MobileSAM only".
  It uses a combined **YOLOv7 + GroundingDINO + MobileSAM** detection stack.
  YOLOv7 primarily handles COCO classes, while GroundingDINO covers open-
  vocabulary or non-COCO categories. See
  `vlfm/policy/base_objectnav_policy.py:223`
- The active execution path is aligned with the paper-level "original VLFM"
  components: obstacle map, frontier extraction, value map, detector-triggered
  explore-to-navigate switching, and low-level PointNav are all enabled.
