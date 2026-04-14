# Pilot Protocol

- split: `val_mini`
- selection mode: `test_episode_count=30` (full split, deterministic full-coverage run)
- n_episodes: `30`
- both methods must use identical Hydra overrides except `habitat_baselines.rl.policy.frontier_selector`

Note:

- In this local HM3D `val_mini` shard layout, the top-level split JSON is empty and the actual episodes live in `data/datasets/objectnav/hm3d/v1/val_mini/content/*.json.gz`.
- The extracted `pilot_episode_ids.txt` therefore comes from concatenating the two content shards in sorted filename order.
- Numeric `episode_id` values are not globally unique across scenes in this split, so later alignment checks should prefer `scene_id + episode_id` when comparing per-episode JSON outputs.
