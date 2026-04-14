# Logging Audit

Trigger path: set `ZSOS_LOG_DIR=<dir>` at runtime. At the end of each episode,
`vlfm/utils/vlfm_trainer.py` calls `log_episode_stats(...)`. Then
`vlfm/utils/episode_stats_logger.py` detects `ZSOS_LOG_DIR`, removes the
`top_down_map` field from `infos`, adds `failure_cause`, and passes the result
to `vlfm/utils/log_saver.py`, which writes `<episode_id>_<scene>.json`.
The built-in `scripts/parse_jsons.py` script can summarize these JSON files
directly.

| Field | Already available? | Where (source) | Need custom logging? |
|---|---|---|---|
| `episode_id` | YES | `vlfm/utils/log_saver.py` writes it as a top-level JSON key | NO |
| `scene_id` | YES | `vlfm/utils/log_saver.py` writes it as a top-level JSON key | NO |
| `target_category` | YES | `infos["target_object"]` is preserved by `vlfm/utils/episode_stats_logger.py`; `scripts/parse_jsons.py` aggregates by `episode["target_object"]` | NO |
| `success` | YES | `infos["success"]` is preserved as-is; `scripts/parse_jsons.py` uses it directly | NO |
| `spl` / `soft_spl` | YES | `infos["spl"]` and `infos["soft_spl"]` are preserved as-is; `scripts/parse_jsons.py` uses them directly | NO |
| `failure_cause` | YES | `vlfm/utils/episode_stats_logger.py` computes and stores it | NO |
| `path_length` | MAYBE | The logger preserves non-array fields from `infos`, but this local audit did not find a guaranteed explicit VLFM write for `path_length`, and it did not appear in the first smoke output | Not needed yet |
| `collision_count` | MAYBE | Same issue as above; it would be preserved if Habitat exposed it in `infos`, but no explicit field evidence was found in this repo audit | Not needed yet |
| `runtime` (per ep) | NO | The default JSON logger does not store wall-clock runtime | NO, run-level wall time is enough for now |
| `num_vlm_calls` | NO | No counter exists in the current code | P1 only |
| `detector_triggered` | YES (indirectly) | JSON already contains `target_detected`, `stop_called`, and failure causes such as `false_positive`, `bad_stop_true_positive`, `timeout_true_positive`, and `false_negative` | NO |

Additional observations:

- `top_down_map` is removed by `vlfm/utils/episode_stats_logger.py`, so the
  default JSON output does not include a large map array.
- `scripts/parse_jsons.py` already covers the most important check-in summary
  tables: Success, SPL, SoftSPL, plus breakdowns by `target_object` and
  `failure_cause`.
- For this P0 comparison, the priority is that both methods emit the same JSON
  structure, not that the logger format grows more elaborate.

Conclusion:

The built-in `ZSOS_LOG_DIR` JSON logger plus `scripts/parse_jsons.py` is
already sufficient for the pilot comparisons in STEP 7-10. For the P0 check-in
phase, no custom CSV or logger extension is needed. If a later run shows that
`path_length` or collision fields are genuinely missing and necessary, they can
be added in a small follow-up patch.
