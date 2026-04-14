# Failure Case Analysis

## Case 1
- Episode ID: `1`
- Scene ID: `TEEsavR23oF`
- Target Category: `potted plant`
- Method: `Both`
- Outcome: `wrong_stop`
- Steps taken: `not logged in per-episode JSON`
- SPL: `0.00` for both methods

### Description
This episode failed in exactly the same way for both VLFM Original and the cheapest-frontier baseline. Both runs set `target_detected=True` and `stop_called=True`, but still ended more than 22m from the goal, which is a classic premature-stop pattern driven by a detector false positive rather than by frontier selection.

### Root Cause Category
- [ ] Semantic misranking (the VLM steers exploration toward the wrong region)
- [x] Detector false positive / missed detection
- [ ] Homogeneous environment (insufficient semantic cues)
- [ ] Exploration inefficiency (revisiting the same area repeatedly)
- [ ] Low-level navigation failure (collisions or getting stuck)

### UQ Relevance
This is the clearest sign that stop decisions should not trust a single detector trigger at face value. A future UQ-aware version should gate `stop` on detection confidence, repeated confirmation across nearby viewpoints, or consistency checks against navigation progress; otherwise, uncertainty in perception completely dominates the comparison and hides any benefit from better frontier ranking.

## Case 2
- Episode ID: `2`
- Scene ID: `wcojb4TFT35`
- Target Category: `tv`
- Method: `VLFM Original`
- Outcome: `wrong_stop`
- Steps taken: `not logged in per-episode JSON`
- SPL: `0.00` for VLFM Original, compared with `0.4346` for the baseline on the same episode

### Description
This is a paired flip episode: VLFM Original failed with `false_positive`, while the cheapest-frontier baseline succeeded on the exact same episode. The VLFM run called `stop` after a target detection and still finished about 11.01m from the goal, whereas the baseline finished at 0.04m, which suggests the semantic signal caused an early commitment to the wrong region instead of helping exploration.

### Root Cause Category
- [x] Semantic misranking (the VLM steers exploration toward the wrong region)
- [x] Detector false positive / missed detection
- [ ] Homogeneous environment (insufficient semantic cues)
- [ ] Exploration inefficiency (revisiting the same area repeatedly)
- [ ] Low-level navigation failure (collisions or getting stuck)

### UQ Relevance
This case is directly relevant to the next algorithmic step. When semantic frontier scores are uncertain, brittle, or only weakly better than alternatives, the policy should avoid converting that signal into an irreversible early stop. A useful UQ intervention would be to expose a semantic confidence or top1-top2 margin, then fall back to more conservative exploration when that uncertainty is high or when detector evidence is not persistent.
