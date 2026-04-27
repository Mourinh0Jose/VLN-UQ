# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import zlib
from collections import deque
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


def _stable_int_from_episode_id(episode_id: object) -> int:
    """Convert an episode id into a stable uint32 seed component."""
    return zlib.crc32(str(episode_id).encode("utf-8"))


def make_mc_sample_seeds(
    master_seed: int,
    episode_id: object,
    decision_counter: int,
    n_samples: int,
) -> List[int]:
    """Generate reproducible MC seeds for one decision step."""
    seed_seq = np.random.SeedSequence(
        [
            int(master_seed),
            int(_stable_int_from_episode_id(episode_id)),
            int(decision_counter),
        ]
    )
    return seed_seq.generate_state(n_samples, dtype=np.uint32).astype(int).tolist()


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        self._mc_decision_log: List[Dict[str, Any]] = []

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        self._mc_decision_log = []
        self.reset_mc_for_episode(getattr(self, "_mc_episode_id", "unknown"))

    def reset_mc_for_episode(self, episode_id: object) -> None:
        """Reset MC state for a new episode."""
        self._mc_episode_id = episode_id
        self._mc_decision_counter = 0

    def set_mc_episode_context(self, scene_id: object, episode_id: object) -> None:
        """Set the active Habitat episode context for reproducible MC seeds."""
        episode_key = f"{scene_id}:{episode_id}"
        if getattr(self, "_mc_episode_id", None) != episode_key:
            self.reset_mc_for_episode(episode_key)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        score_text = self._format_frontier_score(best_value)
        os.environ["DEBUG_INFO"] = score_text
        print(score_text)
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        robot_xy = self._observations_cache["robot_xy"]
        selector = self._get_frontier_selector()
        if selector == "semantic":
            # The points and values will be sorted in descending order
            sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        elif selector == "mc_risk_aware":
            sorted_pts, sorted_values = self._sort_frontiers_mc_risk_aware(observations, frontiers)
        else:
            sorted_pts, sorted_values = self._sort_frontiers_geometric(frontiers, robot_xy, selector)
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])
        cyclic_suppressed = False

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    cyclic_suppressed = True
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" {self._format_frontier_score(best_value)}"

        if selector == "mc_risk_aware":
            self._record_mc_decision(
                frontiers=frontiers,
                sorted_pts=sorted_pts,
                sorted_values=sorted_values,
                selected_idx=best_frontier_idx,
                top_two_values=top_two_values,
                cyclic_suppressed=cyclic_suppressed,
            )

        return best_frontier, best_value

    def _get_frontier_selector(self) -> str:
        selector = getattr(self, "_frontier_selector", "semantic")
        if selector not in {"semantic", "nearest", "cheapest", "mc_risk_aware"}:
            print(f"Unknown frontier selector '{selector}', falling back to semantic.")
            return "semantic"
        return selector

    def _format_frontier_score(self, score: float) -> str:
        selector = self._get_frontier_selector()
        if selector == "semantic":
            return f"Best value: {score*100:.2f}%"
        if selector == "mc_risk_aware":
            return f"MC risk-aware frontier score: {score:.4f}"
        return f"{selector.title()} frontier cost: {-score:.2f}m"

    def _sort_frontiers_geometric(
        self,
        frontiers: np.ndarray,
        robot_xy: np.ndarray,
        selector: str,
    ) -> Tuple[np.ndarray, List[float]]:
        if selector == "nearest":
            costs = np.linalg.norm(frontiers - robot_xy, axis=1)
        elif selector == "cheapest":
            costs = self._compute_frontier_path_costs(frontiers, robot_xy)
            if costs is None or not np.isfinite(costs).any():
                print("Cheapest frontier fallback: could not compute grid costs, using nearest frontier.")
                costs = np.linalg.norm(frontiers - robot_xy, axis=1)
        else:
            raise ValueError(f"Unsupported frontier selector: {selector}")

        sorted_inds = np.argsort(costs)
        sorted_frontiers = np.array([frontiers[i] for i in sorted_inds])
        sorted_scores = (-costs[sorted_inds]).tolist()
        return sorted_frontiers, sorted_scores

    def _compute_frontier_path_costs(self, frontiers: np.ndarray, robot_xy: np.ndarray) -> Union[np.ndarray, None]:
        if not hasattr(self, "_obstacle_map"):
            return None

        explored_area = cv2.dilate(
            self._obstacle_map.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        ).astype(bool)
        traversable = np.logical_and(
            self._obstacle_map._navigable_map,
            explored_area,
        ).copy()
        if traversable.size == 0:
            return None

        height, width = traversable.shape
        robot_px = self._obstacle_map._xy_to_px(robot_xy.reshape(1, 2))[0]
        frontier_px = self._obstacle_map._xy_to_px(frontiers)

        robot_col = int(np.clip(robot_px[0], 0, width - 1))
        robot_row = int(np.clip(robot_px[1], 0, height - 1))
        frontier_cols = np.clip(frontier_px[:, 0], 0, width - 1).astype(int)
        frontier_rows = np.clip(frontier_px[:, 1], 0, height - 1).astype(int)

        traversable[robot_row, robot_col] = True
        traversable[frontier_rows, frontier_cols] = True

        frontier_cells = {}
        remaining = set()
        for idx, (row, col) in enumerate(zip(frontier_rows, frontier_cols)):
            key = (int(row), int(col))
            frontier_cells.setdefault(key, []).append(idx)
            remaining.add(key)
        remaining.discard((robot_row, robot_col))

        distances = np.full((height, width), -1, dtype=np.int32)
        distances[robot_row, robot_col] = 0
        queue = deque([(robot_row, robot_col)])

        while queue and remaining:
            row, col = queue.popleft()
            next_distance = distances[row, col] + 1
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_row = row + d_row
                new_col = col + d_col
                if new_row < 0 or new_row >= height or new_col < 0 or new_col >= width:
                    continue
                if not traversable[new_row, new_col] or distances[new_row, new_col] != -1:
                    continue
                distances[new_row, new_col] = next_distance
                queue.append((new_row, new_col))
                remaining.discard((new_row, new_col))

        costs = np.full(len(frontiers), np.inf, dtype=np.float32)
        for (row, col), frontier_indices in frontier_cells.items():
            if distances[row, col] == -1:
                continue
            cost_meters = distances[row, col] / float(self._obstacle_map.pixels_per_meter)
            costs[frontier_indices] = cost_meters

        return costs

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if self._get_frontier_selector() == "mc_risk_aware":
            policy_info["mc_decision_log"] = getattr(self, "_mc_decision_log", [])
            policy_info["mc_episode_summary"] = self._get_mc_episode_summary()

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError

    def _sort_frontiers_mc_risk_aware(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError

    @staticmethod
    def _point_to_log(point: Union[np.ndarray, List[float], Tuple[float, ...]]) -> List[float]:
        return [float(x) for x in np.asarray(point, dtype=np.float32).reshape(-1).tolist()]

    @staticmethod
    def _points_match(left: Union[np.ndarray, List[float]], right: Union[np.ndarray, List[float]]) -> bool:
        return bool(np.allclose(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))

    def _record_mc_decision(
        self,
        frontiers: np.ndarray,
        sorted_pts: np.ndarray,
        sorted_values: List[float],
        selected_idx: int,
        top_two_values: Tuple[float, ...],
        cyclic_suppressed: bool,
    ) -> None:
        if len(sorted_pts) == 0:
            return

        selected_frontier = sorted_pts[selected_idx]
        baseline_top = getattr(self, "_last_mc_baseline_top_frontier", sorted_pts[0])
        mc_top = getattr(self, "_last_mc_top_frontier", sorted_pts[0])
        records = []

        for record in getattr(self, "_last_mc_frontier_records", []):
            point = record["point"]
            records.append(
                {
                    "frontier_rank_baseline": int(record["baseline_rank"]),
                    "frontier_xy": self._point_to_log(point),
                    "baseline_value": float(record["baseline_value"]),
                    "mc_samples": [float(sample) for sample in record["mc_samples"]],
                    "mc_mean": float(record["mc_mean"]),
                    "mc_std": float(record["mc_std"]),
                    "risk_score": float(record["risk_score"]),
                    "selected_by_baseline": int(record["baseline_rank"]) == 0,
                    "selected_by_mc_risk": self._points_match(point, mc_top),
                    "selected_final": self._points_match(point, selected_frontier),
                }
            )

        selected_record = next((record for record in records if record["selected_final"]), None)
        decision_record = {
            "decision_step": int(getattr(self, "_last_mc_decision_step", len(self._mc_decision_log))),
            "selector": "mc_risk_aware",
            "num_frontiers": int(len(frontiers)),
            "mc_top_k": int(self._mc_top_k),
            "mc_n_samples": int(self._mc_n_samples),
            "mc_keep_prob": float(self._mc_keep_prob),
            "mc_noise_rel": float(self._mc_noise_rel),
            "mc_lambda": float(self._mc_lambda),
            "sample_seeds": [int(seed) for seed in getattr(self, "_last_mc_sample_seeds", [])],
            "baseline_top_frontier_xy": self._point_to_log(baseline_top),
            "mc_top_frontier_xy": self._point_to_log(mc_top),
            "final_selected_frontier_xy": self._point_to_log(selected_frontier),
            "mc_changed_top_from_baseline": not self._points_match(mc_top, baseline_top),
            "changed_from_baseline": not self._points_match(selected_frontier, baseline_top),
            "cyclic_suppressed": bool(cyclic_suppressed),
            "top_two_values": [float(value) for value in top_two_values],
            "selected_value": float(sorted_values[selected_idx]),
            "selected_mc_std": None if selected_record is None else float(selected_record["mc_std"]),
            "topk_frontier_records": records,
        }
        self._mc_decision_log.append(decision_record)

    def _get_mc_episode_summary(self) -> Dict[str, Any]:
        decision_log = getattr(self, "_mc_decision_log", [])
        selected_stds = [
            record["selected_mc_std"] for record in decision_log if record.get("selected_mc_std") is not None
        ]
        all_topk_stds = [
            frontier_record["mc_std"]
            for decision_record in decision_log
            for frontier_record in decision_record.get("topk_frontier_records", [])
        ]
        return {
            "method": "mc_risk_aware",
            "episode_seed_id": str(getattr(self, "_mc_episode_id", "unknown")),
            "num_decisions": int(len(decision_log)),
            "num_changed_decisions": int(sum(record["changed_from_baseline"] for record in decision_log)),
            "num_mc_changed_top_decisions": int(
                sum(record["mc_changed_top_from_baseline"] for record in decision_log)
            ),
            "avg_selected_mc_std": float(np.mean(selected_stds)) if len(selected_stds) > 0 else 0.0,
            "avg_all_topk_mc_std": float(np.mean(all_topk_stds)) if len(all_topk_stds) > 0 else 0.0,
            "cyclic_suppression_count": int(sum(record["cyclic_suppressed"] for record in decision_log)),
        }


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        if self._get_frontier_selector() in {"semantic", "mc_risk_aware"}:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values

    def _sort_frontiers_mc_risk_aware(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        if len(frontiers) == 0:
            return frontiers, []

        reduce_fn = None
        if self._value_map._value_channels > 1:
            reduce_fn = lambda values: [value[self._mc_channel_idx] for value in values]
        baseline_pts, baseline_values = self._value_map.sort_waypoints(
            frontiers,
            self._mc_radius_m,
            reduce_fn=reduce_fn,
        )

        def to_scalar(value: Union[float, Tuple[float, ...]]) -> float:
            return float(value[0]) if isinstance(value, tuple) else float(value)

        baseline_values_scalar = [to_scalar(value) for value in baseline_values]
        top_k = min(self._mc_top_k, len(baseline_pts))
        top_pts = baseline_pts[:top_k]
        remaining_pts = baseline_pts[top_k:]

        decision_counter = getattr(self, "_mc_decision_counter", 0)
        sample_seeds = make_mc_sample_seeds(
            master_seed=self._mc_seed_master,
            episode_id=getattr(self, "_mc_episode_id", "unknown"),
            decision_counter=decision_counter,
            n_samples=self._mc_n_samples,
        )
        self._mc_decision_counter = decision_counter + 1

        top_records = []
        for baseline_rank, point in enumerate(top_pts):
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
            top_records.append(
                {
                    "point": point,
                    "baseline_rank": baseline_rank,
                    "baseline_value": baseline_values_scalar[baseline_rank],
                    "mc_samples": summary["samples"],
                    "mc_mean": summary["mean"],
                    "mc_std": summary["std"],
                    "risk_score": summary["risk_score"],
                }
            )

        top_records = sorted(top_records, key=lambda record: record["risk_score"], reverse=True)
        reranked_top_pts = [record["point"] for record in top_records]
        reranked_top_values = [float(record["risk_score"]) for record in top_records]

        if len(remaining_pts) > 0:
            min_top_risk = min(reranked_top_values) if len(reranked_top_values) > 0 else 0.0
            remaining_values = [float(min_top_risk - 0.01 - 0.001 * i) for i in range(len(remaining_pts))]
        else:
            remaining_values = []

        sorted_pts = np.array(reranked_top_pts + list(remaining_pts))
        sorted_values = reranked_top_values + remaining_values

        self._last_mc_baseline_top_frontier = baseline_pts[0]
        self._last_mc_top_frontier = sorted_pts[0]
        self._last_mc_frontier_records = top_records
        self._last_mc_sample_seeds = sample_seeds
        self._last_mc_decision_step = decision_counter

        if getattr(self, "_mc_debug", False):
            print("[MC risk-aware] seeds:", sample_seeds)
            for rank, record in enumerate(top_records):
                print(
                    f"  rank={rank}",
                    f"baseline_rank={record['baseline_rank']}",
                    f"baseline={record['baseline_value']:.4f}",
                    f"mu={record['mc_mean']:.4f}",
                    f"std={record['mc_std']:.4f}",
                    f"risk={record['risk_score']:.4f}",
                    f"samples={record['mc_samples']}",
                )

        return sorted_pts, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
