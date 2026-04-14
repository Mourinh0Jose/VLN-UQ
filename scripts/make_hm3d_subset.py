import argparse
import copy
import gzip
import json
import os
import random
from typing import Any, Dict, List


def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def dump_json_gz(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split", default="val")
    parser.add_argument("--target-split", required=True)
    parser.add_argument("--episodes-per-scene", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--sample-mode", choices=["random", "head"], default="random")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--root", default="data/datasets/objectnav/hm3d/v1")
    args = parser.parse_args()

    source_root = os.path.join(args.root, args.source_split)
    target_root = os.path.join(args.root, args.target_split)
    source_top_level = os.path.join(source_root, f"{args.source_split}.json.gz")
    target_top_level = os.path.join(target_root, f"{args.target_split}.json.gz")

    top_level = load_json_gz(source_top_level)
    target_top = {
        "episodes": [],
        "category_to_task_category_id": top_level["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": top_level["category_to_scene_annotation_category_id"],
    }
    dump_json_gz(target_top_level, target_top)

    manifest: Dict[str, Any] = {
        "source_split": args.source_split,
        "target_split": args.target_split,
        "episodes_per_scene": args.episodes_per_scene,
        "seed": args.seed,
        "sample_mode": args.sample_mode,
        "scenes": [],
    }

    for scene_offset, scene in enumerate(args.scenes):
        source_path = os.path.join(source_root, "content", f"{scene}.json.gz")
        source_payload = load_json_gz(source_path)
        source_episodes = source_payload["episodes"]
        if len(source_episodes) < args.episodes_per_scene:
            raise ValueError(
                f"Scene {scene} only has {len(source_episodes)} episodes, "
                f"but {args.episodes_per_scene} were requested."
            )

        if args.sample_mode == "head":
            chosen_indices = list(range(args.episodes_per_scene))
        else:
            rng = random.Random(args.seed + scene_offset)
            chosen_indices = sorted(rng.sample(range(len(source_episodes)), args.episodes_per_scene))

        chosen_episodes: List[Dict[str, Any]] = []
        source_episode_ids: List[str] = []
        for new_episode_id, source_index in enumerate(chosen_indices):
            episode = copy.deepcopy(source_episodes[source_index])
            source_episode_ids.append(str(episode["episode_id"]))
            episode["episode_id"] = str(new_episode_id)
            info = episode.get("info", {})
            if not isinstance(info, dict):
                info = {}
            info["subset_source_index"] = source_index
            info["subset_source_episode_id"] = source_episode_ids[-1]
            episode["info"] = info
            chosen_episodes.append(episode)

        target_payload = {
            "goals_by_category": source_payload["goals_by_category"],
            "episodes": chosen_episodes,
            "category_to_task_category_id": source_payload["category_to_task_category_id"],
            "category_to_scene_annotation_category_id": source_payload["category_to_scene_annotation_category_id"],
        }
        dump_json_gz(os.path.join(target_root, "content", f"{scene}.json.gz"), target_payload)

        manifest["scenes"].append(
            {
                "scene": scene,
                "selected_indices": chosen_indices,
                "source_episode_ids": source_episode_ids,
            }
        )

    with open(os.path.join(target_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(args.scenes) * args.episodes_per_scene
    print(
        f"Created split {args.target_split} with {len(args.scenes)} scenes and "
        f"{total} episodes at {target_root}"
    )


if __name__ == "__main__":
    main()
