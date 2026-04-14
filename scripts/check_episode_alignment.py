import argparse
import glob
import json
import os
import sys
from typing import Dict, List


def load_episode_keys(directory: str) -> List[str]:
    keys = []
    seen: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.getsize(path) == 0:
            continue
        with open(path, "r") as f:
            data = json.load(f)

        scene_id = str(data["scene_id"])
        episode_id = str(data["episode_id"])
        key = f"{scene_id}:{episode_id}"

        if key in seen:
            print(f"Duplicate episode key detected in {directory}: {key}")
            print(f"  first: {seen[key]}")
            print(f"  again: {path}")
            sys.exit(1)

        seen[key] = path
        keys.append(key)
    return sorted(keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlfm-dir",
        default="results/pilot/vlfm_original_jsons",
        help="Directory containing original VLFM episode JSONs.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="results/pilot/greedy_frontier_jsons",
        help="Directory containing baseline episode JSONs.",
    )
    args = parser.parse_args()

    vlfm_keys = load_episode_keys(args.vlfm_dir)
    baseline_keys = load_episode_keys(args.baseline_dir)

    if vlfm_keys != baseline_keys:
        only_vlfm = sorted(set(vlfm_keys) - set(baseline_keys))
        only_baseline = sorted(set(baseline_keys) - set(vlfm_keys))
        print("Episode mismatch detected.")
        print(f"VLFM count: {len(vlfm_keys)}")
        print(f"Baseline count: {len(baseline_keys)}")
        print("Only in VLFM:", only_vlfm[:20])
        print("Only in Baseline:", only_baseline[:20])
        return 1

    print(f"Episode IDs match exactly by scene_id+episode_id. Count = {len(vlfm_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
