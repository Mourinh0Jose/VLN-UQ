import glob
import argparse
import json
import math
import os
import statistics
from collections import Counter
from typing import Dict, List, Tuple


Row = Dict[str, object]
EpisodeKey = Tuple[str, str]


def load(directory: str) -> List[Row]:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.getsize(path) == 0:
            continue
        with open(path, "r") as f:
            rows.append(json.load(f))
    return rows


def episode_key(row: Row) -> EpisodeKey:
    return (str(row["scene_id"]), str(row["episode_id"]))


def ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    sd = statistics.stdev(values)
    return 1.96 * sd / math.sqrt(len(values))


def summarize(name: str, rows: List[Row]) -> None:
    n = len(rows)
    if n == 0:
        print(f"\n=== {name} ===")
        print("  NO DATA")
        return

    success = [float(r.get("success", 0.0)) for r in rows]
    spl = [float(r.get("spl", 0.0)) for r in rows]
    soft_spl = [float(r.get("soft_spl", 0.0)) for r in rows]
    distance = [float(r.get("distance_to_goal", 0.0)) for r in rows]
    fail_counter = Counter(
        str(r.get("failure_cause", "unknown"))
        for r in rows
        if float(r.get("success", 0.0)) != 1.0
    )

    print(f"\n=== {name} (n={n}) ===")
    print(f"  Success:  {statistics.mean(success):.3f} +/- {ci95(success):.3f}")
    print(f"  SPL:      {statistics.mean(spl):.3f} +/- {ci95(spl):.3f}")
    print(f"  SoftSPL:  {statistics.mean(soft_spl):.3f} +/- {ci95(soft_spl):.3f}")
    print(f"  Dist2Goal:{statistics.mean(distance):.3f}")
    if fail_counter:
        print("  Failure causes:")
        for cause, count in fail_counter.most_common():
            print(f"    - {cause}: {count}")


def paired(vlfm_rows: List[Row], baseline_rows: List[Row]) -> None:
    vlfm_idx = {episode_key(r): r for r in vlfm_rows}
    baseline_idx = {episode_key(r): r for r in baseline_rows}
    keys = sorted(set(vlfm_idx) & set(baseline_idx))

    print(f"\n=== Paired (n={len(keys)}) ===")
    if not keys:
        print("  No paired episodes - alignment broken.")
        return

    success_diffs = []
    spl_diffs = []
    soft_diffs = []
    flips = []

    for key in keys:
        vlfm = vlfm_idx[key]
        baseline = baseline_idx[key]
        vlfm_success = float(vlfm.get("success", 0.0))
        baseline_success = float(baseline.get("success", 0.0))
        success_diffs.append(vlfm_success - baseline_success)
        spl_diffs.append(float(vlfm.get("spl", 0.0)) - float(baseline.get("spl", 0.0)))
        soft_diffs.append(float(vlfm.get("soft_spl", 0.0)) - float(baseline.get("soft_spl", 0.0)))
        if vlfm_success != baseline_success:
            flips.append((key, vlfm, baseline))

    print(f"  mean Delta Success (VLFM - Baseline): {statistics.mean(success_diffs):+.3f} +/- {ci95(success_diffs):.3f}")
    print(f"  mean Delta SPL     (VLFM - Baseline): {statistics.mean(spl_diffs):+.3f} +/- {ci95(spl_diffs):.3f}")
    print(f"  mean Delta SoftSPL (VLFM - Baseline): {statistics.mean(soft_diffs):+.3f} +/- {ci95(soft_diffs):.3f}")
    print(f"  success flips: {len(flips)}")

    if flips:
        print("  Flip episodes:")
        for (scene_id, episode_id), vlfm, baseline in flips:
            print(
                "    - "
                f"scene={scene_id} ep={episode_id} "
                f"VLFM={int(float(vlfm.get('success', 0.0)))}({vlfm.get('failure_cause', 'did_not_fail')}) "
                f"Baseline={int(float(baseline.get('success', 0.0)))}({baseline.get('failure_cause', 'did_not_fail')})"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlfm-dir", default="results/pilot/vlfm_original_jsons")
    parser.add_argument("--baseline-dir", default="results/pilot/greedy_frontier_jsons")
    args = parser.parse_args()

    vlfm_rows = load(args.vlfm_dir)
    baseline_rows = load(args.baseline_dir)
    summarize("VLFM Original", vlfm_rows)
    summarize("Greedy Frontier", baseline_rows)
    paired(vlfm_rows, baseline_rows)


if __name__ == "__main__":
    main()
