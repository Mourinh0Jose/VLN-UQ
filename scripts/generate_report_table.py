import glob
import json
import math
import os
from typing import Dict, List, Tuple


Row = Dict[str, object]


def load(directory: str) -> Dict[str, Row]:
    episodes: Dict[str, Row] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.getsize(path) == 0:
            continue
        with open(path, "r") as f:
            row = json.load(f)
        key = f"{row['scene_id']}/{row['episode_id']}"
        episodes[key] = row
    return episodes


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stderr(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def col(rows: List[Row], key: str) -> List[float]:
    return [float(r.get(key, 0.0)) for r in rows]


def headline_table(settings: List[Tuple[str, Dict[str, Row], Dict[str, Row]]]) -> List[str]:
    out = [
        "## Headline metrics (mean ± stderr)",
        "",
        "| Setting | Method | N | Success Rate | Avg SPL | Avg SoftSPL | Avg dist-to-goal (m) | target_detected | stop_called |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting, vlfm, greedy in settings:
        for name, data in (("VLFM Original", vlfm), ("Greedy Frontier", greedy)):
            rows = list(data.values())
            if not rows:
                out.append(f"| {setting} | {name} | - | - | - | - | - | - | - |")
                continue
            sr = col(rows, "success")
            spl = col(rows, "spl")
            ssp = col(rows, "soft_spl")
            dtg = col(rows, "distance_to_goal")
            det = [1.0 if r.get("target_detected") else 0.0 for r in rows]
            stp = [1.0 if r.get("stop_called") else 0.0 for r in rows]
            out.append(
                f"| {setting} | {name} | {len(rows)} | "
                f"{mean(sr):.1%} ± {stderr(sr):.1%} | "
                f"{mean(spl):.3f} ± {stderr(spl):.3f} | "
                f"{mean(ssp):.3f} ± {stderr(ssp):.3f} | "
                f"{mean(dtg):.2f} ± {stderr(dtg):.2f} | "
                f"{mean(det):.1%} ± {stderr(det):.1%} | "
                f"{mean(stp):.1%} ± {stderr(stp):.1%} |"
            )
    return out


def paired_table(settings: List[Tuple[str, Dict[str, Row], Dict[str, Row]]]) -> List[str]:
    out = [
        "## Paired comparison (same episodes, VLFM vs Greedy)",
        "",
        "| Setting | N paired | Both succeed | Both fail | VLFM only | Greedy only | ΔSR (VLFM−Greedy) | ΔSPL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting, vlfm, greedy in settings:
        keys = sorted(set(vlfm) & set(greedy))
        if not keys:
            out.append(f"| {setting} | - | - | - | - | - | - | - |")
            continue
        both_s = both_f = vonly = gonly = 0
        dsr = dspl = 0.0
        for k in keys:
            sv = float(vlfm[k].get("success", 0.0))
            sg = float(greedy[k].get("success", 0.0))
            if sv >= 1.0 and sg >= 1.0:
                both_s += 1
            elif sv < 1.0 and sg < 1.0:
                both_f += 1
            elif sv >= 1.0:
                vonly += 1
            else:
                gonly += 1
            dsr += sv - sg
            dspl += float(vlfm[k].get("spl", 0.0)) - float(greedy[k].get("spl", 0.0))
        n = len(keys)
        out.append(
            f"| {setting} | {n} | {both_s} | {both_f} | {vonly} | {gonly} | "
            f"{dsr / n:+.1%} | {dspl / n:+.3f} |"
        )
    return out


def outcome_table(settings: List[Tuple[str, Dict[str, Row], Dict[str, Row]]]) -> List[str]:
    out = ["## Outcome breakdown (counts)", ""]
    for setting, vlfm, greedy in settings:
        causes = sorted({r["failure_cause"] for r in list(vlfm.values()) + list(greedy.values())})
        out.append(f"### {setting}")
        out.append("")
        out.append("| outcome | VLFM | Greedy |")
        out.append("|---|---:|---:|")
        for cause in causes:
            cv = sum(1 for r in vlfm.values() if r["failure_cause"] == cause)
            cg = sum(1 for r in greedy.values() if r["failure_cause"] == cause)
            out.append(f"| {cause} | {cv} | {cg} |")
        out.append("")
    return out


def main() -> None:
    settings = [
        (
            "Pilot (30)",
            load("results/pilot/vlfm_original_jsons"),
            load("results/pilot/greedy_frontier_jsons"),
        ),
        (
            "Extended (90)",
            load("results/val5scene90/vlfm_original_jsons"),
            load("results/val5scene90/greedy_frontier_jsons"),
        ),
    ]

    lines: List[str] = []
    lines += headline_table(settings)
    lines += ["", ""]
    lines += paired_table(settings)
    lines += ["", ""]
    lines += outcome_table(settings)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
