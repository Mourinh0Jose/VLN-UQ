import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Row = Dict[str, object]
EpisodeKey = Tuple[str, str]


def load_rows(directory: Path) -> Dict[EpisodeKey, Row]:
    rows: Dict[EpisodeKey, Row] = {}
    for path in sorted(directory.glob("*.json")):
        if path.stat().st_size == 0:
            continue
        with path.open("r") as f:
            row = json.load(f)
        rows[(str(row["scene_id"]), str(row["episode_id"]))] = row
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def stderr(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def col(rows: Iterable[Row], key: str) -> List[float]:
    return [float(row.get(key, 0.0)) for row in rows]


def summary_value(row: Row, key: str) -> float:
    summary = row.get("mc_episode_summary", {})
    if not isinstance(summary, dict):
        return 0.0
    return float(summary.get(key, 0.0))


def fmt_mean_stderr(values: Iterable[float], digits: int = 3) -> str:
    values = list(values)
    return f"{mean(values):.{digits}f} +/- {stderr(values):.{digits}f}"


def headline_table(method_rows: Dict[str, Dict[EpisodeKey, Row]]) -> List[str]:
    lines = [
        "## Headline Metrics",
        "",
        "| Method | N | Success | SPL | SoftSPL | Dist-to-goal | Target detected | Stop called |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, rows_by_key in method_rows.items():
        rows = list(rows_by_key.values())
        lines.append(
            f"| {method} | {len(rows)} | "
            f"{fmt_mean_stderr(col(rows, 'success'))} | "
            f"{fmt_mean_stderr(col(rows, 'spl'))} | "
            f"{fmt_mean_stderr(col(rows, 'soft_spl'))} | "
            f"{fmt_mean_stderr(col(rows, 'distance_to_goal'))} | "
            f"{fmt_mean_stderr(col(rows, 'target_detected'))} | "
            f"{fmt_mean_stderr(col(rows, 'stop_called'))} |"
        )
    return lines


def paired_table(method_rows: Dict[str, Dict[EpisodeKey, Row]], baseline: str) -> List[str]:
    baseline_rows = method_rows[baseline]
    lines = [
        "## Paired Against Semantic",
        "",
        "| Method | N paired | Both succeed | Both fail | Method only | Semantic only | Delta Success | Delta SPL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, rows_by_key in method_rows.items():
        if method == baseline:
            continue
        keys = sorted(set(baseline_rows) & set(rows_by_key))
        both_s = both_f = method_only = baseline_only = 0
        delta_success: List[float] = []
        delta_spl: List[float] = []
        for key in keys:
            base_row = baseline_rows[key]
            row = rows_by_key[key]
            base_success = float(base_row.get("success", 0.0))
            method_success = float(row.get("success", 0.0))
            if method_success >= 1.0 and base_success >= 1.0:
                both_s += 1
            elif method_success < 1.0 and base_success < 1.0:
                both_f += 1
            elif method_success >= 1.0:
                method_only += 1
            else:
                baseline_only += 1
            delta_success.append(method_success - base_success)
            delta_spl.append(float(row.get("spl", 0.0)) - float(base_row.get("spl", 0.0)))
        lines.append(
            f"| {method} | {len(keys)} | {both_s} | {both_f} | {method_only} | {baseline_only} | "
            f"{mean(delta_success):+.3f} | {mean(delta_spl):+.3f} |"
        )
    return lines


def outcome_table(method_rows: Dict[str, Dict[EpisodeKey, Row]]) -> List[str]:
    causes = sorted(
        {
            str(row.get("failure_cause", "unknown"))
            for rows_by_key in method_rows.values()
            for row in rows_by_key.values()
        }
    )
    lines = ["## Outcome Counts", ""]
    lines.append("| Outcome | " + " | ".join(method_rows.keys()) + " |")
    lines.append("|---|" + "|".join("---:" for _ in method_rows) + "|")
    for cause in causes:
        counts = []
        for rows_by_key in method_rows.values():
            counts.append(sum(1 for row in rows_by_key.values() if str(row.get("failure_cause")) == cause))
        lines.append(f"| {cause} | " + " | ".join(str(count) for count in counts) + " |")
    return lines


def mc_episode_table(method_rows: Dict[str, Dict[EpisodeKey, Row]]) -> List[str]:
    keys = [
        "num_decisions",
        "num_changed_decisions",
        "num_mc_changed_top_decisions",
        "avg_selected_mc_std",
        "avg_all_topk_mc_std",
        "cyclic_suppression_count",
    ]
    lines = [
        "## MC Episode Diagnostics",
        "",
        "| Method | " + " | ".join(keys) + " |",
        "|---|" + "|".join("---:" for _ in keys) + "|",
    ]
    for method, rows_by_key in method_rows.items():
        if method == "semantic":
            continue
        cells = [
            fmt_mean_stderr((summary_value(row, key) for row in rows_by_key.values()), digits=4)
            for key in keys
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines


def find_record(decision: Row, flag_key: str) -> Optional[Row]:
    records = decision.get("topk_frontier_records", [])
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get(flag_key):
            return record
    return None


def changed_decision_stats(rows_by_key: Dict[EpisodeKey, Row]) -> Dict[str, float]:
    changed = []
    for row in rows_by_key.values():
        for decision in row.get("mc_decision_log", []):
            if isinstance(decision, dict) and decision.get("changed_from_baseline"):
                changed.append(decision)
    baseline_stds: List[float] = []
    selected_stds: List[float] = []
    baseline_risks: List[float] = []
    selected_risks: List[float] = []
    lower_std_count = 0
    paired_std_count = 0
    for decision in changed:
        baseline = find_record(decision, "selected_by_baseline")
        selected = find_record(decision, "selected_final")
        if baseline is not None:
            baseline_stds.append(float(baseline.get("mc_std", 0.0)))
            baseline_risks.append(float(baseline.get("risk_score", 0.0)))
        if selected is not None:
            selected_stds.append(float(selected.get("mc_std", 0.0)))
            selected_risks.append(float(selected.get("risk_score", 0.0)))
        if baseline is not None and selected is not None:
            paired_std_count += 1
            if float(selected.get("mc_std", 0.0)) < float(baseline.get("mc_std", 0.0)):
                lower_std_count += 1
    return {
        "changed_decisions": float(len(changed)),
        "baseline_std": mean(baseline_stds),
        "selected_std": mean(selected_stds),
        "baseline_risk": mean(baseline_risks),
        "selected_risk": mean(selected_risks),
        "selected_lower_std_rate": lower_std_count / paired_std_count if paired_std_count > 0 else 0.0,
    }


def uq_table(method_rows: Dict[str, Dict[EpisodeKey, Row]]) -> List[str]:
    lines = [
        "## UQ Diagnostics",
        "",
        "| Method | Success avg selected std | Failure avg selected std | Changed decisions | "
        "Changed baseline std | Changed selected std | Selected lower-std rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, rows_by_key in method_rows.items():
        if method == "semantic":
            continue
        success_stds = [
            summary_value(row, "avg_selected_mc_std")
            for row in rows_by_key.values()
            if float(row.get("success", 0.0)) >= 1.0
        ]
        failure_stds = [
            summary_value(row, "avg_selected_mc_std")
            for row in rows_by_key.values()
            if float(row.get("success", 0.0)) < 1.0
        ]
        changed_stats = changed_decision_stats(rows_by_key)
        lines.append(
            f"| {method} | {mean(success_stds):.6f} | {mean(failure_stds):.6f} | "
            f"{changed_stats['changed_decisions']:.0f} | "
            f"{changed_stats['baseline_std']:.6f} | {changed_stats['selected_std']:.6f} | "
            f"{changed_stats['selected_lower_std_rate']:.3f} |"
        )
    return lines


def topk_stds(rows_by_key: Dict[EpisodeKey, Row]) -> List[float]:
    values: List[float] = []
    for row in rows_by_key.values():
        for decision in row.get("mc_decision_log", []):
            if not isinstance(decision, dict):
                continue
            for record in decision.get("topk_frontier_records", []):
                if isinstance(record, dict):
                    values.append(float(record.get("mc_std", 0.0)))
    return values


def histogram_counts(values: List[float], bins: int = 5) -> List[Tuple[float, float, int]]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [(lo, hi, len(values))]
    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(int((value - lo) / width), bins - 1)
        counts[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(bins)]


def std_histogram_table(method_rows: Dict[str, Dict[EpisodeKey, Row]]) -> List[str]:
    lines = [
        "## MC Std Histogram",
        "",
        "| Method | Bin low | Bin high | Count |",
        "|---|---:|---:|---:|",
    ]
    for method, rows_by_key in method_rows.items():
        if method == "semantic":
            continue
        for lo, hi, count in histogram_counts(topk_stds(rows_by_key)):
            lines.append(f"| {method} | {lo:.6f} | {hi:.6f} | {count} |")
    return lines


def episode_std_bins_table(method_rows: Dict[str, Dict[EpisodeKey, Row]], bins: int = 5) -> List[str]:
    lines = [
        "## Episode Uncertainty Bins",
        "",
        "| Method | Bin | N | Avg selected std range | Success rate | Avg SPL |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for method, rows_by_key in method_rows.items():
        if method == "semantic":
            continue
        rows = sorted(rows_by_key.values(), key=lambda row: summary_value(row, "avg_selected_mc_std"))
        if not rows:
            continue
        bin_count = min(bins, len(rows))
        for idx in range(bin_count):
            start = idx * len(rows) // bin_count
            end = (idx + 1) * len(rows) // bin_count
            bucket = rows[start:end]
            stds = [summary_value(row, "avg_selected_mc_std") for row in bucket]
            success = [float(row.get("success", 0.0)) for row in bucket]
            spl = [float(row.get("spl", 0.0)) for row in bucket]
            lines.append(
                f"| {method} | {idx + 1} | {len(bucket)} | "
                f"{min(stds):.6f} - {max(stds):.6f} | {mean(success):.3f} | {mean(spl):.3f} |"
            )
    return lines


def flip_lines(method_rows: Dict[str, Dict[EpisodeKey, Row]], baseline: str) -> List[str]:
    baseline_rows = method_rows[baseline]
    lines = ["## Success Flips", ""]
    for method, rows_by_key in method_rows.items():
        if method == baseline:
            continue
        lines.append(f"### {method} vs {baseline}")
        lines.append("")
        keys = sorted(set(baseline_rows) & set(rows_by_key))
        flips = []
        for key in keys:
            base_row = baseline_rows[key]
            row = rows_by_key[key]
            base_success = int(float(base_row.get("success", 0.0)))
            method_success = int(float(row.get("success", 0.0)))
            if base_success != method_success:
                flips.append((key, method_success, base_success, row, base_row))
        if not flips:
            lines.append("No success flips.")
            lines.append("")
            continue
        for key, method_success, base_success, row, base_row in flips:
            scene, episode = key
            lines.append(
                f"- episode={episode} scene={Path(scene).stem}: "
                f"{method}={method_success} ({row.get('failure_cause')}) "
                f"semantic={base_success} ({base_row.get('failure_cause')})"
            )
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/mc_risk_aware_eval10_20260426"))
    parser.add_argument("--methods", nargs="+", default=["semantic", "mc_mean", "mc_risk"])
    parser.add_argument("--baseline", default="semantic")
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    method_rows = {method: load_rows(args.root / method) for method in args.methods}
    lines: List[str] = [f"# MC Risk-Aware Analysis: {args.root}", ""]
    lines += headline_table(method_rows)
    lines += ["", ""]
    lines += paired_table(method_rows, args.baseline)
    lines += ["", ""]
    lines += outcome_table(method_rows)
    lines += ["", ""]
    lines += mc_episode_table(method_rows)
    lines += ["", ""]
    lines += uq_table(method_rows)
    lines += ["", ""]
    lines += std_histogram_table(method_rows)
    lines += ["", ""]
    lines += episode_std_bins_table(method_rows)
    lines += ["", ""]
    lines += flip_lines(method_rows, args.baseline)
    text = "\n".join(lines)
    print(text)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text + "\n")


if __name__ == "__main__":
    main()
