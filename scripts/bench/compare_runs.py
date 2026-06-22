from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


def _parse_summary(path: str) -> dict:
    """Parse a *_summary.txt into {category: score_pct} dict."""
    text = Path(path).read_text(encoding="utf-8")
    result = {}
    for line in text.splitlines():
        m = re.match(r'^(\S[\w\s-]+?)\s+\d+\s+([\d.]+)%', line.strip())
        if m:
            category = m.group(1).strip()
            score = float(m.group(2))
            result[category] = score
    return result


THRESHOLD_REGRESS = 2.0   # pp drop flagged as WARNING
THRESHOLD_CRITICAL = 5.0  # pp drop flagged as ERROR


def compare(baseline_path: str, new_path: str) -> bool:
    """Returns True if no critical regressions found."""
    baseline = _parse_summary(baseline_path)
    new = _parse_summary(new_path)

    all_categories = sorted(set(baseline) | set(new))
    ok = True
    rows = []
    for cat in all_categories:
        b = baseline.get(cat)
        n = new.get(cat)
        if b is None or n is None:
            rows.append(f"  {cat:<30} {'N/A':>8} -> {'N/A':>8}")
            continue
        delta = n - b
        if delta <= -THRESHOLD_CRITICAL:
            flag = "CRITICAL"
            ok = False
        elif delta <= -THRESHOLD_REGRESS:
            flag = "WARN"
        elif delta >= THRESHOLD_REGRESS:
            flag = "IMPROVE"
        else:
            flag = "ok"
        rows.append(f"  {cat:<30} {b:>7.1f}% -> {n:>7.1f}%  ({delta:+.1f}pp)  {flag}")

    print(f"Baseline: {baseline_path}")
    print(f"New:      {new_path}")
    print()
    for row in rows:
        print(row)
    print()
    if ok:
        print("No critical regressions.")
    else:
        print("Critical regression detected -- consider rollback.")
    return ok


def set_baseline(summary_path: str, name: str, out_dir: str = "runs") -> None:
    """Copy a summary as the named baseline."""
    baseline_path = Path(out_dir) / f"{name}_baseline.txt"
    baseline_path.write_text(Path(summary_path).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Baseline saved: {baseline_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark summary files")
    parser.add_argument("--set-baseline", metavar="SUMMARY_PATH",
                        help="Save this summary as the baseline for NAME")
    parser.add_argument("--name", default="locomo",
                        help="Baseline name (locomo / longmemeval)")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("baseline", nargs="?", help="Baseline summary file")
    parser.add_argument("new", nargs="?", help="New summary file to compare")
    args = parser.parse_args()

    if args.set_baseline:
        set_baseline(args.set_baseline, args.name, args.out_dir)
        return

    if not args.baseline or not args.new:
        parser.print_help()
        sys.exit(1)

    ok = compare(args.baseline, args.new)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
