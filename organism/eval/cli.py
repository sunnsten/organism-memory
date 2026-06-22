import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from organism.config import OrganismConfig

from .runner.run import run_scenario
from .runner.artifact import RunArtifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    col_w = [
        max(len("scenario"), max(len(r["scenario"]) for r in rows)),
        max(len("mode"), max(len(r["mode"]) for r in rows)),
        6,  # pass
        6,  # fail
        7,  # total
        10, # success%
    ]
    headers = ["scenario", "mode", "pass", "fail", "total", "success%"]
    sep = "  ".join("-" * w for w in col_w)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + header_line)
    print(sep)
    for r in rows:
        rate = r["success_rate"]
        rate_str = f"{rate * 100:.1f}%" if rate is not None else "n/a"
        cells = [r["scenario"], r["mode"], str(r["pass"]), str(r["fail"]), str(r["total"]), rate_str]
        print("  ".join(c.ljust(w) for c, w in zip(cells, col_w)))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation scenarios for Organism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        type=str,
        help="Path to a single scenario JSON file",
    )
    group.add_argument(
        "--scenario-dir",
        type=str,
        help="Directory of scenario JSON files — runs all *.json files found",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="runs",
        help="Output directory for run artifacts (default: runs)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["B_memory_on"],
        choices=["A_memory_off", "B_memory_on"],
        help="Modes to run (default: B_memory_on)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to OrganismConfig YAML file (optional)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to database file (optional, creates temp if not specified)",
    )

    args = parser.parse_args()

    # Collect scenario paths
    scenario_paths: List[str] = []
    if args.scenario:
        scenario_paths = [args.scenario]
    else:
        scenario_dir = Path(args.scenario_dir)
        if not scenario_dir.is_dir():
            logger.error(f"--scenario-dir is not a directory: {scenario_dir}")
            sys.exit(1)
        scenario_paths = sorted(str(p) for p in scenario_dir.rglob("*.json"))
        if not scenario_paths:
            logger.error(f"No *.json files found in {scenario_dir}")
            sys.exit(1)
        logger.info(f"Found {len(scenario_paths)} scenario file(s) in {scenario_dir}")

    base_config: Optional[OrganismConfig] = None
    if args.config:
        try:
            base_config = OrganismConfig.from_yaml(args.config)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    any_failed = False

    for scenario_path in scenario_paths:
        try:
            scenario = load_scenario(scenario_path)
        except Exception as e:
            logger.error(f"Failed to load scenario {scenario_path}: {e}")
            any_failed = True
            continue

        test_id = scenario.get("test_id", Path(scenario_path).stem)

        for mode in args.modes:
            logger.info(f"Running {test_id} [{mode}] ...")
            try:
                artifact = run_scenario(
                    scenario=scenario,
                    mode=mode,
                    base_config=base_config,
                    db_path=args.db_path,
                    output_dir=args.out,
                )

                artifact_filename = (
                    f"{test_id}_{mode}_{artifact.run_id.replace(':', '-').replace('.', '-')}.json"
                )
                artifact_path = out_dir / artifact_filename
                artifact.save(str(artifact_path))

                m = artifact.metrics
                rate = m.get("success_rate")
                summary_rows.append({
                    "scenario": test_id,
                    "mode": mode,
                    "pass": m.get("successful_turns", 0),
                    "fail": m.get("failed_turns", 0),
                    "total": m.get("turns_with_expect", 0),
                    "success_rate": rate,
                })

                if m.get("failed_turns", 0) > 0:
                    any_failed = True

                logger.info(
                    f"  {test_id} [{mode}]: "
                    f"{m.get('successful_turns', 0)}/{m.get('turns_with_expect', 0)} passed "
                    f"— artifact: {artifact_path}"
                )

            except Exception as e:
                logger.error(f"Failed to run {test_id} [{mode}]: {e}", exc_info=True)
                any_failed = True
                summary_rows.append({
                    "scenario": test_id,
                    "mode": mode,
                    "pass": 0,
                    "fail": -1,
                    "total": -1,
                    "success_rate": None,
                })

    _print_summary(summary_rows)

    if any_failed:
        logger.warning("Some scenarios failed — see summary above.")
        sys.exit(1)
    else:
        logger.info("All scenarios passed.")


if __name__ == "__main__":
    main()
