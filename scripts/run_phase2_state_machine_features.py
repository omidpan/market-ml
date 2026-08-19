#!/usr/bin/env python3
"""
Milestone 1 Phase 2 — state-machine feature layer build/validate entry point.

Orchestrates the existing src/state_machine_features.py CLI (build) and adds
a lightweight, read-only post-build validation pass (row-count invariance,
forbidden-column absence, provenance presence) suitable for Concourse. Does
not duplicate src/state_machine_features.py's own join/feature logic, and
does not touch data/raw.

    python scripts/run_phase2_state_machine_features.py --mode build
    python scripts/run_phase2_state_machine_features.py --mode validate
    python scripts/run_phase2_state_machine_features.py --mode all
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from state_machine_features import (  # noqa: E402
    ACD_NUMERIC_BOOLEAN_COLUMNS,
    ALL_CATEGORICAL_COLUMNS,
    COMMON_MODELING_START,
    _load_state_machine_config_section,
)

SYMBOL = "nvda"
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"
FEATURES_1M = REPO_ROOT / "data/parquet/features_1m"
STATE_MACHINE_FEATURES_1M = REPO_ROOT / "data/parquet/state_machine_features_1m"
FEATURES_1M_ACD_V1 = REPO_ROOT / "data/parquet/features_1m_acd_v1"

FORBIDDEN_FIELDS = {
    "signal_outcomes", "regime_evaluation", "daily_acd_summary",
    "total_successful", "total_failed", "or_width_class", "buffer_width_class",
    "spacing_class", "environment_id", "or_width_category",
    "minutes_since_open",
}


def _run(cmd: list) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ {printable[:300]}{'...' if len(printable) > 300 else ''}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO_ROOT)


def build(
    features_root: Path,
    acd_root: Path | None,
    state_machine_output_root: Path,
    combined_output_root: Path,
) -> None:
    cmd = [
        sys.executable, SRC / "state_machine_features.py",
        "--mode", "all",
        "--symbol", SYMBOL,
        "--features-root", features_root,
        "--state-machine-output-root", state_machine_output_root,
        "--combined-output-root", combined_output_root,
        "--config", CONFIG_PATH,
    ]
    if acd_root is not None:
        cmd += ["--acd-root", acd_root]
    _run(cmd)


def _state_machine_features_1m_partitions(state_machine_output_root: Path) -> list[str]:
    """Exact canonical layout written by build_state_machine_features_1m():
    {root}/config_id=.../regime_config_id=.../symbol=.../year=.../month=.../*.parquet
    (state_machine_features.py:662-667)."""

    section = _load_state_machine_config_section(CONFIG_PATH)
    config_id = section["config_id"]
    regime_config_id = section["regime_config_id"]
    pattern = (
        state_machine_output_root
        / f"config_id={config_id}"
        / f"regime_config_id={regime_config_id}"
        / f"symbol={SYMBOL}"
        / "year=*" / "month=*" / "*.parquet"
    )
    return sorted(glob.glob(str(pattern)))


def _features_1m_acd_v1_partitions(combined_output_root: Path) -> list[str]:
    """Exact canonical layout written by build_combined_feature_table():
    {root}/symbol=.../year=.../month=.../*.parquet (state_machine_features.py:770-773)."""

    pattern = combined_output_root / f"symbol={SYMBOL}" / "year=*" / "month=*" / "*.parquet"
    return sorted(glob.glob(str(pattern)))


def _partition_row_count(files: list[str], root: Path) -> int:
    if not files:
        raise FileNotFoundError(f"No Parquet partitions found under {root}.")
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def _base_row_count(features_root: Path, min_trading_date: date | None) -> int:
    files = sorted(glob.glob(str(features_root / f"symbol={SYMBOL}" / "year=*/month=*/*.parquet")))
    if not files:
        raise FileNotFoundError(f"No Parquet partitions found under {features_root}.")
    total = 0
    for f in files:
        table = pq.ParquetFile(f).read(columns=["trading_date"])
        dates = table.column("trading_date").to_pandas()
        if min_trading_date is not None:
            total += int((dates >= min_trading_date).sum())
        else:
            total += len(dates)
    return total


def validate(
    features_root: Path,
    state_machine_output_root: Path,
    combined_output_root: Path,
) -> bool:
    checks: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))

    print("\nPhase-2 output validation:")

    common_start = date.fromisoformat(str(COMMON_MODELING_START))

    sm_files = _state_machine_features_1m_partitions(state_machine_output_root)
    acd_files = _features_1m_acd_v1_partitions(combined_output_root)

    sm_rows = _partition_row_count(sm_files, state_machine_output_root)
    base_rows_common = _base_row_count(features_root, common_start)
    check(
        "1. state_machine_features_1m row count == base rows on/after common start",
        sm_rows == base_rows_common,
        f"sm_rows={sm_rows} base_rows_common={base_rows_common}",
    )

    combined_rows = _partition_row_count(acd_files, combined_output_root)
    base_rows_common_v1 = _base_row_count(features_root, common_start)
    check(
        "2. features_1m_acd_v1 row count == base rows on/after common start",
        combined_rows == base_rows_common_v1,
        f"combined_rows={combined_rows} base_rows={base_rows_common_v1}",
    )

    sample_file = acd_files[0]
    schema_columns = set(pq.ParquetFile(sample_file).schema_arrow.names)

    check(
        "3. no forbidden/outcome fields present in combined schema",
        not (FORBIDDEN_FIELDS & schema_columns),
        f"present={FORBIDDEN_FIELDS & schema_columns}",
    )
    check(
        "4. provenance columns present",
        {"state_machine_feature_schema_version", "market_ml_feature_set_identity"}
        <= schema_columns,
    )
    check(
        "5. approved numeric/boolean ACD whitelist present",
        set(ACD_NUMERIC_BOOLEAN_COLUMNS) <= schema_columns,
    )
    check(
        "6. all raw ACD categorical columns present (pre-encoding)",
        set(ALL_CATEGORICAL_COLUMNS) <= schema_columns,
    )

    passed = all(ok for _, ok, _ in checks)
    print(f"\nOverall: {'PASS' if passed else 'FAIL'} ({sum(ok for _, ok, _ in checks)}/{len(checks)} checks)")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["build", "validate", "all"], default="all")
    parser.add_argument(
        "--features-root", type=Path, default=FEATURES_1M,
        help="Base features_1m root. Default: data/parquet/features_1m "
             "(unchanged local-run behavior). Override to point at a "
             "Concourse-mounted input directory.",
    )
    parser.add_argument(
        "--acd-root", type=Path, default=None,
        help="External ACD source root, passed through to "
             "state_machine_features.py --acd-root. Default: unset, which "
             "resolves to $HOME/acd_experiments_local (unchanged local-run "
             "behavior) or config/pipeline.yaml's state_machine_features.acd_root.",
    )
    parser.add_argument(
        "--state-machine-output-root", type=Path, default=STATE_MACHINE_FEATURES_1M,
        help="Default: data/parquet/state_machine_features_1m.",
    )
    parser.add_argument(
        "--combined-output-root", type=Path, default=FEATURES_1M_ACD_V1,
        help="Default: data/parquet/features_1m_acd_v1.",
    )
    args = parser.parse_args()

    if args.mode in ("build", "all"):
        build(
            args.features_root,
            args.acd_root,
            args.state_machine_output_root,
            args.combined_output_root,
        )

    passed = True
    if args.mode in ("validate", "all"):
        passed = validate(
            args.features_root,
            args.state_machine_output_root,
            args.combined_output_root,
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
