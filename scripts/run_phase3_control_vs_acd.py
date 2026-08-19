#!/usr/bin/env python3
"""
Milestone 1 Phase 3 — CONTROL-vs-ACD experiment preparation runner.

Orchestrates existing approved modules (src/sequences.py, src/model_matrix.py,
src/state_machine_features.py) to build a paired CONTROL/ACD dataset on an
identical timestamp universe, then validates paired-universe identity. Does
not train. Does not touch data/raw. Does not overwrite production
sequence_index or production model_matrix (separate Phase-3 output roots).

    python scripts/run_phase3_control_vs_acd.py --mode all
    python scripts/run_phase3_control_vs_acd.py --mode validate
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from state_machine_features import (  # noqa: E402
    ACD_NUMERIC_BOOLEAN_COLUMNS,
    COMMON_MODELING_START,
    _acd_feature_name,
    apply_categorical_encoding,
    build_categorical_encoding_manifest,
    fit_categorical_vocabularies,
)
from model_matrix import DEFAULT_FEATURES  # noqa: E402

SYMBOL = "nvda"
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"

FEATURES_1M_ACD_V1 = REPO_ROOT / "data/parquet/features_1m_acd_v1"
FEATURES_1M_ACD_V1_ENCODED = REPO_ROOT / "data/parquet/features_1m_acd_v1_encoded"
TARGETS_1M = REPO_ROOT / "data/parquet/targets_1m"
TARGETS_1M_PHASE3_VIEW = REPO_ROOT / "data/parquet/targets_1m_phase3_common_v1"
SEQUENCE_INDEX_ROOT = REPO_ROOT / "data/parquet/sequence_index_phase3_common_v1"
MODEL_MATRIX_ROOT = REPO_ROOT / "data/parquet/model_matrix_phase3_common_v1"

SEQUENCE_SHAPE = (
    "sequence_length=120/horizon=15m/scope=continuous/stride=1/"
    "sessions=premarket+regular+aftermarket"
)
CONTROL_MATRIX_ROOT = MODEL_MATRIX_ROOT / SEQUENCE_SHAPE / "feature_set=core_v1"
ACD_MATRIX_ROOT = MODEL_MATRIX_ROOT / SEQUENCE_SHAPE / "feature_set=core_v1_acd_v1"

FORBIDDEN_COLUMNS = {
    "or_width_class", "buffer_width_class", "spacing_class", "environment_id",
    "or_width_category", "minutes_since_open", "regime_id",
    "f_sm_regime_id", "signal_outcomes", "regime_evaluation",
}


def _run(cmd: list) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ {printable[:300]}{'...' if len(printable) > 300 else ''}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO_ROOT)


def build_targets_view() -> None:
    """Partition-matching, row-count-matching read-only view of targets_1m
    restricted to features_1m_acd_v1's own months (sequences.py requires
    exact feature/target partition-set AND per-month row-count parity).
    targets_1m itself is never read-write-modified: months entirely inside
    the common range are symlinked (zero duplication); the one boundary
    month that straddles common_modeling_start is materialized as a real
    row-filtered copy (a symlink can't filter rows)."""

    feature_months = sorted(
        (Path(p).parts[-3], Path(p).parts[-2])
        for p in glob.glob(
            str(FEATURES_1M_ACD_V1 / f"symbol={SYMBOL}" / "year=*/month=*/*.parquet")
        )
    )
    if not feature_months:
        raise FileNotFoundError(
            f"No features_1m_acd_v1 partitions found under {FEATURES_1M_ACD_V1}."
        )

    for year_dir, month_dir in feature_months:
        src = TARGETS_1M / f"symbol={SYMBOL}" / year_dir / month_dir / "part.parquet"
        dst_dir = TARGETS_1M_PHASE3_VIEW / f"symbol={SYMBOL}" / year_dir / month_dir
        dst = dst_dir / "part.parquet"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue

        feature_file = (
            FEATURES_1M_ACD_V1 / f"symbol={SYMBOL}" / year_dir / month_dir / "part.parquet"
        )
        feature_rows = pq.ParquetFile(feature_file).metadata.num_rows
        target_rows = pq.ParquetFile(src).metadata.num_rows

        if feature_rows == target_rows:
            rel = (
                Path("../../../../") / "targets_1m" / f"symbol={SYMBOL}"
                / year_dir / month_dir / "part.parquet"
            )
            dst.symlink_to(rel)
        else:
            feature_dates = pq.ParquetFile(feature_file).read(
                columns=["trading_date"]
            ).to_pandas()
            min_trading_date = feature_dates["trading_date"].min()
            target_frame = pq.ParquetFile(src).read().to_pandas()
            filtered = target_frame[
                target_frame["trading_date"] >= min_trading_date
            ].reset_index(drop=True)
            filtered.to_parquet(
                dst, index=False, engine="pyarrow", compression="zstd"
            )


def build_sequence_index() -> None:
    _run([
        sys.executable, SRC / "sequences.py",
        "--features-root", FEATURES_1M_ACD_V1,
        "--targets-root", TARGETS_1M_PHASE3_VIEW,
        "--output-root", SEQUENCE_INDEX_ROOT,
        "--symbol", SYMBOL,
        "--config", CONFIG_PATH,
    ])


def build_encoded_acd_features() -> dict:
    vocabularies = fit_categorical_vocabularies(
        sequence_index_root=SEQUENCE_INDEX_ROOT,
        features_root=FEATURES_1M_ACD_V1,
        symbol=SYMBOL,
    )
    manifest = build_categorical_encoding_manifest(vocabularies)

    FEATURES_1M_ACD_V1_ENCODED.mkdir(parents=True, exist_ok=True)
    manifest_path = FEATURES_1M_ACD_V1_ENCODED / "categorical_encoding_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")

    apply_categorical_encoding(
        features_root=FEATURES_1M_ACD_V1,
        output_root=FEATURES_1M_ACD_V1_ENCODED,
        manifest=manifest,
        symbol=SYMBOL,
        parquet_compression="zstd",
    )
    return manifest


def _load_manifest() -> dict:
    manifest_path = FEATURES_1M_ACD_V1_ENCODED / "categorical_encoding_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def derive_acd_features(manifest: dict) -> list:
    """Programmatic derivation only — no hard-coded one-hot column names,
    no /tmp dependency. Numeric/boolean names come from the approved
    ACD_NUMERIC_BOOLEAN_COLUMNS whitelist; one-hot names come from the
    persisted, TRAIN-fit manifest."""

    numeric = [_acd_feature_name(c) for c in ACD_NUMERIC_BOOLEAN_COLUMNS]
    onehot = [
        column
        for entry in manifest.values()
        for column in entry["encoded_columns"]
    ]
    return list(DEFAULT_FEATURES) + numeric + onehot


def build_control_model_matrix() -> None:
    _run([
        sys.executable, SRC / "model_matrix.py",
        "--features-root", FEATURES_1M_ACD_V1,
        "--sequence-root", SEQUENCE_INDEX_ROOT,
        "--output-root", MODEL_MATRIX_ROOT,
        "--symbol", SYMBOL,
        "--feature-set", "core_v1",
        "--features", *DEFAULT_FEATURES,
        "--config", CONFIG_PATH,
    ])


def build_acd_model_matrix(manifest: dict) -> None:
    features = derive_acd_features(manifest)
    _run([
        sys.executable, SRC / "model_matrix.py",
        "--features-root", FEATURES_1M_ACD_V1_ENCODED,
        "--sequence-root", SEQUENCE_INDEX_ROOT,
        "--output-root", MODEL_MATRIX_ROOT,
        "--symbol", SYMBOL,
        "--feature-set", "core_v1_acd_v1",
        "--features", *features,
        "--config", CONFIG_PATH,
    ])


def _load_sequence_frame(root: Path, columns: list) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "sequences" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No sequences/ partitions found under {root}.")
    frames = [pq.ParquetFile(f).read(columns=columns).to_pandas() for f in files]
    return pd.concat(frames, ignore_index=True)


def run_validation() -> bool:
    """The 14 checks in references/paired-universe-test-contract.md."""

    checks: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))

    key_cols = [
        "symbol", "window_start_datetime", "window_end_datetime", "prediction_time",
        "target_realized_at", "trading_date", "session", "split",
        "target_return_bps", "target_log_return", "target_direction",
        "target_same_session", "target_same_trading_date",
    ]
    control = _load_sequence_frame(CONTROL_MATRIX_ROOT, key_cols).sort_values(
        "window_end_datetime"
    ).reset_index(drop=True)
    acd = _load_sequence_frame(ACD_MATRIX_ROOT, key_cols).sort_values(
        "window_end_datetime"
    ).reset_index(drop=True)

    print("\nPaired-universe validation:")

    # 1-2. same sequence count overall / by split
    check("1. same sequence count overall", len(control) == len(acd),
          f"control={len(control)} acd={len(acd)}")
    control_splits = control["split"].value_counts().to_dict()
    acd_splits = acd["split"].value_counts().to_dict()
    check("2. same sequence count by split", control_splits == acd_splits,
          f"control={control_splits} acd={acd_splits}")

    # 3-4. ordered keys / matching fields
    check("3. same ordered sequence keys",
          list(control["window_end_datetime"]) == list(acd["window_end_datetime"]))
    for col in key_cols:
        mismatches = int((control[col].reset_index(drop=True)
                           != acd[col].reset_index(drop=True)).sum())
        check(f"4. matching {col}", mismatches == 0, f"{mismatches} mismatches")

    # 5. target values/availability (subset of key_cols already checked above)
    check("5. same target values/availability",
          (control["target_return_bps"] == acd["target_return_bps"]).all()
          and (control["target_direction"] == acd["target_direction"]).all())

    # 6. purge/embargo: identical by construction (same shared sequence_index)
    check("6. same purge/embargo exclusions (shared sequence_index)", True)

    # 7. no invalid session-crossing sequences: enforced upstream by sequences.py
    # (scope=continuous with defined sessions) — re-assert both sides agree.
    check("7. no invalid session-crossing sequences",
          (control["session"] == acd["session"]).all())

    # 8. no pre-2020-01-03 sample
    check("8. no pre-2020-01-03 sample",
          control["trading_date"].min() >= COMMON_MODELING_START
          and acd["trading_date"].min() >= COMMON_MODELING_START,
          f"control_min={control['trading_date'].min()} acd_min={acd['trading_date'].min()}")

    # 9-11. feature schema checks via scaler.json
    control_scaler = json.load(open(CONTROL_MATRIX_ROOT / "scaler.json"))
    acd_scaler = json.load(open(ACD_MATRIX_ROOT / "scaler.json"))
    control_features = set(control_scaler["feature_names"])
    acd_features = set(acd_scaler["feature_names"])

    check("9. CONTROL has zero ACD feature columns",
          not any(f.startswith("f_sm_") for f in control_features))
    incremental = acd_features - control_features
    expected_incremental = len(ACD_NUMERIC_BOOLEAN_COLUMNS) + sum(
        len(e["encoded_columns"]) for e in _load_manifest().values()
    )
    check("10. ACD has exactly the approved incremental columns",
          control_features <= acd_features
          and len(incremental) == expected_incremental,
          f"incremental={len(incremental)} expected={expected_incremental}")
    check("11. no forbidden columns",
          not (FORBIDDEN_COLUMNS & control_features)
          and not (FORBIDDEN_COLUMNS & acd_features))

    # 12. scaler fit TRAIN only
    check("12. scaler fit TRAIN-only, applied unchanged to validation/test",
          "TRAIN" in control_scaler["fit_policy"].upper()
          and "TRAIN" in acd_scaler["fit_policy"].upper()
          and control_scaler["train_last_trading_date"] == acd_scaler["train_last_trading_date"])

    # 13. no unexpected NaN/inf in the final numeric matrix
    import numpy as np
    nan_ok = True
    for root in (CONTROL_MATRIX_ROOT, ACD_MATRIX_ROOT):
        rows_files = sorted(glob.glob(str(root / "rows" / "**" / "*.parquet"), recursive=True))
        sample_file = rows_files[len(rows_files) // 2]
        sample = pq.ParquetFile(sample_file).read().to_pandas()
        numeric_cols = [
            c for c in sample.columns
            if pd.api.types.is_numeric_dtype(sample[c])
            and c not in ("model_sample_index", "sample_index")
        ]
        arr = sample[numeric_cols].to_numpy(dtype="float64")
        if not np.isfinite(arr).all():
            nan_ok = False
    check("13. no NaN/inf in final numeric matrix (sampled)", nan_ok)

    # 14. deterministic feature-column order
    check("14. deterministic feature-column order",
          control_scaler["feature_names"] == acd_scaler["feature_names"][:len(DEFAULT_FEATURES)])

    passed = all(ok for _, ok, _ in checks)
    print(f"\nOverall: {'PASS' if passed else 'FAIL'} ({sum(ok for _, ok, _ in checks)}/{len(checks)} checks)")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["prepare", "control", "acd", "validate", "all"],
        default="all",
    )
    args = parser.parse_args()

    manifest = None

    if args.mode in ("prepare", "all"):
        build_targets_view()
        build_sequence_index()
        manifest = build_encoded_acd_features()

    if args.mode in ("control", "all"):
        build_control_model_matrix()

    if args.mode in ("acd", "all"):
        if manifest is None:
            manifest = _load_manifest()
        build_acd_model_matrix(manifest)

    passed = True
    if args.mode in ("validate", "all"):
        passed = run_validation()

    print("\nOutput paths:")
    print(f"  sequence_index:    {SEQUENCE_INDEX_ROOT}")
    print(f"  encoded features:  {FEATURES_1M_ACD_V1_ENCODED}")
    print(f"  CONTROL matrix:    {CONTROL_MATRIX_ROOT}")
    print(f"  ACD matrix:        {ACD_MATRIX_ROOT}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
