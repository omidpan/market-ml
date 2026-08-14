#!/usr/bin/env python3
"""
build_label_policy.py

Build a NEW derived ATR-relative 3-class label-policy Parquet artifact.

This file is intentionally self-contained. It does NOT import implementation
objects from label_diagnostics.py, so the diagnostic CLI and production builder
can evolve independently without import/interface failures.

Inputs are READ-ONLY:
  data/parquet/features_1m/
  data/parquet/model_matrix/.../sequences/

Default policy:
  alpha_t = ATR(period)_t / close_t
  tau_t   = multiplier * alpha_t

  target_log_return < -tau_t       -> DOWN    (-1)
  abs(target_log_return) <= tau_t  -> NEUTRAL (0)
  target_log_return > +tau_t       -> UP      (+1)

Safety:
- Default invocation is a full dry-run: NOTHING is written.
- --write is required to materialize the derived Parquet artifact.
- Existing input Parquet is never modified.
- Existing label-policy Parquet is never overwritten.
- TEST labels may be deterministically materialized after the policy is frozen,
  but TEST class distributions and model metrics are intentionally not reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


VERSION = "1.1.0"
POLICY_TYPE = "atr_relative_3class"
POLICY_VERSION = "v1"


# ---------------------------------------------------------------------------
# CLI/path helpers
# ---------------------------------------------------------------------------

def parse_sessions(value: str) -> tuple[str, ...]:
    items = tuple(x.strip() for x in value.split(",") if x.strip())
    if not items:
        raise argparse.ArgumentTypeError("At least one session is required.")
    return items


def normalize_symbol(value: str) -> str:
    return value.strip().lower()


def path_number(value: float) -> str:
    text = f"{value:.12g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def sequence_root(project: Path, args: argparse.Namespace) -> Path:
    session_key = "+".join(args.sessions)
    return (
        project
        / "data"
        / "parquet"
        / "model_matrix"
        / f"sequence_length={args.sequence_length}"
        / f"horizon={args.horizon}m"
        / f"scope={args.scope}"
        / f"stride={args.stride}"
        / f"sessions={session_key}"
        / f"feature_set={args.feature_set}"
        / "sequences"
    )


def output_root(
    project: Path,
    args: argparse.Namespace,
    symbol: str,
) -> Path:
    return (
        project
        / "data"
        / "parquet"
        / "label_policy"
        / f"policy={args.policy_name}"
        / f"atr_period={args.atr_period}"
        / f"horizon={args.horizon}m"
        / f"multiplier={path_number(args.multiplier)}"
        / f"symbol={symbol}"
    )


def report_file(
    project: Path,
    args: argparse.Namespace,
    symbol: str,
) -> Path:
    name = (
        f"label_policy_build_{args.policy_name}"
        f"_atr{args.atr_period}"
        f"_h{args.horizon}"
        f"_m{path_number(args.multiplier)}.json"
    )
    return project / "reports" / "label_policy" / symbol / name


def symbol_parquet_files(root: Path, symbol: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    needle = f"symbol={symbol}"
    files = sorted(
        p
        for p in root.rglob("*.parquet")
        if needle in p.as_posix().lower()
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files for symbol={symbol!r} under {root}"
        )

    return files


def utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        utc=True,
        errors="raise",
    ).astype("int64")


# ---------------------------------------------------------------------------
# features_1m loading + causal Wilder ATR
# ---------------------------------------------------------------------------

def load_feature_ohlc(
    feature_root: Path,
    symbol: str,
) -> pd.DataFrame:
    print("\n[features] Loading features_1m OHLC...")

    files = symbol_parquet_files(feature_root, symbol)

    columns = [
        "prediction_time",
        "source_id",
        "high",
        "low",
        "close",
        "is_current_bar_usable",
        "session",
        "trading_date",
    ]

    frames: list[pd.DataFrame] = []

    for i, path in enumerate(files, 1):
        frames.append(pd.read_parquet(path, columns=columns))
        if i % 20 == 0 or i == len(files):
            print(f"  features: loaded {i:>3}/{len(files)} files")

    df = pd.concat(frames, ignore_index=True)

    df["_ts"] = utc_ns(df["prediction_time"])
    df["source_id"] = df["source_id"].astype(str)

    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_current_bar_usable"] = (
        df["is_current_bar_usable"]
        .fillna(False)
        .astype(bool)
    )

    df = (
        df.sort_values(["source_id", "_ts"], kind="stable")
        .reset_index(drop=True)
    )

    dup = df.duplicated(["source_id", "_ts"])
    if dup.any():
        raise RuntimeError(
            f"Duplicate features_1m endpoint keys: {int(dup.sum()):,}"
        )

    print(f"Feature rows: {len(df):,}")

    return df


def compute_wilder_atr(
    features: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """
    Causal Wilder ATR on exact physical 1-minute continuity blocks.

    Continuity:
      same source_id
      + exactly 60 seconds since prior row
      + current and prior row both usable

    At a continuity break:
      previous close is NOT imported across the gap;
      first TR in the new block is high-low.

    Seed:
      arithmetic mean of first `period` TR values in the block.

    Recurrence:
      ATR_t = ((period - 1) * ATR_{t-1} + TR_t) / period
    """
    if period <= 0:
        raise ValueError("ATR period must be positive.")

    print(f"\n[atr] Calculating causal Wilder ATR{period}...")

    high = features["high"].to_numpy(dtype=np.float64)
    low = features["low"].to_numpy(dtype=np.float64)
    close = features["close"].to_numpy(dtype=np.float64)
    ts = features["_ts"].to_numpy(dtype=np.int64)
    source = features["source_id"].to_numpy()

    usable = (
        features["is_current_bar_usable"].to_numpy()
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (high > 0)
        & (low > 0)
        & (close > 0)
    )

    n = len(features)

    same_source = np.zeros(n, dtype=bool)
    same_source[1:] = source[1:] == source[:-1]

    one_minute = np.zeros(n, dtype=bool)
    one_minute[1:] = (
        (ts[1:] - ts[:-1]) == 60_000_000_000
    )

    previous_usable = np.zeros(n, dtype=bool)
    previous_usable[1:] = usable[:-1]

    continuous = (
        same_source
        & one_minute
        & usable
        & previous_usable
    )

    # Any row not continuous from its predecessor starts a new block.
    block_start = ~continuous

    # First row of each block uses high-low.
    tr = high - low

    # Continuous rows may use previous close.
    idx = np.flatnonzero(continuous)
    if len(idx):
        prev_close = close[idx - 1]
        tr[idx] = np.maximum.reduce(
            [
                high[idx] - low[idx],
                np.abs(high[idx] - prev_close),
                np.abs(low[idx] - prev_close),
            ]
        )

    tr[~usable] = np.nan

    atr = np.full(n, np.nan, dtype=np.float64)

    starts = np.flatnonzero(block_start)
    ends = np.r_[starts[1:], n]

    qualifying_blocks = 0

    for start, end in zip(starts, ends):
        if not usable[start]:
            continue

        length = end - start
        if length < period:
            continue

        values = tr[start:end]

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Non-finite true range inside usable continuity block "
                f"[{start}:{end}]"
            )

        seed_index = start + period - 1
        atr[seed_index] = np.mean(values[:period])

        for pos in range(seed_index + 1, end):
            atr[pos] = (
                ((period - 1) * atr[pos - 1]) + tr[pos]
            ) / period

        qualifying_blocks += 1

    relative = np.divide(
        atr,
        close,
        out=np.full_like(atr, np.nan),
        where=(
            np.isfinite(atr)
            & np.isfinite(close)
            & (close > 0)
        ),
    )

    out = features.copy()
    out[f"atr{period}_1m"] = atr
    out[f"atr{period}_relative"] = relative
    out[f"atr{period}_bps"] = relative * 10_000.0

    print(f"Usable OHLC rows: {int(usable.sum()):,}")
    print(f"Rows with ATR{period}: {int(np.isfinite(atr).sum()):,}")
    print(f"Continuous ATR blocks: {qualifying_blocks:,}")

    return out


# ---------------------------------------------------------------------------
# Model-ready sequence loading
# ---------------------------------------------------------------------------

def load_sequences(root: Path, symbol: str) -> pd.DataFrame:
    print("\n[sequences] Loading all model-ready sequence rows...")

    files = symbol_parquet_files(root, symbol)

    # Read only one file to discover schema.
    first_columns = list(pd.read_parquet(files[0]).columns)

    required = [
        "prediction_time",
        "source_id",
        "split",
        "session",
        "trading_date",
        "target_log_return",
        "target_return_bps",
    ]

    missing = sorted(set(required) - set(first_columns))
    if missing:
        raise RuntimeError(
            f"Sequence Parquet missing required columns: {missing}"
        )

    optional = [
        "model_sample_index",
        "sample_index",
        "symbol",
        "calendar_date",
        "target_realized_at",
        "target_direction",
        "target_same_session",
        "target_same_trading_date",
        "feature_set",
        "feature_signature",
        "sequence_length",
        "target_horizon_minutes",
        "stride_minutes",
        "sequence_scope",
    ]

    columns = required + [
        col for col in optional if col in first_columns
    ]

    frames: list[pd.DataFrame] = []

    for i, path in enumerate(files, 1):
        frames.append(pd.read_parquet(path, columns=columns))
        if i % 20 == 0 or i == len(files):
            print(f"  sequences: loaded {i:>3}/{len(files)} files")

    df = pd.concat(frames, ignore_index=True)

    df["source_id"] = df["source_id"].astype(str)
    df["_ts"] = utc_ns(df["prediction_time"])

    for col in ["target_log_return", "target_return_bps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        values = df[col].to_numpy(dtype=np.float64)
        bad = int((~np.isfinite(values)).sum())
        if bad:
            raise RuntimeError(
                f"Non-finite {col} rows: {bad:,}"
            )

    dup = df.duplicated(["source_id", "_ts"])
    if dup.any():
        raise RuntimeError(
            "Duplicate model-ready endpoint keys "
            f"(source_id,prediction_time): {int(dup.sum()):,}"
        )

    if "model_sample_index" in df.columns:
        dup_model_index = df["model_sample_index"].duplicated()
        if dup_model_index.any():
            raise RuntimeError(
                "Duplicate model_sample_index rows: "
                f"{int(dup_model_index.sum()):,}"
            )

    print(f"Model-ready sequence rows: {len(df):,}")

    split_counts = (
        df["split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .sort_index()
    )

    for name, count in split_counts.items():
        print(f"  {name:>10}: {int(count):,}")

    return df


# ---------------------------------------------------------------------------
# Join + label policy
# ---------------------------------------------------------------------------

def join_atr(
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    print("\n[join] Joining causal ATR to model-ready endpoints...")

    rel_col = f"atr{period}_relative"

    atr_map = features.loc[
        np.isfinite(features[rel_col]),
        [
            "source_id",
            "_ts",
            "close",
            f"atr{period}_1m",
            rel_col,
            f"atr{period}_bps",
        ],
    ].copy()

    df = sequences.merge(
        atr_map,
        on=["source_id", "_ts"],
        how="left",
        validate="one_to_one",
    )

    missing = int(df[rel_col].isna().sum())

    print(f"Model-ready rows missing ATR{period}: {missing:,}")

    if missing:
        examples = df.loc[
            df[rel_col].isna(),
            [
                "prediction_time",
                "source_id",
                "split",
                "session",
                "trading_date",
            ],
        ].head(10)

        print("\nExamples with missing ATR:")
        print(examples.to_string(index=False))

        raise RuntimeError(
            "Missing causal ATR on model-ready endpoints. "
            "No label artifact will be written."
        )

    return df


def make_policy_metadata(
    args: argparse.Namespace,
    symbol: str,
) -> dict:
    return {
        "policy_type": POLICY_TYPE,
        "policy_name": args.policy_name,
        "policy_version": POLICY_VERSION,
        "symbol": symbol,
        "target_horizon_minutes": args.horizon,
        "sequence_length": args.sequence_length,
        "feature_set": args.feature_set,
        "scope": args.scope,
        "stride_minutes": args.stride,
        "sessions": list(args.sessions),
        "atr": {
            "timeframe": "1min",
            "period": args.atr_period,
            "method": "Wilder",
            "seed": f"mean of first {args.atr_period} TR values",
            "continuity_rule": (
                "reset whenever exact physical 1-minute continuity breaks"
            ),
            "first_tr_after_break": (
                "high-low; previous close is not imported across the gap"
            ),
            "causal": True,
        },
        "threshold": {
            "multiplier": args.multiplier,
            "alpha": f"ATR{args.atr_period}_t / close_t",
            "tau": "multiplier * alpha_t",
        },
        "label_rule": {
            "return": f"target_log_return at exact H={args.horizon}m",
            "down": "target_log_return < -tau",
            "neutral": "abs(target_log_return) <= tau",
            "up": "target_log_return > +tau",
            "mapping": {
                "down": -1,
                "neutral": 0,
                "up": 1,
            },
        },
        "selection_note": (
            "Multiplier selected/frozen using TRAIN-only diagnostics."
        ),
    }


def policy_signature(metadata: dict) -> str:
    raw = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def apply_policy(
    df: pd.DataFrame,
    args: argparse.Namespace,
    signature: str,
) -> pd.DataFrame:
    print("\n[labels] Applying frozen ATR-relative policy...")

    relative = df[
        f"atr{args.atr_period}_relative"
    ].to_numpy(dtype=np.float64)

    returns = df[
        "target_log_return"
    ].to_numpy(dtype=np.float64)

    tau = args.multiplier * relative

    labels = np.zeros(len(df), dtype=np.int8)
    labels[returns < -tau] = -1
    labels[returns > tau] = 1

    names = np.empty(len(labels), dtype=object)
    names[labels == -1] = "down"
    names[labels == 0] = "neutral"
    names[labels == 1] = "up"

    out = df.copy()

    out["threshold_multiplier"] = np.float64(args.multiplier)
    out["threshold_relative"] = tau
    out["threshold_bps"] = tau * 10_000.0

    out["label_direction"] = labels
    out["label_name"] = names

    out["policy_name"] = args.policy_name
    out["policy_version"] = POLICY_VERSION
    out["policy_signature"] = signature

    return out


# ---------------------------------------------------------------------------
# Validation/reporting
# ---------------------------------------------------------------------------

def validate_table(
    df: pd.DataFrame,
    expected_rows: int,
    period: int,
) -> None:
    print("\n[validate] Validating label table...")

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Row-count mismatch: expected {expected_rows:,}, "
            f"got {len(df):,}"
        )

    dup = df.duplicated(["source_id", "_ts"])
    if dup.any():
        raise RuntimeError(
            f"Duplicate output keys: {int(dup.sum()):,}"
        )

    actual_labels = set(
        df["label_direction"].astype(int).unique().tolist()
    )

    if not actual_labels.issubset({-1, 0, 1}):
        raise RuntimeError(
            f"Unexpected label values: {sorted(actual_labels)}"
        )

    finite_columns = [
        "target_log_return",
        "target_return_bps",
        f"atr{period}_1m",
        f"atr{period}_relative",
        f"atr{period}_bps",
        "threshold_relative",
        "threshold_bps",
    ]

    for col in finite_columns:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        bad = int((~np.isfinite(values)).sum())

        if bad:
            raise RuntimeError(
                f"Non-finite values in {col}: {bad:,}"
            )

    negative_thresholds = int(
        (df["threshold_relative"] < 0).sum()
    )

    if negative_thresholds:
        raise RuntimeError(
            f"Negative thresholds: {negative_thresholds:,}"
        )

    print("Validation: PASS")


def class_distribution(df: pd.DataFrame) -> dict:
    total = len(df)
    result: dict[str, dict[str, float | int]] = {}

    for value, name in [
        (-1, "down"),
        (0, "neutral"),
        (1, "up"),
    ]:
        count = int((df["label_direction"] == value).sum())
        result[name] = {
            "count": count,
            "pct": (
                100.0 * count / total
                if total
                else float("nan")
            ),
        }

    return result


def print_train_distribution(df: pd.DataFrame) -> dict:
    train = df.loc[
        df["split"].astype(str).str.lower().eq("train")
    ]

    dist = class_distribution(train)

    print("\nTRAIN label distribution:")

    for key in ["down", "neutral", "up"]:
        print(
            f"  {key.upper():<7} "
            f"{dist[key]['count']:>10,} "
            f"({dist[key]['pct']:7.3f}%)"
        )

    return dist


def build_report(
    df: pd.DataFrame,
    args: argparse.Namespace,
    metadata: dict,
    signature: str,
    out_root: Path,
    train_distribution: dict,
) -> dict:
    split_counts = (
        df["split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    threshold_bps = df["threshold_bps"].to_numpy(
        dtype=np.float64
    )

    return {
        "builder": "build_label_policy.py",
        "builder_version": VERSION,
        "status": "PASS",
        "write_enabled": bool(args.write),
        "input_parquet_impact": "READ_ONLY",
        "policy_metadata": metadata,
        "policy_signature": signature,
        "output_root": str(out_root),
        "rows_total": int(len(df)),
        "split_row_counts": {
            str(k): int(v)
            for k, v in split_counts.items()
        },
        "train_class_distribution": train_distribution,
        "threshold_bps_all_model_ready": {
            "p25": float(np.quantile(threshold_bps, 0.25)),
            "p50": float(np.quantile(threshold_bps, 0.50)),
            "p75": float(np.quantile(threshold_bps, 0.75)),
        },
        "test_policy": (
            "TEST labels may be deterministically materialized from the "
            "already-frozen policy, but TEST class distributions/model "
            "metrics are intentionally not reported."
        ),
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def refuse_existing_artifact(out_root: Path) -> None:
    if (
        out_root.exists()
        and any(out_root.rglob("*.parquet"))
    ):
        raise FileExistsError(
            "Target label-policy artifact already contains Parquet files:\n"
            f"  {out_root}\n"
            "Refusing to overwrite it."
        )


def output_columns(
    df: pd.DataFrame,
    period: int,
) -> list[str]:
    preferred = [
        "model_sample_index",
        "sample_index",
        "prediction_time",
        "target_realized_at",
        "source_id",
        "symbol",
        "calendar_date",
        "trading_date",
        "session",
        "split",
        "sequence_length",
        "target_horizon_minutes",
        "stride_minutes",
        "sequence_scope",
        "feature_set",
        "feature_signature",
        "target_return_bps",
        "target_log_return",
        "target_direction",
        "target_same_session",
        "target_same_trading_date",
        "close",
        f"atr{period}_1m",
        f"atr{period}_relative",
        f"atr{period}_bps",
        "threshold_multiplier",
        "threshold_relative",
        "threshold_bps",
        "label_direction",
        "label_name",
        "policy_name",
        "policy_version",
        "policy_signature",
        "year",
        "month",
    ]

    return [
        col for col in preferred
        if col in df.columns
    ]


def write_artifact(
    df: pd.DataFrame,
    out_root: Path,
    report: dict,
    args: argparse.Namespace,
) -> int:
    refuse_existing_artifact(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    frame = df.copy()

    trading_dates = pd.to_datetime(
        frame["trading_date"],
        errors="raise",
    )

    frame["year"] = trading_dates.dt.year.astype(np.int16)
    frame["month"] = trading_dates.dt.month.astype(np.int8)

    cols = output_columns(
        frame,
        args.atr_period,
    )

    files_written = 0

    for (year, month), group in frame.groupby(
        ["year", "month"],
        sort=True,
    ):
        partition = (
            out_root
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
        )

        partition.mkdir(
            parents=True,
            exist_ok=True,
        )

        group[cols].to_parquet(
            partition / "part.parquet",
            index=False,
            compression=args.compression,
        )

        files_written += 1

    manifest = dict(report)
    manifest["artifact_kind"] = "derived_label_policy"
    manifest["partitioning"] = ["symbol", "year", "month"]
    manifest["compression"] = args.compression
    manifest["files_written"] = files_written

    with open(
        out_root / "manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    return files_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned ATR-relative 3-class label-policy artifact."
        )
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help="market_ml project root. Default: current directory.",
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol, e.g. nvda.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--feature-set",
        default="core_v1",
    )

    parser.add_argument(
        "--scope",
        default="continuous",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--sessions",
        type=parse_sessions,
        default=("premarket", "regular", "aftermarket"),
    )

    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--multiplier",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--policy-name",
        default=f"{POLICY_TYPE}_{POLICY_VERSION}",
    )

    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4"],
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Create the NEW derived Parquet artifact. "
            "Without this flag the run is validation-only."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon <= 0:
        raise SystemExit("--horizon must be positive.")

    if args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be positive.")

    if args.stride <= 0:
        raise SystemExit("--stride must be positive.")

    if args.atr_period <= 0:
        raise SystemExit("--atr-period must be positive.")

    if (
        not math.isfinite(args.multiplier)
        or args.multiplier < 0
    ):
        raise SystemExit(
            "--multiplier must be finite and non-negative."
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    project = Path(
        args.project_root
    ).expanduser().resolve()

    symbol = normalize_symbol(args.symbol)

    feature_root = (
        project
        / "data"
        / "parquet"
        / "features_1m"
    )

    seq_root = sequence_root(
        project,
        args,
    )

    out_root = output_root(
        project,
        args,
        symbol,
    )

    rpt_file = report_file(
        project,
        args,
        symbol,
    )

    print(f"build_label_policy.py version {VERSION}")
    print(f"Project root: {project}")
    print(f"Symbol: {symbol}")
    print(f"Policy: {args.policy_name}")
    print(
        f"ATR: causal Wilder ATR{args.atr_period} "
        "on exact-continuity 1-minute bars"
    )
    print(f"Horizon: {args.horizon}m")
    print(f"Multiplier: {args.multiplier}")
    print("Input Parquet impact: READ-ONLY")
    print(f"Write enabled: {args.write}")
    print(f"Feature root: {feature_root}")
    print(f"Sequence root: {seq_root}")
    print(f"Output root: {out_root}")

    if args.write:
        refuse_existing_artifact(out_root)

    # 1. Read frozen feature rows and compute causal ATR.
    features = load_feature_ohlc(
        feature_root,
        symbol,
    )

    features = compute_wilder_atr(
        features,
        args.atr_period,
    )

    # 2. Read all frozen model-ready sequence endpoints.
    sequences = load_sequences(
        seq_root,
        symbol,
    )

    # 3. Exact endpoint join.
    joined = join_atr(
        features,
        sequences,
        args.atr_period,
    )

    # 4. Freeze metadata/signature and materialize deterministic labels.
    metadata = make_policy_metadata(
        args,
        symbol,
    )

    signature = policy_signature(
        metadata
    )

    print(f"\nPolicy signature: {signature}")

    labeled = apply_policy(
        joined,
        args,
        signature,
    )

    # 5. Validate before any write is possible.
    validate_table(
        labeled,
        expected_rows=len(sequences),
        period=args.atr_period,
    )

    train_dist = print_train_distribution(
        labeled
    )

    report = build_report(
        labeled,
        args,
        metadata,
        signature,
        out_root,
        train_dist,
    )

    if not args.write:
        print("\nDRY-RUN PASS")
        print("No Parquet or report files were created.")
        print(
            "Re-run with --write only after reviewing this output."
        )
        return

    # 6. Write only the NEW derived label artifact.
    files_written = write_artifact(
        labeled,
        out_root,
        report,
        args,
    )

    report["files_written"] = files_written
    report["manifest_file"] = str(
        out_root / "manifest.json"
    )

    rpt_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        rpt_file,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print("\nWRITE PASS")
    print(f"Parquet files written: {files_written}")
    print(f"Derived artifact: {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    print(f"Report: {rpt_file}")
    print(
        "Existing input Parquet datasets were not modified."
    )


if __name__ == "__main__":
    main()