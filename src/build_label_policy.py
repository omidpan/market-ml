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
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq


VERSION = "1.1.0"
POLICY_TYPE = "atr_relative_3class"
POLICY_VERSION = "v1"
ARTIFACT_CONTRACT_VERSION = "v2"


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
        / args.model_matrix_root
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
        / args.label_policy_root
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
    return project / args.reports_root / symbol / name


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


def _partition_year_month(path: Path) -> tuple[int, int] | None:
    """
    Best-effort extraction of (year, month) from a `year=YYYY/month=MM`
    partition path component. Returns None if the path doesn't follow that
    convention — callers must treat None as "cannot skip this file", not
    as an error; the Arrow-level row filter remains fully correct and
    sufficient either way. This is purely a whole-file skip optimization,
    never a substitute for the row-level predicate.
    """
    year = None
    month = None
    for part in path.parts:
        if part.startswith("year="):
            try:
                year = int(part.split("=", 1)[1])
            except ValueError:
                return None
        elif part.startswith("month="):
            try:
                month = int(part.split("=", 1)[1])
            except ValueError:
                return None
    if year is None or month is None:
        return None
    return (year, month)


def load_feature_ohlc_bounded(
    feature_root: Path,
    symbol: str,
    *,
    max_trading_date,
) -> pd.DataFrame:
    """
    v2 build-path counterpart to `load_feature_ohlc`. `features_1m` has no
    split column at all — it is the continuous, pre-split causal series,
    same status as `features_root` in `model_matrix.py`. Causal Wilder ATR
    is strictly backward-looking (a TRAIN/VALIDATION row's ATR value never
    depends on a future, TEST-period bar), so truncating this read at the
    TRAIN+VALIDATION boundary produces identical ATR values for every
    retained row while never reading a TEST-period OHLC bar into Python at
    all.

    Whole files entirely beyond `max_trading_date`'s month are skipped
    without being opened at all (best-effort, via partition-path parsing).
    Every file that is opened is read through an Arrow-level
    predicate-pushdown filter (`trading_date <= max_trading_date`) inside
    `pyarrow.dataset(...).to_table(filter=...)`, evaluated before
    `.to_pandas()` — never a plain unfiltered read followed by a pandas
    filter.
    """
    print(
        "\n[features] Loading features_1m OHLC "
        f"(v2, bounded at trading_date <= {max_trading_date})..."
    )

    files = symbol_parquet_files(feature_root, symbol)

    boundary_year_month = (
        max_trading_date.year,
        max_trading_date.month,
    )

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
    skipped = 0
    opened = 0

    for i, path in enumerate(files, 1):
        year_month = _partition_year_month(path)
        if (
            year_month is not None
            and year_month > boundary_year_month
        ):
            skipped += 1
            continue

        dataset = pa_dataset.dataset(
            path,
            format="parquet",
            partitioning=None,
        )

        bound = pa.scalar(
            max_trading_date,
            type=dataset.schema.field(
                "trading_date"
            ).type,
        )

        table = dataset.to_table(
            columns=columns,
            filter=(
                pc.field("trading_date")
                <= bound
            ),
        )

        opened += 1

        if table.num_rows:
            out = table.to_pandas()
            if (
                out["trading_date"].max()
                > max_trading_date
            ):
                raise AssertionError(
                    f"{path}: predicate-pushdown filter failed to "
                    "exclude rows beyond max_trading_date."
                )
            frames.append(out)

        if i % 20 == 0 or i == len(files):
            print(
                f"  features: scanned {i:>3}/{len(files)} files "
                f"(opened={opened}, skipped_whole_file={skipped})"
            )

    if not frames:
        raise RuntimeError(
            "No features_1m rows survived the TRAIN+VALIDATION boundary "
            "filter."
        )

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

    print(f"Feature rows (bounded): {len(df):,}")

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

SEQUENCE_REQUIRED_INPUT_COLUMNS = [
    "prediction_time",
    "source_id",
    "split",
    "session",
    "trading_date",
    "target_log_return",
    "target_return_bps",
]

SEQUENCE_OPTIONAL_INPUT_COLUMNS = [
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


def _normalize_sequences_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shared normalization/validation for a concatenated sequence-endpoint
    frame, regardless of how its rows were read (unfiltered
    `pd.read_parquet`, for v1's `load_sequences`, or an Arrow-filtered
    TRAIN+VALIDATION-only read, for v2's `load_sequences_trainval_v2`).
    Pure extraction of v1's existing logic — behavior for any given input
    frame is unchanged.
    """
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


def load_sequences(root: Path, symbol: str) -> pd.DataFrame:
    print("\n[sequences] Loading all model-ready sequence rows...")

    files = symbol_parquet_files(root, symbol)

    # Read only one file to discover schema.
    first_columns = list(pd.read_parquet(files[0]).columns)

    missing = sorted(
        set(SEQUENCE_REQUIRED_INPUT_COLUMNS) - set(first_columns)
    )
    if missing:
        raise RuntimeError(
            f"Sequence Parquet missing required columns: {missing}"
        )

    columns = SEQUENCE_REQUIRED_INPUT_COLUMNS + [
        col
        for col in SEQUENCE_OPTIONAL_INPUT_COLUMNS
        if col in first_columns
    ]

    frames: list[pd.DataFrame] = []

    for i, path in enumerate(files, 1):
        frames.append(pd.read_parquet(path, columns=columns))
        if i % 20 == 0 or i == len(files):
            print(f"  sequences: loaded {i:>3}/{len(files)} files")

    df = pd.concat(frames, ignore_index=True)

    return _normalize_sequences_frame(df)


def load_sequences_trainval_v2(
    root: Path,
    symbol: str,
) -> pd.DataFrame:
    """
    Fail-closed v2 sequence-endpoint reader. Unlike `load_sequences` (v1,
    unfiltered — reads and materializes every column, including
    `target_log_return`/`target_return_bps`, for every row before
    anything is checked), this function never projects a target/value
    column into Python until every file has been structurally proven
    TEST-free.

    STEP 1 — structural-only precheck: project ONLY the `split` column,
    per file, via `pyarrow.dataset`. Require the physical split values to
    be a subset of {"train", "validation"}. Fail immediately — before any
    target/value column is projected or converted to pandas, for any
    file — on "test", any other unexpected value, or a null/invalid split.

    STEP 2 — only after STEP 1 passes for every file, read the full
    required+optional column set via `dataset.to_table(filter=...)` with
    an Arrow predicate equivalent to `split == "train" OR split ==
    "validation"`, evaluated inside the Arrow scanner before
    `.to_pandas()`. Defense in depth on top of STEP 1, not a substitute.

    STEP 3 — the resulting concatenated pandas frame is passed through the
    same `_normalize_sequences_frame` validation/normalization v1 uses.
    """
    print(
        "\n[sequences] Loading model-ready TRAIN+VALIDATION sequence rows "
        "(v2, fail-closed)..."
    )

    files = symbol_parquet_files(root, symbol)

    first_columns = list(
        pq.read_schema(files[0]).names
    )

    missing = sorted(
        set(SEQUENCE_REQUIRED_INPUT_COLUMNS) - set(first_columns)
    )
    if missing:
        raise RuntimeError(
            f"Sequence Parquet missing required columns: {missing}"
        )

    columns = SEQUENCE_REQUIRED_INPUT_COLUMNS + [
        col
        for col in SEQUENCE_OPTIONAL_INPUT_COLUMNS
        if col in first_columns
    ]

    frames: list[pd.DataFrame] = []

    for i, path in enumerate(files, 1):
        dataset = pa_dataset.dataset(
            path,
            format="parquet",
            partitioning=None,
        )

        # STEP 1: structural-only precheck. No target/value column is
        # projected or touched here.
        split_table = dataset.to_table(
            columns=["split"]
        )
        observed_splits = set(
            split_table.column(
                "split"
            ).to_pylist()
        )

        if None in observed_splits:
            raise AssertionError(
                f"{path}: null/invalid split value(s) present — "
                "refusing to read target/value columns."
            )

        disallowed = (
            observed_splits
            - {"train", "validation"}
        )
        if disallowed:
            raise AssertionError(
                f"{path}: v2 sequence input contains disallowed split "
                f"value(s) {sorted(disallowed)} — refusing to read "
                "target/value columns. This artifact is not a valid "
                "model_matrix.py v2 TRAINVAL-only output."
            )

        # STEP 2: only now, an Arrow-filtered read of the full required
        # column set (including target VALUES), filtered inside the
        # scanner.
        table = dataset.to_table(
            columns=columns,
            filter=(
                (pc.field("split") == "train")
                | (
                    pc.field("split")
                    == "validation"
                )
            ),
        )

        frames.append(table.to_pandas())

        if i % 20 == 0 or i == len(files):
            print(
                f"  sequences: loaded {i:>3}/{len(files)} files "
                "(fail-closed precheck PASS on all so far)"
            )

    df = pd.concat(frames, ignore_index=True)

    # STEP 3: shared normalization/validation, identical to v1's.
    return _normalize_sequences_frame(df)


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


def validate_trainval_only(df: pd.DataFrame) -> None:
    """
    v2 defense-in-depth check, independent of `load_sequences_trainval_v2`'s
    own fail-closed read: explicit, fail-closed confirmation that the fully
    labeled output frame contains no split value other than
    {"train", "validation"} immediately before it is written. Do not rely
    only on the upstream reader as an implicit guarantee.
    """
    observed = set(
        df["split"].astype(str).str.lower().unique()
    )

    disallowed = observed - {"train", "validation"}
    if disallowed:
        raise AssertionError(
            f"v2 labeled output contains disallowed split value(s) "
            f"{sorted(disallowed)} immediately before write."
        )

    if (df["split"].astype(str).str.lower() == "test").any():
        raise AssertionError(
            "v2 labeled output contains split == 'test' rows "
            "immediately before write."
        )

    print("TRAIN+VALIDATION-only validation: PASS")


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


def verify_trainval_label_artifact(
    out_root: Path,
    *,
    expected_policy_signature: str,
) -> dict:
    """
    Reusable, persistent verification (importable by Phase 5, not an
    ad-hoc scratch check) for an already-written v2 label-policy artifact.
    Reads back only the structural `split` and `policy_signature` columns
    first — never a target/label value — to prove, fail-closed:

      - zero rows with split == "test";
      - split values are a subset of {"train", "validation"};
      - no duplicate (source_id, prediction_time) keys;
      - the artifact's policy_signature is byte-identical to
        `expected_policy_signature` (policy semantics unchanged).

    Returns a result dict with counts and a `passed` flag; raises
    AssertionError immediately on any TEST row found.
    """
    files = sorted(out_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found under {out_root}"
        )

    total_rows = 0
    test_rows = 0
    disallowed_splits: set[str] = set()
    signatures: set[str] = set()
    key_frames: list[pd.DataFrame] = []

    for path in files:
        dataset = pa_dataset.dataset(
            path,
            format="parquet",
            partitioning=None,
        )

        structural = dataset.to_table(
            columns=[
                "source_id",
                "prediction_time",
                "split",
                "policy_signature",
            ]
        ).to_pandas()

        total_rows += len(structural)

        splits = set(
            structural["split"]
            .astype(str)
            .str.lower()
            .unique()
        )
        test_rows += int(
            (
                structural["split"]
                .astype(str)
                .str.lower()
                == "test"
            ).sum()
        )
        disallowed_splits |= (
            splits - {"train", "validation"}
        )
        signatures |= set(
            structural["policy_signature"].unique()
        )

        key_frames.append(
            structural[
                ["source_id", "prediction_time"]
            ]
        )

    keys = pd.concat(key_frames, ignore_index=True)
    duplicate_keys = int(
        keys.duplicated().sum()
    )

    signature_matches = signatures == {
        expected_policy_signature
    }

    passed = (
        test_rows == 0
        and not disallowed_splits
        and duplicate_keys == 0
        and signature_matches
    )

    if test_rows > 0:
        raise AssertionError(
            f"{out_root}: {test_rows} TEST row(s) found in a v2 "
            "trainval-only label-policy artifact."
        )

    return {
        "total_rows": total_rows,
        "test_rows": test_rows,
        "disallowed_splits": sorted(disallowed_splits),
        "duplicate_keys": duplicate_keys,
        "observed_policy_signatures": sorted(signatures),
        "expected_policy_signature": expected_policy_signature,
        "policy_signature_matches": signature_matches,
        "passed": passed,
    }


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
        "--model-matrix-root",
        default="data/parquet/model_matrix",
        help=(
            "Base model_matrix root (relative to --project-root) whose "
            "sequence-endpoint tree provides the label rows. Default: "
            "data/parquet/model_matrix (unchanged production behavior)."
        ),
    )

    parser.add_argument(
        "--label-policy-root",
        default="data/parquet/label_policy",
        help=(
            "Base output root (relative to --project-root) for the derived "
            "label-policy artifact. Default: data/parquet/label_policy "
            "(unchanged production behavior)."
        ),
    )

    parser.add_argument(
        "--reports-root",
        default="reports/label_policy",
        help=(
            "Base report root (relative to --project-root) for the build "
            "report JSON. Default: reports/label_policy (unchanged "
            "production behavior)."
        ),
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
        "--legacy-v1",
        action="store_true",
        help=(
            "Explicit opt-in to the frozen v1 build path: reads "
            "--model-matrix-root's sequence rows unfiltered (all splits, "
            "including TEST) and computes ATR over the full features_1m "
            "history. Without this flag, the v2 build path runs by "
            "default: --model-matrix-root must point at a "
            "TRAIN+VALIDATION-only sequences/ artifact (e.g. "
            "model_matrix.py v2's output), the sequence reader fails "
            "closed on any TEST/unexpected split value before reading "
            "target/value columns, and the features_1m OHLC read is "
            "bounded to the TRAIN+VALIDATION boundary."
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
    print(f"Legacy v1 mode: {args.legacy_v1}")
    print(f"Feature root: {feature_root}")
    print(f"Sequence root: {seq_root}")
    print(f"Output root: {out_root}")

    if args.write:
        refuse_existing_artifact(out_root)

    # Metadata/signature are frozen before any data is touched, identically
    # for v1 and v2 — policy semantics never change based on artifact scope.
    metadata = make_policy_metadata(
        args,
        symbol,
    )

    signature = policy_signature(
        metadata
    )

    print(f"\nPolicy signature: {signature}")

    if args.legacy_v1:
        print(
            "--legacy-v1 explicitly requested: running the frozen v1 "
            "build path. This reads and can write label VALUES for every "
            "split, including TEST. Do not point a trainer at this "
            "output."
        )

        # 1. Read frozen feature rows and compute causal ATR (full
        #    history, unbounded).
        features = load_feature_ohlc(
            feature_root,
            symbol,
        )
        features = compute_wilder_atr(
            features,
            args.atr_period,
        )

        # 2. Read all frozen model-ready sequence endpoints (unfiltered,
        #    all splits).
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

        labeled = apply_policy(
            joined,
            args,
            signature,
        )

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
    else:
        # v2 (default): sequences FIRST, fail-closed, TRAIN+VALIDATION
        # only. The safe boundary this establishes then bounds the OHLC
        # read — order matters, and is deliberately reversed from v1.

        # 1. Read model-ready TRAIN+VALIDATION sequence endpoints only.
        #    Fails closed before any target/value column is read if the
        #    upstream artifact is not actually TEST-free.
        sequences = load_sequences_trainval_v2(
            seq_root,
            symbol,
        )

        max_trading_date = pd.to_datetime(
            sequences["trading_date"]
        ).max().date()

        print(
            "\nDerived TRAIN+VALIDATION boundary "
            f"(max trading_date): {max_trading_date}"
        )

        # 2. Read causal Wilder ATR only up to that boundary — no
        #    TEST-period OHLC bar is ever read into Python. ATR is
        #    strictly backward-looking, so this produces identical values
        #    for every retained row.
        features = load_feature_ohlc_bounded(
            feature_root,
            symbol,
            max_trading_date=max_trading_date,
        )
        features = compute_wilder_atr(
            features,
            args.atr_period,
        )

        # 3. Exact endpoint join — both inputs are already
        #    TRAIN+VALIDATION-only.
        joined = join_atr(
            features,
            sequences,
            args.atr_period,
        )

        labeled = apply_policy(
            joined,
            args,
            signature,
        )

        # 4. Validate — same generic checks as v1, plus an explicit
        #    fail-closed TRAIN+VALIDATION-only assertion (defense in
        #    depth, not reliant on the reader alone).
        validate_table(
            labeled,
            expected_rows=len(sequences),
            period=args.atr_period,
        )
        validate_trainval_only(
            labeled
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

        # Artifact identity is kept separate from policy identity:
        # policy_signature above is untouched by any of this.
        contract_hash = hashlib.sha256(
            json.dumps(
                {
                    "split_row_counts": report[
                        "split_row_counts"
                    ],
                    "policy_signature": signature,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        report["artifact_scope"] = "trainval"
        report["artifact_contract_version"] = (
            ARTIFACT_CONTRACT_VERSION
        )
        report["artifact_contract_hash"] = contract_hash
        report["generated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

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

    if not args.legacy_v1:
        verification = verify_trainval_label_artifact(
            out_root,
            expected_policy_signature=signature,
        )
        report["required_trainval_verification"] = (
            verification
        )
        print(
            "\nv2 trainval verification: "
            f"{'PASS' if verification['passed'] else 'FAIL'}"
        )
        print(f"  total_rows: {verification['total_rows']}")
        print(f"  test_rows: {verification['test_rows']}")
        print(
            "  policy_signature_matches: "
            f"{verification['policy_signature_matches']}"
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