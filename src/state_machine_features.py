"""
Milestone 1 Phase 2 — causal ACD/state-machine feature layer.

Reads one representative ACD run (state_trace_1min, context_1min,
environment_daily, regime_context_1min for one explicit regime policy),
left-joins it onto the market-ml `features_1m` timestamp universe, applies
availability gating, and writes:

  1. a standalone `state_machine_features_1m` layer (ACD-derived columns
     only, plus provenance and availability flags), and
  2. a combined `features_1m_acd_v1` layer (existing `features_1m` columns
     + the ACD-derived columns), consumable by the existing
     `sequences.py`/`model_matrix.py` builders via their existing
     `--features-root`/`--feature-set`/`--features` CLI overrides.

`core_v1`/`features_1m` are never modified. ACD row availability never
removes a market-ml base row (left join only); it only gates feature
*values* via `sm_available`/`sm_env_available`.

Hard exclusions (never read, never used as features): signal_outcomes,
regime_evaluation (any policy), daily_acd_summary, event-level logs
(regime_evidence_log/regime_transition_log/continuation_signal_log), the
frozen-threshold category fields (or_width_class, buffer_width_class,
spacing_class, environment_id, or_width_category), raw OHLCV duplicates,
and minutes_since_open (redundant with market-ml's own session features).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from common import load_config
from features import atomic_write_parquet

STATE_MACHINE_FEATURE_SCHEMA_VERSION = "1.0.0"
COMBINED_FEATURE_SET_ACD_V1 = "core_v1_acd_v1"
COMMON_MODELING_START = pd.Timestamp("2020-01-03").date()
OHLCV_MISMATCH_TOLERANCE = 1e-6
SENTINEL_NUMERIC = 0.0
SENTINEL_CATEGORICAL = "unavailable"

# Columns read from market-ml's own features_1m to form the left/base
# universe. OHLCV columns are read only for source-identity validation,
# never re-emitted as state-machine features (redundant with features_1m).
BASE_COLUMNS = (
    "datetime",
    "prediction_time",
    "feature_available_at",
    "trading_date",
    "session",
    "symbol",
    "source_id",
    "is_current_bar_usable",
    "is_observed",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# --- ACD-derived whitelist, grouped by source table and availability gate ---
# state_trace_1min — gated by sm_available (opening_range_finalized &
# zone_available), except OHLCV (read for validation only, not emitted).
STATE_TRACE_COLUMNS = (
    "state",
    "prev_state",
    "state_changed",
    "price_zone",
    "prev_price_zone",
)

# context_1min — gated by sm_available (also requires acd_available).
CONTEXT_COLUMNS = (
    "acd_phase",
    "location_state",
    "acd_regime",
    "acd_regime_age",
    "active_signal_type",
    "active_signal_level",
    "active_signal_direction",
    "active_signal_status",
    "active_signal_age",
    "minutes_since_last_event",
    "active_signal_count",
    "active_confirmed_signal_count",
    "active_pending_signal_count",
    "active_bullish_signal_count",
    "active_bearish_signal_count",
    # acd_today_* / rolling reliability fields: Phase 1 verification
    # explicitly, individually tested only these four (monotonic-within-day
    # for the two acd_today_* counters; constant-within-day/changes-across-
    # days for acd_reliability_10d[+_n]) — see verify-state-machine-
    # integration-contract report. Every other field in these two families
    # shares the naming pattern but was NOT individually verified, and per
    # "do not infer safety from field names" is deferred, not whitelisted,
    # for this first Phase-2 build.
    "acd_today_confirmed_count",
    "acd_today_touch_count",
    "acd_reliability_10d",
    "acd_reliability_10d_n",
    "or_width_atr",
    "a_gap_atr",
    "c_gap_atr",
    "a_gap_to_or",
    "c_gap_to_or",
    "total_envelope_atr",
)

# environment_daily — gated separately by sm_env_available
# (prediction_time >= or_finalized_at, same session_date only, atr_valid,
# or_width_valid). *_class / environment_id are hard-excluded (frozen
# quantile threshold fit through 2022-12-30 uses future information
# relative to earlier rows; see Phase 1 verification report).
ENVIRONMENT_COLUMNS = (
    "or_width_atr",
    "a_up_gap_atr",
    "a_down_gap_atr",
    "c_up_spacing_atr",
    "c_down_spacing_atr",
    "total_up_buffer_atr",
    "total_down_buffer_atr",
    "up_spacing_balance",
    "down_spacing_balance",
    "total_envelope_atr",
)
# Renamed on the environment side only, to avoid colliding with the
# same-named context_1min columns above (both groups are kept distinct
# features: env_* are session-grain, broadcast; the context ones are
# already 1-minute native).
ENVIRONMENT_COLUMN_RENAME = {
    column: f"env_{column}" for column in ENVIRONMENT_COLUMNS
}

# regime_context_1min (one explicit policy) — gated by sm_available.
REGIME_COLUMNS = (
    "regime_id",
    "direction",
    "phase",
    "regime_age_bars",
    "current_retest_depth",
    "max_retest_depth",
    "regime_strength_score",
    "regime_evidence_ambiguous",
    "opposite_or_touched",
    "opposite_or_close_streak",
    "opposite_a_confirmed",
    "opposite_c_confirmed",
    "same_direction_confirmation_count",
    "same_direction_reinforcement_count",
    "same_direction_reentry_count",
    "minutes_since_last_same_direction_confirmation",
    "effective_or_cancel_bars",
)
REGIME_COLUMN_RENAME = {
    "direction": "regime_direction",
    "phase": "regime_phase",
}

REGIME_OUTPUT_COLUMNS = {
    REGIME_COLUMN_RENAME.get(c, c) for c in REGIME_COLUMNS
}

# Regime-derived columns are gated by sm_regime_available (a regime row was
# actually found for this timestamp/policy), never by sm_available alone —
# regime_context_1min's own row existence is the authoritative signal, kept
# separate from the state/context availability flags.
BOOLEAN_GROUP_COLUMNS = {
    "sm_available": {
        "state_changed",
    },
    "sm_regime_available": {
        "opposite_or_touched", "opposite_a_confirmed", "opposite_c_confirmed",
        "regime_evidence_ambiguous",
    },
}
CATEGORICAL_GROUP_COLUMNS = {
    "sm_available": {
        "state", "prev_state",
        "price_zone", "prev_price_zone",
        "acd_phase", "location_state", "acd_regime",
        "active_signal_type", "active_signal_level",
        "active_signal_direction", "active_signal_status",
    },
    "sm_regime_available": {
        "regime_id", "regime_direction", "regime_phase",
    },
}
NUMERIC_GROUP_COLUMNS = {
    "sm_available": (
        set(STATE_TRACE_COLUMNS) | set(CONTEXT_COLUMNS)
    )
    - CATEGORICAL_GROUP_COLUMNS["sm_available"]
    - BOOLEAN_GROUP_COLUMNS["sm_available"],
    "sm_regime_available": (
        REGIME_OUTPUT_COLUMNS
        - CATEGORICAL_GROUP_COLUMNS["sm_regime_available"]
        - BOOLEAN_GROUP_COLUMNS["sm_regime_available"]
    ),
    "sm_env_available": set(ENVIRONMENT_COLUMN_RENAME.values()),
}

# All 14 nominal-typed ACD string columns. model_matrix.py requires every
# selected feature to be float64-castable (feature_row_finite_mask), so
# none of these can be passed to it directly. All 14 must be dropped from
# any features tree consumed by model_matrix.py.
ALL_CATEGORICAL_COLUMNS = tuple(sorted(
    CATEGORICAL_GROUP_COLUMNS["sm_available"]
    | CATEGORICAL_GROUP_COLUMNS["sm_regime_available"]
))
# regime_id is a per-episode identifier (10,541 distinct TRAIN values,
# confirmed empirically), not a bounded nominal category, and is excluded
# from one-hot encoding — it is dropped from the ACD feature set entirely
# rather than encoded. The other 13 are genuine bounded categories (1-8
# distinct TRAIN values each, confirmed empirically) and are one-hot
# encoded normally.
ONE_HOT_CATEGORICAL_COLUMNS = tuple(
    c for c in ALL_CATEGORICAL_COLUMNS if c != "regime_id"
)
UNAVAILABLE_INDICATOR = "unavailable"
UNSEEN_INDICATOR = "unseen"

# model_matrix.py's resolve_matrix_settings() rejects any selected feature
# name that doesn't start with "f_" (model_matrix.py:452-460) — a naming
# convention already satisfied by every core_v1 feature, but not by any
# ACD-derived column. Rather than modify model_matrix.py, every ACD
# feature column materialized into features_1m_acd_v1_encoded (both the
# pre-existing numeric/boolean columns and the one-hot indicator columns)
# is prefixed with this before being written.
ACD_FEATURE_COLUMN_PREFIX = "f_sm_"

# The 46 non-categorical ACD numeric/boolean columns that need the f_
# prefix (everything ACD-derived except the 14 raw categorical strings,
# which are dropped/replaced by one-hot columns instead of renamed).
ACD_NUMERIC_BOOLEAN_COLUMNS = tuple(sorted(
    NUMERIC_GROUP_COLUMNS["sm_available"]
    | NUMERIC_GROUP_COLUMNS["sm_regime_available"]
    | NUMERIC_GROUP_COLUMNS["sm_env_available"]
    | BOOLEAN_GROUP_COLUMNS["sm_available"]
    | BOOLEAN_GROUP_COLUMNS["sm_regime_available"]
    | {"sm_available", "sm_env_available", "sm_regime_available"}
))


def _acd_feature_name(raw_name: str) -> str:
    return f"{ACD_FEATURE_COLUMN_PREFIX}{raw_name}"


def _enforce_common_modeling_start(start_date):
    """common_modeling_start (2020-01-03) is a hard floor, not merely a
    default — a caller-supplied start_date earlier than this is clamped
    up, never honored as-is."""

    if start_date is None:
        return COMMON_MODELING_START
    return max(start_date, COMMON_MODELING_START)


def _read_parquet_table(path: Path, columns: list[str] | None = None, *,
                         attempts: int = 3, retry_delay_seconds: float = 1.0):
    """Read a Parquet file via the low-level ParquetFile reader (not
    pyarrow's dataset factory: some existing market-ml files have
    row-group-level dictionary-encoding drift on partition columns that
    trips the dataset factory's schema-merge step, but read cleanly this
    way), with a short retry for transient cloud-storage read failures
    (observed on this Google-Drive-backed filesystem under sustained
    sequential reads across many files — not a data-corruption finding)."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return pq.ParquetFile(path).read(columns=columns)
        except OSError as error:
            last_error = error
            if attempt < attempts:
                time.sleep(retry_delay_seconds)
    raise last_error


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from a single Parquet file as a DataFrame."""

    return _read_parquet_table(path, columns).to_pandas()


def _read_features_1m(
    features_root: Path,
    symbol: str,
    *,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    pattern = str(
        features_root / f"symbol={symbol}" / "year=*" / "month=*" / "*.parquet"
    )
    files = sorted(glob.glob(pattern))

    frames = []
    for file_path in files:
        frame = _read_parquet_columns(Path(file_path), list(BASE_COLUMNS))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No features_1m partitions found under {features_root} "
            f"for symbol={symbol}."
        )

    base = pd.concat(frames, ignore_index=True)

    if start_date is not None:
        base = base[base["trading_date"] >= start_date]
    if end_date is not None:
        base = base[base["trading_date"] <= end_date]

    return base.reset_index(drop=True)


def _read_acd_table(
    acd_run_root: Path,
    relative_path: str,
    columns: list[str],
    *,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> pd.DataFrame:
    path = acd_run_root / relative_path
    frame = _read_parquet_columns(path, columns)

    if filter_column is not None:
        frame = frame[frame[filter_column] == filter_value].reset_index(
            drop=True
        )

    return frame


def _to_market_ml_timestamp(series: pd.Series, timezone: str) -> pd.Series:
    return series.dt.tz_convert(timezone)


@dataclass(frozen=True)
class SourceIdentityReport:
    acd_rows: int
    base_rows_in_range: int
    matched_rows: int
    acd_duplicate_keys: int
    base_duplicate_keys: int
    mismatch_counts: dict[str, int]
    max_abs_diff: dict[str, float]

    @property
    def unmatched_acd_rows(self) -> int:
        return self.acd_rows - self.matched_rows

    def raise_if_unsafe(self) -> None:
        if self.acd_duplicate_keys or self.base_duplicate_keys:
            raise AssertionError(
                "state-machine source-identity check failed: duplicate "
                f"keys (acd={self.acd_duplicate_keys}, "
                f"base={self.base_duplicate_keys})."
            )

        if self.unmatched_acd_rows:
            raise AssertionError(
                "state-machine source-identity check failed: "
                f"{self.unmatched_acd_rows} ACD rows have no matching "
                "market-ml base row."
            )

        bad = {
            column: count
            for column, count in self.mismatch_counts.items()
            if count
        }
        if bad:
            raise AssertionError(
                "state-machine source-identity check failed: OHLCV "
                f"mismatches against market-ml on matched rows: {bad}"
            )


def validate_source_identity(
    state_trace: pd.DataFrame,
    base: pd.DataFrame,
) -> SourceIdentityReport:
    """Compare ACD state_trace_1min OHLCV against market-ml features_1m
    OHLCV on matched (symbol, datetime) timestamps. Normalization is not a
    substitute for this check — it compares raw values directly."""

    merged = state_trace.merge(
        base,
        on="datetime",
        how="inner",
        suffixes=("_acd", "_mml"),
    )

    mismatch_counts: dict[str, int] = {}
    max_abs_diff: dict[str, float] = {}

    for column in ("open", "high", "low", "close", "volume"):
        acd_values = merged[f"{column}_acd"].astype(float)
        mml_values = merged[f"{column}_mml"].astype(float)
        diff = (acd_values - mml_values).abs()
        mismatch_counts[column] = int(
            (diff > OHLCV_MISMATCH_TOLERANCE).sum()
        )
        max_abs_diff[column] = float(diff.max()) if len(diff) else 0.0

    return SourceIdentityReport(
        acd_rows=len(state_trace),
        base_rows_in_range=len(base),
        matched_rows=len(merged),
        acd_duplicate_keys=int(state_trace.duplicated(["datetime"]).sum()),
        base_duplicate_keys=int(base.duplicated(["datetime"]).sum()),
        mismatch_counts=mismatch_counts,
        max_abs_diff=max_abs_diff,
    )


def build_state_machine_features_1m(
    *,
    features_root: Path,
    acd_root: Path,
    output_root: Path,
    symbol: str,
    config_id: str,
    regime_config_id: str,
    regime_policy_name: str,
    regime_policy_version: str,
    timezone: str,
    parquet_compression: str,
    start_date=None,
    end_date=None,
) -> list[Path]:
    start_date = _enforce_common_modeling_start(start_date)

    base = _read_features_1m(
        features_root, symbol, start_date=start_date, end_date=end_date
    )

    acd_run_root = acd_root / symbol / config_id

    state_trace = _read_acd_table(
        acd_run_root,
        "state/state_trace_1min.parquet",
        [
            "timestamp", "symbol", "config_id",
            "open", "high", "low", "close", "volume",
            "opening_range_finalized", "zone_available",
            *STATE_TRACE_COLUMNS,
        ],
        filter_column="config_id",
        filter_value=config_id,
    )
    state_trace["datetime"] = _to_market_ml_timestamp(
        state_trace["timestamp"], timezone
    )

    # Identity is only meaningful over the range actually being built —
    # `base` may be narrower than ACD's own history (the default
    # COMMON_MODELING_START trims ~2019-12-23..2020-01-02 that ACD has but
    # base does not; test callers may narrow further). Rows outside
    # base's own range are expected to be "unmatched" and are not a
    # safety violation; rows inside the range must match exactly.
    in_range_state_trace = state_trace[
        state_trace["datetime"].between(
            base["datetime"].min(), base["datetime"].max()
        )
    ]
    identity_report = validate_source_identity(in_range_state_trace, base)
    identity_report.raise_if_unsafe()

    context = _read_acd_table(
        acd_run_root,
        "context/context_1min.parquet",
        ["timestamp", "symbol", "config_id", "acd_available", *CONTEXT_COLUMNS],
        filter_column="config_id",
        filter_value=config_id,
    )
    context["datetime"] = _to_market_ml_timestamp(context["timestamp"], timezone)

    environment = _read_acd_table(
        acd_run_root,
        "environment/environment_daily.parquet",
        [
            "symbol", "session_date", "signal_config_id",
            "or_finalized_at", "atr_valid", "or_width_valid",
            "environment_schema_version", "category_method",
            "category_threshold_set_id",
            *ENVIRONMENT_COLUMNS,
        ],
        filter_column="signal_config_id",
        filter_value=config_id,
    )
    environment["or_finalized_at"] = _to_market_ml_timestamp(
        environment["or_finalized_at"], timezone
    )
    environment = environment.rename(columns=ENVIRONMENT_COLUMN_RENAME)

    regime_relative_path = (
        f"regimes/{regime_config_id}/regime_context_1min.parquet"
    )
    regime = _read_acd_table(
        acd_run_root,
        regime_relative_path,
        [
            "timestamp", "symbol", "signal_config_id", "regime_config_id",
            "policy_name", "policy_version",
            *REGIME_COLUMNS,
        ],
        filter_column="regime_config_id",
        filter_value=regime_config_id,
    )
    regime["datetime"] = _to_market_ml_timestamp(regime["timestamp"], timezone)
    regime = regime.rename(columns=REGIME_COLUMN_RENAME)

    # --- left join: market-ml base universe is authoritative ---
    # Strict join keys throughout: symbol + datetime for the 1-minute
    # sources, symbol + trading_date/session_date for the daily source.
    # validate= enforces the expected cardinality and raises loudly on any
    # unexpected duplicate key rather than silently fanning out rows.
    merged = base[
        ["datetime", "prediction_time", "feature_available_at",
         "trading_date", "session", "symbol", "source_id"]
    ].copy()

    merged = merged.merge(
        state_trace[
            ["symbol", "datetime", "opening_range_finalized", "zone_available",
             *STATE_TRACE_COLUMNS]
        ],
        on=["symbol", "datetime"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        context[["symbol", "datetime", "acd_available", *CONTEXT_COLUMNS]],
        on=["symbol", "datetime"],
        how="left",
        validate="one_to_one",
    )

    regime_columns_to_merge = [
        "symbol", "datetime", *REGIME_COLUMN_RENAME.values(),
        *[c for c in REGIME_COLUMNS if c not in REGIME_COLUMN_RENAME],
    ]
    merged = merged.merge(
        regime[regime_columns_to_merge],
        on=["symbol", "datetime"],
        how="left",
        validate="one_to_one",
        indicator="_regime_merge_indicator",
    )
    merged["sm_regime_available"] = (
        merged["_regime_merge_indicator"] == "both"
    )
    merged = merged.drop(columns=["_regime_merge_indicator"])

    # environment_daily: same symbol + trading_date/session_date only, then
    # availability gate on or_finalized_at (never carry forward from a
    # prior session). many_to_one: many 1-minute base rows per one daily
    # environment row.
    merged = merged.merge(
        environment[["symbol", "session_date", "or_finalized_at", "atr_valid",
                     "or_width_valid", *ENVIRONMENT_COLUMN_RENAME.values()]],
        left_on=["symbol", "trading_date"],
        right_on=["symbol", "session_date"],
        how="left",
        validate="many_to_one",
    )

    merged["sm_available"] = (
        merged["opening_range_finalized"].fillna(False)
        & merged["zone_available"].fillna(False)
        & merged["acd_available"].fillna(False)
    )
    merged["sm_env_available"] = (
        (merged["prediction_time"] >= merged["or_finalized_at"])
        & merged["atr_valid"].fillna(False)
        & merged["or_width_valid"].fillna(False)
    ).fillna(False)

    for flag_column, columns in NUMERIC_GROUP_COLUMNS.items():
        mask = ~merged[flag_column]
        for column in columns:
            if column in merged.columns:
                merged.loc[mask, column] = SENTINEL_NUMERIC
                merged[column] = merged[column].fillna(SENTINEL_NUMERIC)

    for flag_column, columns in BOOLEAN_GROUP_COLUMNS.items():
        mask = ~merged[flag_column]
        for column in columns:
            if column in merged.columns:
                merged[column] = merged[column].astype("boolean")
                merged.loc[mask, column] = False
                merged[column] = merged[column].fillna(False)

    for flag_column, columns in CATEGORICAL_GROUP_COLUMNS.items():
        mask = ~merged[flag_column]
        for column in columns:
            if column in merged.columns:
                merged.loc[mask, column] = SENTINEL_CATEGORICAL
                merged[column] = merged[column].fillna(SENTINEL_CATEGORICAL)

    merged = merged.drop(
        columns=[
            "opening_range_finalized", "zone_available", "acd_available",
            "session_date", "or_finalized_at", "atr_valid", "or_width_valid",
        ]
    )

    # --- provenance ---
    merged["signal_config_id"] = config_id
    merged["regime_config_id"] = regime_config_id
    merged["regime_policy_name"] = regime_policy_name
    merged["regime_policy_version"] = regime_policy_version
    merged["environment_schema_version"] = (
        environment["environment_schema_version"].iloc[0]
        if len(environment) else None
    )
    merged["category_method"] = (
        environment["category_method"].iloc[0] if len(environment) else None
    )
    merged["category_threshold_set_id"] = (
        environment["category_threshold_set_id"].iloc[0]
        if len(environment) else None
    )
    merged["state_machine_feature_schema_version"] = (
        STATE_MACHINE_FEATURE_SCHEMA_VERSION
    )
    merged["market_ml_feature_set_identity"] = COMBINED_FEATURE_SET_ACD_V1
    merged["build_timestamp"] = pd.Timestamp.now(tz="UTC")

    if merged.duplicated(["datetime", "source_id"]).any():
        raise AssertionError(
            "state_machine_features_1m: duplicate (datetime, source_id) "
            "keys after join."
        )

    if (merged["feature_available_at"] > merged["prediction_time"]).any():
        raise AssertionError(
            "state_machine_features_1m: feature availability violation "
            "(feature_available_at > prediction_time)."
        )

    written: list[Path] = []
    merged["_year"] = merged["datetime"].dt.year
    merged["_month"] = merged["datetime"].dt.month

    for (year, month), group in merged.groupby(["_year", "_month"], sort=True):
        output_path = (
            output_root
            / f"config_id={config_id}"
            / f"regime_config_id={regime_config_id}"
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )
        atomic_write_parquet(
            group.drop(columns=["_year", "_month"]),
            output_path,
            compression=parquet_compression,
        )
        written.append(output_path)

    return written


def build_combined_feature_table(
    *,
    features_root: Path,
    state_machine_root: Path,
    output_root: Path,
    symbol: str,
    config_id: str,
    regime_config_id: str,
    parquet_compression: str,
    start_date=None,
    end_date=None,
) -> list[Path]:
    """Left-join state_machine_features_1m onto features_1m, producing a
    schema-superset combined table restricted to
    trading_date >= common_modeling_start (2020-01-03 by default) —
    matching state_machine_features_1m's own universe. features_1m itself
    is never read-modified; this only writes a new tree. No pre-common-
    start partition is created."""

    start_date = _enforce_common_modeling_start(start_date)

    sm_pattern = str(
        state_machine_root
        / f"config_id={config_id}"
        / f"regime_config_id={regime_config_id}"
        / f"symbol={symbol}"
        / "year=*"
        / "month=*"
        / "*.parquet"
    )
    sm_files = sorted(glob.glob(sm_pattern))
    if not sm_files:
        raise FileNotFoundError(
            f"No state_machine_features_1m partitions found matching "
            f"{sm_pattern}. Run build_state_machine_features_1m first."
        )

    sm_frame = pd.concat(
        [_read_parquet_table(f).to_pandas() for f in sm_files],
        ignore_index=True,
    )
    sm_join_columns = [
        c for c in sm_frame.columns
        if c not in ("prediction_time", "feature_available_at",
                     "trading_date", "session", "source_id")
    ]

    features_pattern = str(
        features_root / f"symbol={symbol}" / "year=*" / "month=*" / "*.parquet"
    )
    feature_files = sorted(glob.glob(features_pattern))
    if not feature_files:
        raise FileNotFoundError(
            f"No features_1m partitions found matching {features_pattern}."
        )

    written: list[Path] = []
    for file_path in feature_files:
        features_frame = _read_parquet_table(Path(file_path)).to_pandas()

        features_frame = features_frame[
            features_frame["trading_date"] >= start_date
        ]
        if end_date is not None:
            features_frame = features_frame[
                features_frame["trading_date"] <= end_date
            ]

        if len(features_frame) == 0:
            # Entirely before common_modeling_start (or after end_date) —
            # no combined partition is written for this month at all.
            continue

        combined = features_frame.merge(
            sm_frame[sm_join_columns],
            on=["symbol", "datetime"],
            how="left",
            validate="one_to_one",
        )

        if len(combined) != len(features_frame):
            raise AssertionError(
                f"{file_path}: combined row count {len(combined)} != "
                f"filtered base features_1m row count {len(features_frame)} "
                "(row-universe invariance violated)."
            )

        year = int(Path(file_path).parts[-3].split("=")[1])
        month = int(Path(file_path).parts[-2].split("=")[1])
        output_path = (
            output_root
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )
        atomic_write_parquet(
            combined, output_path, compression=parquet_compression
        )
        written.append(output_path)

    return written


def _merge_intervals(pairs: list[tuple]) -> list[list]:
    """Merge (start, end) timestamp pairs into non-overlapping, sorted
    intervals. Adjacent/overlapping pairs collapse into one interval —
    used to turn ~1M individual TRAIN sequence windows (heavily
    overlapping at stride=1) into a small number of covering ranges
    without ever materializing every window's 120 rows."""

    if not pairs:
        return []

    ordered = sorted(pairs)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def _select_rows_in_intervals(
    frame: pd.DataFrame, intervals: list[list]
) -> pd.DataFrame:
    mask = pd.Series(False, index=frame.index)
    for start, end in intervals:
        mask |= frame["datetime"].between(start, end)
    return frame[mask]


def fit_categorical_vocabularies(
    *,
    sequence_index_root: Path,
    features_root: Path,
    symbol: str,
) -> dict[str, list[str]]:
    """Fit one TRAIN-only vocabulary per categorical column, sourced from
    every feature row actually covered by a TRAIN-split sequence's
    [window_start_datetime, window_end_datetime] — not just each
    sequence's endpoint row — per the approved Phase 3 correction."""

    sequence_files = sorted(
        glob.glob(str(sequence_index_root / "**" / "*.parquet"), recursive=True)
    )
    if not sequence_files:
        raise FileNotFoundError(
            f"No sequence_index partitions found under {sequence_index_root}."
        )

    train_windows = []
    for file_path in sequence_files:
        frame = _read_parquet_columns(
            Path(file_path),
            ["split", "window_start_datetime", "window_end_datetime"],
        )
        train_frame = frame[frame["split"] == "train"]
        train_windows.extend(
            zip(
                train_frame["window_start_datetime"],
                train_frame["window_end_datetime"],
            )
        )

    if not train_windows:
        raise ValueError("No TRAIN-split sequences found; cannot fit vocabulary.")

    merged_intervals = _merge_intervals(train_windows)

    feature_files = sorted(
        glob.glob(str(features_root / f"symbol={symbol}" / "year=*" / "month=*" / "*.parquet"))
    )
    vocabularies: dict[str, set] = {c: set() for c in ONE_HOT_CATEGORICAL_COLUMNS}

    for file_path in feature_files:
        frame = _read_parquet_columns(
            Path(file_path), ["datetime", *ONE_HOT_CATEGORICAL_COLUMNS]
        )
        train_rows = _select_rows_in_intervals(frame, merged_intervals)
        if train_rows.empty:
            continue
        for column in ONE_HOT_CATEGORICAL_COLUMNS:
            values = set(train_rows[column].unique()) - {UNAVAILABLE_INDICATOR}
            vocabularies[column] |= values

    return {
        column: sorted(values) for column, values in vocabularies.items()
    }


def build_categorical_encoding_manifest(
    vocabularies: dict[str, list[str]],
) -> dict:
    """Deterministic ordering per categorical: <col>_unavailable,
    <col>_unseen, then one <col>_<value> column per sorted TRAIN value."""

    manifest = {}
    for column in ONE_HOT_CATEGORICAL_COLUMNS:
        values = vocabularies[column]
        # encoded_columns holds the FINAL materialized (f_sm_-prefixed)
        # column names — the manifest is the source of truth for what's
        # actually written and what model_matrix.py's --features must name.
        encoded_columns = [
            _acd_feature_name(f"{column}_{UNAVAILABLE_INDICATOR}"),
            _acd_feature_name(f"{column}_{UNSEEN_INDICATOR}"),
        ] + [_acd_feature_name(f"{column}_{value}") for value in values]
        manifest[column] = {
            "reserved": [UNAVAILABLE_INDICATOR, UNSEEN_INDICATOR],
            "train_vocabulary": values,
            "encoded_columns": encoded_columns,
        }
    return manifest


def _one_hot_encode_column(
    series: pd.Series, column: str, manifest_entry: dict
) -> pd.DataFrame:
    encoded_columns = manifest_entry["encoded_columns"]
    train_vocabulary = manifest_entry["train_vocabulary"]

    out = pd.DataFrame(
        0, index=series.index, columns=encoded_columns, dtype="int8"
    )
    is_unavailable = series == UNAVAILABLE_INDICATOR
    is_known = series.isin(train_vocabulary)
    is_unseen = ~is_unavailable & ~is_known

    out.loc[is_unavailable, _acd_feature_name(f"{column}_{UNAVAILABLE_INDICATOR}")] = 1
    out.loc[is_unseen, _acd_feature_name(f"{column}_{UNSEEN_INDICATOR}")] = 1
    for value in train_vocabulary:
        out.loc[series == value, _acd_feature_name(f"{column}_{value}")] = 1

    # Exactly one indicator per row.
    assert (out.sum(axis=1) == 1).all(), (
        f"{column}: one-hot encoding did not produce exactly one "
        "indicator for every row."
    )
    return out


def apply_categorical_encoding(
    *,
    features_root: Path,
    output_root: Path,
    manifest: dict,
    symbol: str,
    parquet_compression: str,
) -> list[Path]:
    """Apply the frozen TRAIN vocabulary/manifest to every row (train,
    validation, and test alike) of features_root, replacing the 14 raw
    categorical string columns with their one-hot indicator columns.
    features_root itself is not modified — this writes a new tree."""

    feature_files = sorted(
        glob.glob(str(features_root / f"symbol={symbol}" / "year=*" / "month=*" / "*.parquet"))
    )
    if not feature_files:
        raise FileNotFoundError(f"No partitions found under {features_root}.")

    written: list[Path] = []
    for file_path in feature_files:
        frame = _read_parquet_table(Path(file_path)).to_pandas()

        # Drop all 14 raw categorical string columns (model_matrix.py can't
        # consume strings); regime_id is dropped without replacement
        # (per-episode identifier, not a bounded category — see
        # ONE_HOT_CATEGORICAL_COLUMNS), the other 13 are replaced by their
        # one-hot indicator blocks. The 46 remaining ACD numeric/boolean
        # columns are renamed with the f_ prefix model_matrix.py requires.
        base = frame.drop(columns=list(ALL_CATEGORICAL_COLUMNS)).rename(
            columns={
                column: _acd_feature_name(column)
                for column in ACD_NUMERIC_BOOLEAN_COLUMNS
            }
        )
        encoded_blocks = [base]
        for column in ONE_HOT_CATEGORICAL_COLUMNS:
            encoded_blocks.append(
                _one_hot_encode_column(frame[column], column, manifest[column])
            )
        encoded = pd.concat(encoded_blocks, axis=1)

        relative = Path(file_path).relative_to(features_root)
        output_path = output_root / relative
        atomic_write_parquet(
            encoded, output_path, compression=parquet_compression
        )
        written.append(output_path)

    return written


def _load_state_machine_config_section(config_path: Path) -> dict:
    """Read pipeline.state_machine_features directly (independent of
    common.py's PipelineConfig, which does not expose this new section)."""

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return (raw.get("pipeline") or {}).get("state_machine_features") or {}


def _coalesce_arg(cli_value, section: dict, key: str, default=None):
    if cli_value is not None:
        return cli_value
    return section.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the causal ACD/state-machine feature layer and the "
            "combined core_v1_acd_v1 feature table for market-ml."
        )
    )
    parser.add_argument("--mode", choices=["state-machine", "combine", "all"],
                         default="all")
    parser.add_argument("--symbol", default="nvda")
    # None defaults preserve CLI > pipeline.yaml (state_machine_features
    # section) > error-if-still-missing precedence, resolved in main().
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--regime-config-id", default=None)
    parser.add_argument("--regime-policy-name", default=None)
    parser.add_argument("--regime-policy-version", default=None)
    parser.add_argument("--acd-root", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--state-machine-output-root", type=Path,
                         default=Path("data/parquet/state_machine_features_1m"))
    parser.add_argument("--combined-output-root", type=Path,
                         default=Path("data/parquet/features_1m_acd_v1"))
    parser.add_argument("--config", type=Path,
                         default=Path("config/pipeline.yaml"))
    parser.add_argument("--show-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_config = load_config(args.config)
    section = _load_state_machine_config_section(args.config)

    config_id = _coalesce_arg(args.config_id, section, "config_id")
    regime_config_id = _coalesce_arg(
        args.regime_config_id, section, "regime_config_id"
    )
    regime_policy_name = _coalesce_arg(
        args.regime_policy_name, section, "regime_policy_name"
    )
    regime_policy_version = _coalesce_arg(
        args.regime_policy_version, section, "regime_policy_version"
    )
    acd_root = _coalesce_arg(args.acd_root, section, "acd_root")
    acd_root = Path(acd_root) if acd_root else Path.home() / "acd_experiments_local"

    missing = [
        name for name, value in (
            ("config_id", config_id),
            ("regime_config_id", regime_config_id),
            ("regime_policy_name", regime_policy_name),
            ("regime_policy_version", regime_policy_version),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required value(s) not found via CLI or "
            f"pipeline.state_machine_features: {', '.join(missing)}"
        )

    if args.show_config:
        print(f"mode={args.mode}")
        print(f"symbol={args.symbol}")
        print(f"config_id={config_id}")
        print(f"regime_config_id={regime_config_id}")
        print(f"regime_policy_name={regime_policy_name}")
        print(f"regime_policy_version={regime_policy_version}")
        print(f"features_root={args.features_root}")
        print(f"acd_root={acd_root}")
        print(f"state_machine_output_root={args.state_machine_output_root}")
        print(f"combined_output_root={args.combined_output_root}")
        return 0

    if args.mode in ("state-machine", "all"):
        build_state_machine_features_1m(
            features_root=args.features_root,
            acd_root=acd_root,
            output_root=args.state_machine_output_root,
            symbol=args.symbol,
            config_id=config_id,
            regime_config_id=regime_config_id,
            regime_policy_name=regime_policy_name,
            regime_policy_version=regime_policy_version,
            timezone=pipeline_config.timezone,
            parquet_compression=pipeline_config.parquet_compression,
        )

    if args.mode in ("combine", "all"):
        build_combined_feature_table(
            features_root=args.features_root,
            state_machine_root=args.state_machine_output_root,
            output_root=args.combined_output_root,
            symbol=args.symbol,
            config_id=config_id,
            regime_config_id=regime_config_id,
            parquet_compression=pipeline_config.parquet_compression,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
