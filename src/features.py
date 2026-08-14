from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import (
    load_config,
    session_windows_for_trading_date,
)


FEATURE_VERSION = "2.1.0"

# ---------------------------------------------------------------------------
# Feature policy
# ---------------------------------------------------------------------------
#
# 1-minute bars are bar-start labeled:
#     09:30 represents [09:30, 09:31)
#
# Therefore the 09:30 OHLCV row is available at 09:31, and every feature
# using that row is stamped:
#     prediction_time = 09:31
#     feature_available_at = 09:31
#
# No backward filling is performed here.
# Unresolved regularized rows remain unusable and break continuity.
#
# Rolling activity/statistical features reset at:
#     source + trading_date + session + continuity segment
#
# That prevents regular-hours baselines from being contaminated by thin
# premarket/aftermarket activity and prevents rolling calculations from
# jumping across unresolved data.
# ---------------------------------------------------------------------------


DEFAULT_TIMEFRAMES = ("5m", "15m", "30m", "1h")

# Immediate causal return horizons measured in 1-minute bars.
RETURN_WINDOWS_MINUTES = (1, 5, 15, 30, 60)

# Session-local rolling statistics.
VOLATILITY_WINDOWS_MINUTES = (5, 15, 30, 60)
ACTIVITY_WINDOWS_MINUTES = (20, 60)
QUALITY_WINDOWS_MINUTES = (20, 60)
DISTRIBUTION_WINDOW_MINUTES = 20

# Session-local trend windows.
EMA_FAST = 20
EMA_SLOW = 50

# We retain enough previous-month rows so continuity and return calculations
# are valid across a month boundary. The exported continuity count is capped
# at this value on purpose: a future sequence builder can safely interpret
# "512" as "at least 512 consecutive usable minutes".
CONTINUITY_CAP_MINUTES = 512
ONE_MINUTE_CONTEXT_ROWS = CONTINUITY_CAP_MINUTES + 2

# Higher-timeframe context is inexpensive relative to 1-minute data. This tail
# is used only to make the latest completed previous-month bars available to
# the current month.
HIGHER_TIMEFRAME_CONTEXT_ROWS = 512

# Useful audit thresholds for the feature report. These are not hard-coded
# training-window requirements.
CONTINUITY_REPORT_WINDOWS = (30, 60, 120, 240)


REQUIRED_1M_COLUMNS = [
    "datetime",
    "calendar_date",
    "trading_date",
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "wap",
    "bar_count",
    "symbol",
    "source_id",
    "is_observed",
    "is_imputed",
    "quality_status",
]

REQUIRED_RESAMPLED_COLUMNS = [
    "datetime",
    "bar_end",
    "available_at",
    "trading_date",
    "calendar_date",
    "session",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "wap",
    "bar_count",
    "symbol",
    "source_id",
    "expected_minutes",
    "source_grid_minutes",
    "observed_minutes",
    "imputed_minutes",
    "missing_minutes",
    "is_complete",
    "has_imputation",
    "quality_status",
]


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------

def normalize_symbol(value: str) -> str:
    return str(value).strip().lower()


def _parse_bool_series(
    values: pd.Series,
    column: str,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )
    parsed = normalized.map(mapping)

    if parsed.isna().any():
        examples = (
            values.loc[parsed.isna()]
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Invalid boolean values in {column}: {examples}"
        )

    return parsed.astype(bool)


def _ensure_timezone(
    values: pd.Series,
    *,
    timezone: str,
    column: str,
) -> pd.Series:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert(timezone)

    if pd.api.types.is_datetime64_dtype(values.dtype):
        return values.dt.tz_localize(
            timezone,
            ambiguous="raise",
            nonexistent="raise",
        )

    # Strings can contain both DST offsets (-04:00 and -05:00).
    parsed = pd.to_datetime(
        values,
        errors="raise",
        utc=True,
    )
    return parsed.dt.tz_convert(timezone)


def _normalize_dates(
    values: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        values,
        errors="raise",
    ).dt.date


def _numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )


def normalize_regularized(
    df: pd.DataFrame,
    *,
    symbol: str,
    timezone: str,
) -> pd.DataFrame:
    missing = [
        column
        for column in REQUIRED_1M_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Regularized input is missing columns: {missing}"
        )

    work = df.copy()

    work["datetime"] = _ensure_timezone(
        work["datetime"],
        timezone=timezone,
        column="datetime",
    )
    work["calendar_date"] = _normalize_dates(
        work["calendar_date"]
    )
    work["trading_date"] = _normalize_dates(
        work["trading_date"]
    )

    _numeric(
        work,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "wap",
            "bar_count",
        ],
    )

    work["bar_count"] = (
        work["bar_count"]
        .round()
        .astype("Int64")
    )

    work["is_observed"] = _parse_bool_series(
        work["is_observed"],
        "is_observed",
    )
    work["is_imputed"] = _parse_bool_series(
        work["is_imputed"],
        "is_imputed",
    )

    expected_symbol = normalize_symbol(symbol)

    work["symbol"] = (
        work["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["source_id"] = (
        work["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["session"] = (
        work["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["quality_status"] = (
        work["quality_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    symbols = work["symbol"].dropna().unique()
    if len(symbols) != 1 or symbols[0] != expected_symbol:
        raise ValueError(
            f"Expected symbol={expected_symbol!r}; "
            f"found {symbols.tolist()}"
        )

    if work.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            "Duplicate regularized (datetime, source_id) keys exist."
        )

    return (
        work.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def normalize_resampled(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    timezone: str,
) -> pd.DataFrame:
    missing = [
        column
        for column in REQUIRED_RESAMPLED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Resampled {timeframe} input is missing columns: {missing}"
        )

    work = df.copy()

    for column in (
        "datetime",
        "bar_end",
        "available_at",
    ):
        work[column] = _ensure_timezone(
            work[column],
            timezone=timezone,
            column=column,
        )

    work["calendar_date"] = _normalize_dates(
        work["calendar_date"]
    )
    work["trading_date"] = _normalize_dates(
        work["trading_date"]
    )

    _numeric(
        work,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "wap",
            "bar_count",
            "expected_minutes",
            "source_grid_minutes",
            "observed_minutes",
            "imputed_minutes",
            "missing_minutes",
        ],
    )

    work["is_complete"] = _parse_bool_series(
        work["is_complete"],
        "is_complete",
    )
    work["has_imputation"] = _parse_bool_series(
        work["has_imputation"],
        "has_imputation",
    )

    expected_symbol = normalize_symbol(symbol)
    expected_timeframe = str(timeframe).strip().lower()

    work["symbol"] = (
        work["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["source_id"] = (
        work["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["session"] = (
        work["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["timeframe"] = (
        work["timeframe"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    symbols = work["symbol"].dropna().unique()
    if len(symbols) != 1 or symbols[0] != expected_symbol:
        raise ValueError(
            f"Expected symbol={expected_symbol!r}; "
            f"found {symbols.tolist()}"
        )

    timeframes = work["timeframe"].dropna().unique()
    if (
        len(timeframes) != 1
        or timeframes[0] != expected_timeframe
    ):
        raise ValueError(
            f"Expected timeframe={expected_timeframe!r}; "
            f"found {timeframes.tolist()}"
        )

    if work.duplicated(
        ["datetime", "timeframe", "source_id"]
    ).any():
        raise ValueError(
            f"Duplicate resampled {timeframe} keys exist."
        )

    if (
        work["available_at"]
        < work["bar_end"]
    ).any():
        raise ValueError(
            f"{timeframe}: available_at cannot be before bar_end."
        )

    return (
        work.sort_values(
            ["source_id", "available_at"]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Partition discovery / month context
# ---------------------------------------------------------------------------

def discover_monthly_parts(
    root: Path,
    *,
    symbol: str,
) -> dict[tuple[int, int], Path]:
    symbol = normalize_symbol(symbol)
    base = root / f"symbol={symbol}"

    if not base.exists():
        raise FileNotFoundError(
            f"Partition root not found: {base}"
        )

    result: dict[tuple[int, int], Path] = {}

    for path in sorted(
        base.glob("year=*/month=*/part.parquet")
    ):
        try:
            year = int(
                path.parent.parent.name.split("=", 1)[1]
            )
            month = int(
                path.parent.name.split("=", 1)[1]
            )
        except (
            IndexError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid partition path: {path}"
            ) from exc

        key = (year, month)

        if key in result:
            raise ValueError(
                f"Duplicate monthly partition for {key}: "
                f"{result[key]} and {path}"
            )

        result[key] = path

    if not result:
        raise FileNotFoundError(
            f"No monthly Parquet files found under {base}"
        )

    return result


def discover_resampled_parts(
    root: Path,
    *,
    symbol: str,
    timeframe: str,
) -> dict[tuple[int, int], Path]:
    return discover_monthly_parts(
        root / f"timeframe={timeframe}",
        symbol=symbol,
    )


def previous_key(
    keys: list[tuple[int, int]],
    current: tuple[int, int],
) -> tuple[int, int] | None:
    position = keys.index(current)
    if position == 0:
        return None
    return keys[position - 1]


def _tail_per_source(
    df: pd.DataFrame,
    rows: int,
    *,
    sort_column: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    return (
        df.sort_values(
            ["source_id", sort_column]
        )
        .groupby(
            "source_id",
            group_keys=False,
            sort=False,
        )
        .tail(rows)
        .reset_index(drop=True)
    )


def load_regularized_month_with_context(
    *,
    current_path: Path,
    previous_path: Path | None,
    symbol: str,
    timezone: str,
) -> pd.DataFrame:
    current = normalize_regularized(
        pd.read_parquet(
            current_path,
            engine="pyarrow",
        ),
        symbol=symbol,
        timezone=timezone,
    )
    current["_is_current_month"] = True

    frames = []

    if previous_path is not None:
        previous = normalize_regularized(
            pd.read_parquet(
                previous_path,
                engine="pyarrow",
            ),
            symbol=symbol,
            timezone=timezone,
        )
        previous = _tail_per_source(
            previous,
            ONE_MINUTE_CONTEXT_ROWS,
            sort_column="datetime",
        )
        previous["_is_current_month"] = False
        frames.append(previous)

    frames.append(current)

    return (
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def load_resampled_month_with_context(
    *,
    current_path: Path,
    previous_path: Path | None,
    symbol: str,
    timeframe: str,
    timezone: str,
) -> pd.DataFrame:
    current = normalize_resampled(
        pd.read_parquet(
            current_path,
            engine="pyarrow",
        ),
        symbol=symbol,
        timeframe=timeframe,
        timezone=timezone,
    )

    frames = []

    if previous_path is not None:
        previous = normalize_resampled(
            pd.read_parquet(
                previous_path,
                engine="pyarrow",
            ),
            symbol=symbol,
            timeframe=timeframe,
            timezone=timezone,
        )
        previous = _tail_per_source(
            previous,
            HIGHER_TIMEFRAME_CONTEXT_ROWS,
            sort_column="available_at",
        )
        frames.append(previous)

    frames.append(current)

    return (
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            ["source_id", "available_at"]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Generic math helpers
# ---------------------------------------------------------------------------

def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype(float)
    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype(float)

    valid = (
        numerator.notna()
        & denominator.notna()
        & denominator.ne(0)
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )
    result.loc[valid] = (
        numerator.loc[valid]
        / denominator.loc[valid]
    )
    return result


def _safe_ratio_zero_when_both_zero(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide causally while treating the mathematically undefined 0/0 case as
    zero ONLY when both numerator and denominator are known to be exactly zero.

    This is used for candle morphology. For a true flat candle:

        body = 0
        upper shadow = 0
        lower shadow = 0
        range = 0

    the economically meaningful morphology fractions are zero, not missing.
    Missing inputs still remain NaN, and a non-zero numerator divided by a
    zero denominator still remains NaN because that would indicate an invalid
    geometry rather than a flat candle.
    """
    numerator_numeric = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype(float)
    denominator_numeric = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype(float)

    result = _safe_ratio(
        numerator_numeric,
        denominator_numeric,
    )

    known_zero_over_zero = (
        numerator_numeric.notna()
        & denominator_numeric.notna()
        & numerator_numeric.eq(0.0)
        & denominator_numeric.eq(0.0)
    )

    result.loc[
        known_zero_over_zero
    ] = 0.0

    return result


def _safe_bps(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return (
        _safe_ratio(
            numerator,
            denominator,
        )
        * 10_000.0
    )


def _rolling_transform(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
    window: int,
    operation: str,
    shift_periods: int = 0,
) -> pd.Series:
    """
    Session/segment-aware rolling transform.

    operation:
        mean
        std
        median
        skew
        kurt
        q25
        q75
    """

    def apply(values: pd.Series) -> pd.Series:
        if shift_periods:
            values = values.shift(
                shift_periods
            )

        rolling = values.rolling(
            window=window,
            min_periods=window,
        )

        if operation == "mean":
            return rolling.mean()
        if operation == "std":
            return rolling.std(
                ddof=0
            )
        if operation == "median":
            return rolling.median()
        if operation == "skew":
            return rolling.skew()
        if operation == "kurt":
            return rolling.kurt()
        if operation == "q25":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                return rolling.quantile(0.25)

        if operation == "q75":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                return rolling.quantile(0.75)

        raise ValueError(
            f"Unsupported rolling operation: {operation}"
        )

    return (
        frame.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )[value_column]
        .transform(apply)
    )


def _ewm_transform(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
    span: int,
) -> pd.Series:
    return (
        frame.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )[value_column]
        .transform(
            lambda values: (
                values.ewm(
                    span=span,
                    adjust=False,
                    min_periods=span,
                )
                .mean()
            )
        )
    )


# ---------------------------------------------------------------------------
# 1-minute continuity
# ---------------------------------------------------------------------------

def add_continuity_metadata(
    frame: pd.DataFrame,
) -> None:
    """
    Add explicit continuity information for future LSTM sequence generation.

    Unresolved rows break a sequence. A row after an unresolved row begins a
    new usable segment. A physical timestamp jump also begins a new segment.

    The segment id itself is internal. The exported fields are:
        is_current_bar_usable
        is_contiguous_from_previous
        continuity_break_type
        contiguous_history_minutes
    """
    market_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "bar_count",
    ]

    frame["is_current_bar_usable"] = (
        frame[market_columns]
        .notna()
        .all(axis=1)
    )

    previous_datetime = (
        frame.groupby(
            "source_id",
            sort=False,
        )["datetime"]
        .shift(1)
    )

    previous_usable = (
        frame.groupby(
            "source_id",
            sort=False,
        )["is_current_bar_usable"]
        .shift(1)
        .eq(True)
    )

    exact_one_minute = (
        frame["datetime"]
        .sub(previous_datetime)
        .eq(
            timedelta(minutes=1)
        )
    )

    frame[
        "is_contiguous_from_previous"
    ] = (
        previous_datetime.notna()
        & exact_one_minute
        & previous_usable
        & frame["is_current_bar_usable"]
    )

    frame["continuity_break_type"] = np.select(
        [
            previous_datetime.isna(),
            ~frame["is_current_bar_usable"],
            previous_datetime.notna()
            & ~exact_one_minute,
            frame["is_current_bar_usable"]
            & ~previous_usable,
        ],
        [
            "source_start",
            "current_unusable",
            "time_gap",
            "previous_unusable",
        ],
        default="contiguous",
    )

    segment_break = (
        ~frame[
            "is_contiguous_from_previous"
        ]
    )

    frame["_segment_id"] = (
        segment_break.astype("int64")
        .groupby(
            frame["source_id"],
            sort=False,
        )
        .cumsum()
        .astype("int64")
    )

    segment_group = (
        frame.groupby(
            [
                "source_id",
                "_segment_id",
            ],
            sort=False,
            dropna=False,
        )
    )

    continuity = (
        segment_group.cumcount()
        .add(1)
        .where(
            frame[
                "is_current_bar_usable"
            ],
            0,
        )
        .clip(
            upper=CONTINUITY_CAP_MINUTES
        )
        .astype("int32")
    )

    frame[
        "contiguous_history_minutes"
    ] = continuity


# ---------------------------------------------------------------------------
# Session/time features
# ---------------------------------------------------------------------------

def add_session_time_features(
    frame: pd.DataFrame,
    *,
    config,
) -> None:
    one_minute = timedelta(minutes=1)

    window_cache: dict[
        tuple[object, str],
        tuple[pd.Timestamp, pd.Timestamp],
    ] = {}

    unique_pairs = (
        frame[
            ["trading_date", "session"]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    for trading_date, session_name in unique_pairs:
        windows, _ = (
            session_windows_for_trading_date(
                trading_date,
                config,
            )
        )

        if session_name not in windows:
            raise ValueError(
                "No configured session window for "
                f"{trading_date} {session_name!r}"
            )

        window_cache[
            (
                trading_date,
                session_name,
            )
        ] = windows[
            session_name
        ]

    starts = []
    ends = []

    for trading_date, session_name in zip(
        frame["trading_date"],
        frame["session"],
    ):
        start, end = window_cache[
            (
                trading_date,
                session_name,
            )
        ]
        starts.append(start)
        ends.append(end)

    session_start = pd.Series(
        starts,
        index=frame.index,
    )
    session_end = pd.Series(
        ends,
        index=frame.index,
    )

    elapsed = (
        frame["prediction_time"]
        - session_start
    ) / one_minute

    remaining = (
        session_end
        - frame["prediction_time"]
    ) / one_minute

    duration = (
        session_end
        - session_start
    ) / one_minute

    frame[
        "f_session_elapsed_minutes"
    ] = pd.to_numeric(
        elapsed,
        errors="coerce",
    ).astype(float)

    frame[
        "f_session_remaining_minutes"
    ] = pd.to_numeric(
        remaining,
        errors="coerce",
    ).astype(float)

    frame[
        "f_session_progress"
    ] = (
        frame[
            "f_session_elapsed_minutes"
        ]
        / pd.to_numeric(
            duration,
            errors="coerce",
        ).astype(float)
    )

    local_prediction = (
        frame["prediction_time"]
        .dt.tz_convert(
            config.timezone
        )
    )

    minute_of_day = (
        local_prediction.dt.hour * 60
        + local_prediction.dt.minute
    ).astype(float)

    angle = (
        2.0
        * math.pi
        * minute_of_day
        / 1440.0
    )

    frame[
        "f_time_minute_sin"
    ] = np.sin(angle)

    frame[
        "f_time_minute_cos"
    ] = np.cos(angle)

    dow = pd.Series(
        [
            value.weekday()
            for value in frame[
                "trading_date"
            ]
        ],
        index=frame.index,
        dtype="float64",
    )

    dow_angle = (
        2.0
        * math.pi
        * dow
        / 5.0
    )

    frame[
        "f_time_dow_sin"
    ] = np.sin(dow_angle)

    frame[
        "f_time_dow_cos"
    ] = np.cos(dow_angle)

    for session_name in (
        "premarket",
        "regular",
        "aftermarket",
        "overnight",
    ):
        frame[
            f"f_session_is_{session_name}"
        ] = (
            frame["session"]
            .eq(session_name)
            .astype("int8")
        )


# ---------------------------------------------------------------------------
# 1-minute feature families
# ---------------------------------------------------------------------------

def add_candle_features(
    frame: pd.DataFrame,
) -> None:
    body = (
        frame["close"]
        - frame["open"]
    )
    candle_range = (
        frame["high"]
        - frame["low"]
    )

    upper_shadow = (
        frame["high"]
        - frame[
            ["open", "close"]
        ].max(axis=1)
    )

    lower_shadow = (
        frame[
            ["open", "close"]
        ].min(axis=1)
        - frame["low"]
    )

    frame[
        "f_1m_body_bps"
    ] = _safe_bps(
        body,
        frame["open"],
    )

    frame[
        "f_1m_range_bps"
    ] = _safe_bps(
        candle_range,
        frame["close"],
    )

    frame[
        "f_1m_upper_shadow_fraction"
    ] = _safe_ratio_zero_when_both_zero(
        upper_shadow,
        candle_range,
    )

    frame[
        "f_1m_lower_shadow_fraction"
    ] = _safe_ratio_zero_when_both_zero(
        lower_shadow,
        candle_range,
    )

    frame[
        "f_1m_body_to_range"
    ] = _safe_ratio_zero_when_both_zero(
        body.abs(),
        candle_range,
    )

    frame[
        "f_1m_bullish"
    ] = (
        frame["close"]
        > frame["open"]
    ).astype("int8")

    flat = (
        frame["open"].eq(
            frame["high"]
        )
        & frame["open"].eq(
            frame["low"]
        )
        & frame["open"].eq(
            frame["close"]
        )
    )

    frame[
        "f_1m_flat_candle"
    ] = flat.astype("int8")

    frame[
        "f_1m_inactive_flat_candle"
    ] = (
        flat
        & frame["volume"].eq(0)
    ).astype("int8")

    frame[
        "f_1m_strange_flat_candle"
    ] = (
        flat
        & frame["volume"].gt(0)
    ).astype("int8")

    frame[
        "f_1m_close_wap_bps"
    ] = _safe_bps(
        frame["close"]
        - frame["wap"],
        frame["wap"],
    )


def add_return_and_true_range_features(
    frame: pd.DataFrame,
) -> None:
    positive_close = (
        frame["close"]
        .where(
            frame["close"] > 0
        )
    )

    frame["_log_close"] = np.log(
        positive_close
    )

    grouped_log_close = (
        frame.groupby(
            [
                "source_id",
                "_segment_id",
            ],
            sort=False,
            dropna=False,
        )["_log_close"]
    )

    for window in RETURN_WINDOWS_MINUTES:
        frame[
            f"f_1m_log_return_{window}"
        ] = (
            grouped_log_close.diff(
                periods=window
            )
        )

    previous_close = (
        frame.groupby(
            [
                "source_id",
                "_segment_id",
            ],
            sort=False,
            dropna=False,
        )["close"]
        .shift(1)
    )

    high_low = (
        frame["high"]
        - frame["low"]
    )
    high_prev = (
        frame["high"]
        - previous_close
    ).abs()
    low_prev = (
        frame["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_prev,
            low_prev,
        ],
        axis=1,
    ).max(
        axis=1,
        skipna=True,
    )

    # Require a real previous close. The first usable row after a break gets
    # NaN rather than pretending the high-low range is a full true range.
    true_range = true_range.where(
        previous_close.notna()
    )

    frame[
        "f_1m_true_range_bps"
    ] = _safe_bps(
        true_range,
        previous_close,
    )


def add_session_rolling_statistics(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> None:
    for window in VOLATILITY_WINDOWS_MINUTES:
        frame[
            f"f_1m_session_volatility_{window}"
        ] = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column=(
                "f_1m_log_return_1"
            ),
            window=window,
            operation="std",
        )

    window = (
        DISTRIBUTION_WINDOW_MINUTES
    )

    rolling_mean = _rolling_transform(
        frame,
        group_columns=group_columns,
        value_column=(
            "f_1m_log_return_1"
        ),
        window=window,
        operation="mean",
    )
    rolling_std = _rolling_transform(
        frame,
        group_columns=group_columns,
        value_column=(
            "f_1m_log_return_1"
        ),
        window=window,
        operation="std",
    )
    rolling_skew = _rolling_transform(
        frame,
        group_columns=group_columns,
        value_column=(
            "f_1m_log_return_1"
        ),
        window=window,
        operation="skew",
    )
    rolling_kurt = _rolling_transform(
        frame,
        group_columns=group_columns,
        value_column=(
            "f_1m_log_return_1"
        ),
        window=window,
        operation="kurt",
    )
    rolling_q25 = _rolling_transform(
        frame,
        group_columns=group_columns,
        value_column=(
            "f_1m_log_return_1"
        ),
        window=window,
        operation="q25",
    )
    rolling_q75 = _rolling_transform(
        frame,
        group_columns=group_columns,
        value_column=(
            "f_1m_log_return_1"
        ),
        window=window,
        operation="q75",
    )

    frame[
        f"f_1m_session_return_mean_{window}"
    ] = rolling_mean

    frame[
        f"f_1m_session_return_std_{window}"
    ] = rolling_std

    frame[
        f"f_1m_session_return_skew_{window}"
    ] = rolling_skew

    frame[
        f"f_1m_session_return_kurt_{window}"
    ] = rolling_kurt

    frame[
        f"f_1m_session_return_iqr_{window}"
    ] = (
        rolling_q75
        - rolling_q25
    )

    frame[
        f"f_1m_session_return_z_{window}"
    ] = _safe_ratio(
        frame[
            "f_1m_log_return_1"
        ]
        - rolling_mean,
        rolling_std,
    )


def add_session_activity_features(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> None:
    frame["_bar_count_float"] = (
        frame["bar_count"]
        .astype("Float64")
        .astype(float)
    )

    frame["_dollar_volume"] = (
        frame["close"]
        * frame["volume"]
    )

    frame[
        "f_1m_log_volume"
    ] = np.log1p(
        frame["volume"]
        .clip(lower=0)
    )

    frame[
        "f_1m_log_bar_count"
    ] = np.log1p(
        frame[
            "_bar_count_float"
        ]
        .clip(lower=0)
    )

    frame[
        "f_1m_log_dollar_volume"
    ] = np.log1p(
        frame[
            "_dollar_volume"
        ]
        .clip(lower=0)
    )

    for window in ACTIVITY_WINDOWS_MINUTES:
        volume_mean = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column="volume",
            window=window,
            operation="mean",
            shift_periods=1,
        )
        volume_std = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column="volume",
            window=window,
            operation="std",
            shift_periods=1,
        )

        count_mean = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column="_bar_count_float",
            window=window,
            operation="mean",
            shift_periods=1,
        )
        count_std = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column="_bar_count_float",
            window=window,
            operation="std",
            shift_periods=1,
        )

        dollar_mean = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column="_dollar_volume",
            window=window,
            operation="mean",
            shift_periods=1,
        )
        dollar_std = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column="_dollar_volume",
            window=window,
            operation="std",
            shift_periods=1,
        )

        frame[
            f"f_1m_session_relative_volume_{window}"
        ] = _safe_ratio(
            frame["volume"],
            volume_mean,
        )

        frame[
            f"f_1m_session_volume_z_{window}"
        ] = _safe_ratio(
            frame["volume"]
            - volume_mean,
            volume_std,
        )

        frame[
            f"f_1m_session_relative_bar_count_{window}"
        ] = _safe_ratio(
            frame[
                "_bar_count_float"
            ],
            count_mean,
        )

        frame[
            f"f_1m_session_bar_count_z_{window}"
        ] = _safe_ratio(
            frame[
                "_bar_count_float"
            ]
            - count_mean,
            count_std,
        )

        frame[
            f"f_1m_session_relative_dollar_volume_{window}"
        ] = _safe_ratio(
            frame[
                "_dollar_volume"
            ],
            dollar_mean,
        )

        frame[
            f"f_1m_session_dollar_volume_z_{window}"
        ] = _safe_ratio(
            frame[
                "_dollar_volume"
            ]
            - dollar_mean,
            dollar_std,
        )


def add_session_ema_features(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> None:
    ema_fast = _ewm_transform(
        frame,
        group_columns=group_columns,
        value_column="close",
        span=EMA_FAST,
    )

    ema_slow = _ewm_transform(
        frame,
        group_columns=group_columns,
        value_column="close",
        span=EMA_SLOW,
    )

    frame[
        f"f_1m_session_price_vs_ema{EMA_FAST}_bps"
    ] = _safe_bps(
        frame["close"]
        - ema_fast,
        ema_fast,
    )

    frame[
        f"f_1m_session_price_vs_ema{EMA_SLOW}_bps"
    ] = _safe_bps(
        frame["close"]
        - ema_slow,
        ema_slow,
    )

    frame[
        f"f_1m_session_ema{EMA_FAST}_vs_ema{EMA_SLOW}_bps"
    ] = _safe_bps(
        ema_fast
        - ema_slow,
        ema_slow,
    )

    ema_fast_previous = (
        ema_fast.groupby(
            [
                frame[column]
                for column in group_columns
            ],
            sort=False,
        )
        .shift(1)
    )

    ema_slow_previous = (
        ema_slow.groupby(
            [
                frame[column]
                for column in group_columns
            ],
            sort=False,
        )
        .shift(1)
    )

    frame[
        f"f_1m_session_ema{EMA_FAST}_slope_bps"
    ] = _safe_bps(
        ema_fast
        - ema_fast_previous,
        ema_fast_previous,
    )

    frame[
        f"f_1m_session_ema{EMA_SLOW}_slope_bps"
    ] = _safe_bps(
        ema_slow
        - ema_slow_previous,
        ema_slow_previous,
    )


def add_quality_features(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> None:
    frame[
        "_imputed_numeric"
    ] = (
        frame["is_imputed"]
        .astype("float64")
    )

    frame[
        "_observed_numeric"
    ] = (
        frame["is_observed"]
        .astype("float64")
    )

    for window in QUALITY_WINDOWS_MINUTES:
        frame[
            f"f_quality_session_imputed_fraction_{window}"
        ] = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column=(
                "_imputed_numeric"
            ),
            window=window,
            operation="mean",
        )

        frame[
            f"f_quality_session_observed_fraction_{window}"
        ] = _rolling_transform(
            frame,
            group_columns=group_columns,
            value_column=(
                "_observed_numeric"
            ),
            window=window,
            operation="mean",
        )


def build_1m_features(
    df: pd.DataFrame,
    *,
    config,
) -> pd.DataFrame:
    """
    Build causal 1-minute features.

    Continuity-sensitive features never cross an unresolved row.
    Activity/statistical/trend rolling features additionally reset at session
    boundaries so premarket, regular, and aftermarket regimes are not mixed.
    """
    work = (
        df.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
        .copy()
    )

    work[
        "prediction_time"
    ] = (
        work["datetime"]
        + timedelta(minutes=1)
    )

    work[
        "feature_available_at"
    ] = work[
        "prediction_time"
    ]

    add_continuity_metadata(
        work
    )

    # A session rolling group is both session-aware and missing-data-aware.
    # _segment_id is the global usable-continuity segment; trading_date/session
    # then prevents session-regime mixing.
    session_group_columns = [
        "source_id",
        "trading_date",
        "session",
        "_segment_id",
    ]

    add_candle_features(
        work
    )

    add_return_and_true_range_features(
        work
    )

    add_session_rolling_statistics(
        work,
        group_columns=(
            session_group_columns
        ),
    )

    add_session_activity_features(
        work,
        group_columns=(
            session_group_columns
        ),
    )

    add_session_ema_features(
        work,
        group_columns=(
            session_group_columns
        ),
    )

    add_quality_features(
        work,
        group_columns=(
            session_group_columns
        ),
    )

    add_session_time_features(
        work,
        config=config,
    )

    if (
        work[
            "feature_available_at"
        ]
        > work[
            "prediction_time"
        ]
    ).any():
        raise AssertionError(
            "1-minute feature availability violation."
        )

    return work


# ---------------------------------------------------------------------------
# Higher-timeframe completed-bar features
# ---------------------------------------------------------------------------

def _add_tf_continuity(
    frame: pd.DataFrame,
) -> None:
    """
    Build a higher-timeframe continuity segment.

    Two complete bars are contiguous only when, within the same
    source/trading_date/session, current bar_start == previous bar_end.
    Therefore removing an incomplete bar cannot silently stitch returns across
    the missing derived interval.
    """
    group = frame.groupby(
        [
            "source_id",
            "trading_date",
            "session",
        ],
        sort=False,
        dropna=False,
    )

    previous_end = (
        group["bar_end"]
        .shift(1)
    )

    contiguous = (
        previous_end.notna()
        & frame["datetime"].eq(
            previous_end
        )
    )

    break_mask = (
        ~contiguous
    )

    frame[
        "_tf_segment_id"
    ] = (
        break_mask.astype("int64")
        .groupby(
            [
                frame["source_id"],
                frame["trading_date"],
                frame["session"],
            ],
            sort=False,
        )
        .cumsum()
        .astype("int64")
    )


def build_higher_timeframe_features(
    df: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    """
    Build context from COMPLETE higher-timeframe bars only.

    Returns/volatility/true-range calculations do not jump across an
    incomplete higher-timeframe bar.
    """
    tf = str(
        timeframe
    ).strip().lower()

    work = (
        df.loc[
            df["is_complete"]
            & df["close"].notna()
        ]
        .sort_values(
            [
                "source_id",
                "trading_date",
                "session",
                "datetime",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    if work.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                f"ctx_{tf}_available_at",
            ]
        )

    _add_tf_continuity(
        work
    )

    group_columns = [
        "source_id",
        "trading_date",
        "session",
        "_tf_segment_id",
    ]

    work[
        "_log_close"
    ] = np.log(
        work["close"]
        .where(
            work["close"] > 0
        )
    )

    grouped_log_close = (
        work.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )["_log_close"]
    )

    work[
        f"f_{tf}_log_return_1"
    ] = grouped_log_close.diff(1)

    work[
        f"f_{tf}_log_return_3"
    ] = grouped_log_close.diff(3)

    work[
        f"f_{tf}_volatility_3"
    ] = (
        work.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )[
            f"f_{tf}_log_return_1"
        ]
        .transform(
            lambda values: (
                values.rolling(
                    window=3,
                    min_periods=3,
                )
                .std(ddof=0)
            )
        )
    )

    body = (
        work["close"]
        - work["open"]
    )
    candle_range = (
        work["high"]
        - work["low"]
    )

    upper_shadow = (
        work["high"]
        - work[
            ["open", "close"]
        ].max(axis=1)
    )

    lower_shadow = (
        work[
            ["open", "close"]
        ].min(axis=1)
        - work["low"]
    )

    work[
        f"f_{tf}_range_bps"
    ] = _safe_bps(
        candle_range,
        work["close"],
    )

    work[
        f"f_{tf}_body_bps"
    ] = _safe_bps(
        body,
        work["open"],
    )

    work[
        f"f_{tf}_upper_shadow_fraction"
    ] = _safe_ratio_zero_when_both_zero(
        upper_shadow,
        candle_range,
    )

    work[
        f"f_{tf}_lower_shadow_fraction"
    ] = _safe_ratio_zero_when_both_zero(
        lower_shadow,
        candle_range,
    )

    work[
        f"f_{tf}_body_to_range"
    ] = _safe_ratio_zero_when_both_zero(
        body.abs(),
        candle_range,
    )

    previous_close = (
        work.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            candle_range,
            (
                work["high"]
                - previous_close
            ).abs(),
            (
                work["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1,
        skipna=True,
    )

    true_range = true_range.where(
        previous_close.notna()
    )

    work[
        f"f_{tf}_true_range_bps"
    ] = _safe_bps(
        true_range,
        previous_close,
    )

    work[
        f"f_{tf}_close_wap_bps"
    ] = _safe_bps(
        work["close"]
        - work["wap"],
        work["wap"],
    )

    work[
        f"f_{tf}_log_volume"
    ] = np.log1p(
        work["volume"]
        .clip(lower=0)
    )

    work[
        f"f_{tf}_log_bar_count"
    ] = np.log1p(
        work["bar_count"]
        .astype("Float64")
        .astype(float)
        .clip(lower=0)
    )

    work[
        f"f_{tf}_has_imputation"
    ] = (
        work[
            "has_imputation"
        ]
        .astype("int8")
    )

    work[
        f"f_{tf}_expected_minutes"
    ] = (
        work[
            "expected_minutes"
        ]
        .astype(float)
    )

    keep = [
        "source_id",
        "available_at",
        "trading_date",
        "session",
        "close",
        f"f_{tf}_log_return_1",
        f"f_{tf}_log_return_3",
        f"f_{tf}_volatility_3",
        f"f_{tf}_range_bps",
        f"f_{tf}_body_bps",
        f"f_{tf}_upper_shadow_fraction",
        f"f_{tf}_lower_shadow_fraction",
        f"f_{tf}_body_to_range",
        f"f_{tf}_true_range_bps",
        f"f_{tf}_close_wap_bps",
        f"f_{tf}_log_volume",
        f"f_{tf}_log_bar_count",
        f"f_{tf}_has_imputation",
        f"f_{tf}_expected_minutes",
    ]

    result = (
        work[keep]
        .copy()
        .rename(
            columns={
                "available_at": (
                    f"ctx_{tf}_available_at"
                ),
                "trading_date": (
                    f"_ctx_{tf}_trading_date"
                ),
                "session": (
                    f"_ctx_{tf}_session"
                ),
                "close": (
                    f"_ctx_{tf}_close"
                ),
            }
        )
    )

    return (
        result.sort_values(
            [
                "source_id",
                f"ctx_{tf}_available_at",
            ]
        )
        .reset_index(drop=True)
    )


def merge_higher_timeframe(
    base: pd.DataFrame,
    context: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    tf = str(
        timeframe
    ).strip().lower()

    available_column = (
        f"ctx_{tf}_available_at"
    )

    right_columns = [
        column
        for column in context.columns
        if column != "source_id"
    ]

    pieces = []

    for (
        source_id,
        left_source,
    ) in base.groupby(
        "source_id",
        sort=False,
        dropna=False,
    ):
        left = (
            left_source
            .sort_values(
                "prediction_time"
            )
            .copy()
        )

        right = (
            context.loc[
                context[
                    "source_id"
                ].eq(source_id)
            ]
            .drop(
                columns=[
                    "source_id"
                ],
                errors="ignore",
            )
            .sort_values(
                available_column
            )
            .copy()
        )

        if right.empty:
            merged = left.copy()

            for column in right_columns:
                if (
                    column
                    not in merged.columns
                ):
                    merged[
                        column
                    ] = np.nan
        else:
            merged = pd.merge_asof(
                left,
                right,
                left_on=(
                    "prediction_time"
                ),
                right_on=(
                    available_column
                ),
                direction="backward",
                allow_exact_matches=True,
            )

        pieces.append(
            merged
        )

    result = (
        pd.concat(
            pieces,
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            "_row_order"
        )
        .reset_index(drop=True)
    )

    available = (
        result[
            available_column
        ]
    )

    leakage = (
        available.notna()
        & (
            available
            > result[
                "prediction_time"
            ]
        )
    )

    if leakage.any():
        examples = (
            result.loc[
                leakage,
                [
                    "prediction_time",
                    available_column,
                ],
            ]
            .head(5)
            .to_dict(
                orient="records"
            )
        )
        raise AssertionError(
            f"{tf} future leakage detected: "
            f"{examples}"
        )

    age = (
        result[
            "prediction_time"
        ]
        - available
    ) / timedelta(
        minutes=1
    )

    result[
        f"f_{tf}_age_minutes"
    ] = pd.to_numeric(
        age,
        errors="coerce",
    ).astype(float)

    context_date_column = (
        f"_ctx_{tf}_trading_date"
    )
    context_session_column = (
        f"_ctx_{tf}_session"
    )
    context_close_column = (
        f"_ctx_{tf}_close"
    )

    valid_context = (
        available.notna()
    )

    result[
        f"f_{tf}_same_trading_date"
    ] = np.where(
        valid_context,
        (
            result[
                context_date_column
            ]
            == result[
                "trading_date"
            ]
        ).astype(float),
        np.nan,
    )

    result[
        f"f_{tf}_same_session"
    ] = np.where(
        valid_context,
        (
            result[
                context_session_column
            ]
            == result[
                "session"
            ]
        ).astype(float),
        np.nan,
    )

    result[
        f"f_{tf}_close_anchor_bps"
    ] = _safe_bps(
        result["close"]
        - result[
            context_close_column
        ],
        result[
            context_close_column
        ],
    )

    return result.drop(
        columns=[
            context_date_column,
            context_session_column,
            context_close_column,
        ],
        errors="ignore",
    )


# ---------------------------------------------------------------------------
# Output schema / IO
# ---------------------------------------------------------------------------

def order_output_columns(
    frame: pd.DataFrame,
    *,
    timeframes: Iterable[str],
) -> pd.DataFrame:
    metadata = [
        "datetime",
        "prediction_time",
        "feature_available_at",
        "calendar_date",
        "trading_date",
        "session",
        "symbol",
        "source_id",
        "is_observed",
        "is_imputed",
        "quality_status",
        "is_current_bar_usable",
        "is_contiguous_from_previous",
        "continuity_break_type",
        "contiguous_history_minutes",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "bar_count",
    ]

    if (
        "imputation_source_datetime"
        in frame.columns
    ):
        metadata.append(
            "imputation_source_datetime"
        )

    context_columns = [
        f"ctx_{str(tf).strip().lower()}_available_at"
        for tf in timeframes
        if (
            f"ctx_{str(tf).strip().lower()}_available_at"
            in frame.columns
        )
    ]

    feature_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("f_")
    )

    ordered = [
        column
        for column in (
            metadata
            + context_columns
            + feature_columns
        )
        if column in frame.columns
    ]

    return frame[
        ordered
    ].copy()


def atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    compression: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}."
        f"{os.getpid()}.tmp"
    )

    try:
        frame.to_parquet(
            temp_path,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}."
        f"{os.getpid()}.tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------------------------
# Symbol processor
# ---------------------------------------------------------------------------

def _continuity_counts(
    output: pd.DataFrame,
) -> dict[str, int]:
    counts = {}

    for window in (
        CONTINUITY_REPORT_WINDOWS
    ):
        counts[
            f"at_least_{window}_minutes"
        ] = int(
            (
                output[
                    "is_current_bar_usable"
                ]
                & (
                    output[
                        "contiguous_history_minutes"
                    ]
                    >= window
                )
            ).sum()
        )

    return counts


def _feature_availability_counts(
    output: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, int]]:
    """
    Count finite/NaN/inf feature values on usable market rows only.

    Warm-up/context NaNs remain visible in the report. Zero-range candle
    morphology no longer contributes structural NaNs because v2.1 encodes
    known flat-candle 0/0 morphology as zero.
    """
    usable = output.loc[
        output["is_current_bar_usable"]
    ]

    result: dict[
        str,
        dict[str, int],
    ] = {}

    for column in feature_columns:
        values = pd.to_numeric(
            usable[column],
            errors="coerce",
        ).to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        result[column] = {
            "finite_rows": int(
                np.isfinite(values).sum()
            ),
            "nan_rows": int(
                np.isnan(values).sum()
            ),
            "positive_inf_rows": int(
                np.isposinf(values).sum()
            ),
            "negative_inf_rows": int(
                np.isneginf(values).sum()
            ),
        }

    return result


def process_symbol(
    *,
    regularized_root: Path,
    resampled_root: Path,
    output_root: Path,
    symbol: str,
    timeframes: list[str],
    config,
    reports_root: Path | None,
) -> dict:
    symbol = normalize_symbol(
        symbol
    )

    timeframes = [
        str(value)
        .strip()
        .lower()
        for value in timeframes
    ]

    regularized_parts = (
        discover_monthly_parts(
            regularized_root,
            symbol=symbol,
        )
    )

    regularized_keys = sorted(
        regularized_parts
    )

    resampled_parts = {
        timeframe: (
            discover_resampled_parts(
                resampled_root,
                symbol=symbol,
                timeframe=timeframe,
            )
        )
        for timeframe in timeframes
    }

    missing_resampled = {
        timeframe: [
            key
            for key in regularized_keys
            if (
                key
                not in resampled_parts[
                    timeframe
                ]
            )
        ]
        for timeframe in timeframes
    }

    missing_resampled = {
        timeframe: values
        for (
            timeframe,
            values,
        ) in missing_resampled.items()
        if values
    }

    if missing_resampled:
        raise FileNotFoundError(
            "Missing resampled monthly "
            "partitions: "
            f"{missing_resampled}"
        )

    total_rows = 0
    total_usable = 0
    total_observed = 0
    total_imputed = 0
    written_files = []
    month_summaries = []

    total_continuity = {
        f"at_least_{window}_minutes": 0
        for window in (
            CONTINUITY_REPORT_WINDOWS
        )
    }

    feature_count = None

    aggregate_feature_availability: dict[
        str,
        dict[str, int],
    ] = {}

    total_zero_range_usable = 0
    total_zero_volume_usable = 0
    total_positive_volume_usable = 0

    for key in regularized_keys:
        year, month = key

        prior_key = previous_key(
            regularized_keys,
            key,
        )

        prior_regularized = (
            regularized_parts[
                prior_key
            ]
            if (
                prior_key
                is not None
            )
            else None
        )

        one_minute = (
            load_regularized_month_with_context(
                current_path=(
                    regularized_parts[
                        key
                    ]
                ),
                previous_path=(
                    prior_regularized
                ),
                symbol=symbol,
                timezone=(
                    config.timezone
                ),
            )
        )

        base = build_1m_features(
            one_minute,
            config=config,
        )

        base = (
            base.loc[
                base[
                    "_is_current_month"
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

        base[
            "_row_order"
        ] = np.arange(
            len(base),
            dtype="int64",
        )

        for timeframe in timeframes:
            tf_keys = sorted(
                resampled_parts[
                    timeframe
                ]
            )

            tf_prior_key = (
                previous_key(
                    tf_keys,
                    key,
                )
            )

            tf_prior_path = (
                resampled_parts[
                    timeframe
                ][tf_prior_key]
                if (
                    tf_prior_key
                    is not None
                )
                else None
            )

            raw_context = (
                load_resampled_month_with_context(
                    current_path=(
                        resampled_parts[
                            timeframe
                        ][key]
                    ),
                    previous_path=(
                        tf_prior_path
                    ),
                    symbol=symbol,
                    timeframe=timeframe,
                    timezone=(
                        config.timezone
                    ),
                )
            )

            context = (
                build_higher_timeframe_features(
                    raw_context,
                    timeframe=timeframe,
                )
            )

            base = merge_higher_timeframe(
                base,
                context,
                timeframe=timeframe,
            )

        base = base.drop(
            columns=[
                "_row_order",
                "_segment_id",
                "_log_close",
                "_bar_count_float",
                "_dollar_volume",
                "_imputed_numeric",
                "_observed_numeric",
                "_is_current_month",
            ],
            errors="ignore",
        )

        output = order_output_columns(
            base,
            timeframes=timeframes,
        )

        if output.duplicated(
            ["datetime", "source_id"]
        ).any():
            raise ValueError(
                f"{year:04d}-{month:02d}: "
                "duplicate feature row keys."
            )

        if (
            output[
                "feature_available_at"
            ]
            > output[
                "prediction_time"
            ]
        ).any():
            raise AssertionError(
                f"{year:04d}-{month:02d}: "
                "feature availability violation."
            )

        output_path = (
            output_root
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )

        atomic_write_parquet(
            output,
            output_path,
            compression=(
                config.parquet_compression
            ),
        )

        feature_columns = [
            column
            for column in output.columns
            if column.startswith("f_")
        ]

        if feature_count is None:
            feature_count = len(
                feature_columns
            )
        elif feature_count != len(
            feature_columns
        ):
            raise AssertionError(
                "Feature column count changed "
                "between months."
            )

        usable = int(
            output[
                "is_current_bar_usable"
            ].sum()
        )

        observed = int(
            output[
                "is_observed"
            ].sum()
        )

        imputed = int(
            output[
                "is_imputed"
            ].sum()
        )

        continuity = (
            _continuity_counts(
                output
            )
        )

        for name, value in (
            continuity.items()
        ):
            total_continuity[
                name
            ] += value

        usable_mask = output[
            "is_current_bar_usable"
        ]

        zero_range_usable = int(
            (
                usable_mask
                & (
                    output["high"]
                    - output["low"]
                ).eq(0)
            ).sum()
        )

        zero_volume_usable = int(
            (
                usable_mask
                & output["volume"].eq(0)
            ).sum()
        )

        positive_volume_usable = int(
            (
                usable_mask
                & output["volume"].gt(0)
            ).sum()
        )

        total_zero_range_usable += (
            zero_range_usable
        )
        total_zero_volume_usable += (
            zero_volume_usable
        )
        total_positive_volume_usable += (
            positive_volume_usable
        )

        month_feature_availability = (
            _feature_availability_counts(
                output,
                feature_columns,
            )
        )

        for (
            feature_name,
            counts,
        ) in (
            month_feature_availability.items()
        ):
            aggregate = (
                aggregate_feature_availability
                .setdefault(
                    feature_name,
                    {
                        "finite_rows": 0,
                        "nan_rows": 0,
                        "positive_inf_rows": 0,
                        "negative_inf_rows": 0,
                    },
                )
            )

            for (
                count_name,
                count_value,
            ) in counts.items():
                aggregate[
                    count_name
                ] += int(
                    count_value
                )

        month_summary = {
            "year": year,
            "month": month,
            "rows": int(
                len(output)
            ),
            "usable_rows": usable,
            "observed_rows": observed,
            "imputed_rows": imputed,
            "feature_columns": len(
                feature_columns
            ),
            "zero_range_usable_rows": (
                zero_range_usable
            ),
            "zero_volume_usable_rows": (
                zero_volume_usable
            ),
            "positive_volume_usable_rows": (
                positive_volume_usable
            ),
            "continuity": continuity,
            "output_file": str(
                output_path
            ),
        }

        month_summaries.append(
            month_summary
        )

        written_files.append(
            output_path
        )

        total_rows += len(output)
        total_usable += usable
        total_observed += observed
        total_imputed += imputed

        print(
            f"[{year:04d}-{month:02d}] "
            f"rows={len(output):,}, "
            f"usable={usable:,}, "
            f"features={len(feature_columns):,}, "
            f"contig60="
            f"{continuity['at_least_60_minutes']:,}"
        )

    feature_availability = {}

    for (
        feature_name,
        counts,
    ) in sorted(
        aggregate_feature_availability.items()
    ):
        finite_rows = int(
            counts["finite_rows"]
        )
        nan_rows = int(
            counts["nan_rows"]
        )
        positive_inf_rows = int(
            counts["positive_inf_rows"]
        )
        negative_inf_rows = int(
            counts["negative_inf_rows"]
        )

        denominator = (
            finite_rows
            + nan_rows
            + positive_inf_rows
            + negative_inf_rows
        )

        feature_availability[
            feature_name
        ] = {
            **counts,
            "finite_percent": (
                round(
                    100.0
                    * finite_rows
                    / denominator,
                    6,
                )
                if denominator
                else None
            ),
        }

    summary = {
        "feature_version": (
            FEATURE_VERSION
        ),
        "symbol": symbol,
        "timeframes": (
            timeframes
        ),
        "months": len(
            regularized_keys
        ),
        "rows": int(
            total_rows
        ),
        "usable_rows": int(
            total_usable
        ),
        "observed_rows": int(
            total_observed
        ),
        "imputed_rows": int(
            total_imputed
        ),
        "feature_columns": int(
            feature_count or 0
        ),
        "zero_range_morphology_policy": (
            "When candle range=0 and the morphology numerator "
            "is also exactly 0, upper/lower shadow fractions "
            "and body-to-range are encoded as 0. Missing inputs "
            "and inconsistent nonzero/0 geometry remain NaN."
        ),
        "zero_range_usable_rows": int(
            total_zero_range_usable
        ),
        "zero_volume_usable_rows": int(
            total_zero_volume_usable
        ),
        "positive_volume_usable_rows": int(
            total_positive_volume_usable
        ),
        "feature_availability_on_usable_rows": (
            feature_availability
        ),
        "continuity_cap_minutes": (
            CONTINUITY_CAP_MINUTES
        ),
        "continuity_eligible_rows": (
            total_continuity
        ),
        "output_root": str(
            output_root
        ),
        "written_files": len(
            written_files
        ),
        "month_summaries": (
            month_summaries
        ),
    }

    if reports_root is not None:
        report_path = (
            reports_root
            / symbol
            / (
                "feature_summary_"
                f"v{FEATURE_VERSION}.json"
            )
        )

        atomic_write_json(
            summary,
            report_path,
        )

        summary[
            "report_file"
        ] = str(
            report_path
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build causal 1-minute ML features from "
            "regularized 1-minute data plus completed "
            "higher-timeframe bars."
        )
    )

    parser.add_argument(
        "--regularized-root",
        type=Path,
        required=True,
        help=(
            "Root containing monthly regularized "
            "1-minute Parquet partitions."
        ),
    )

    parser.add_argument(
        "--resampled-root",
        type=Path,
        required=True,
        help=(
            "Root containing timeframe=5m/15m/"
            "30m/1h resampled partitions."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Output root for monthly 1-minute "
            "feature Parquet."
        ),
    )

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(
            DEFAULT_TIMEFRAMES
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/pipeline.yaml"
        ),
    )

    parser.add_argument(
        "--reports",
        type=Path,
        default=None,
        help=(
            "Optional feature report root."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(
        args.config
    )

    summary = process_symbol(
        regularized_root=(
            args.regularized_root
        ),
        resampled_root=(
            args.resampled_root
        ),
        output_root=(
            args.output_root
        ),
        symbol=args.symbol,
        timeframes=args.timeframes,
        config=config,
        reports_root=args.reports,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())