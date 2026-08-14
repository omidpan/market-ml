"""
Validation V2 for source OHLCV/IBKR historical bars.

Goals
-----
1. Validate source rows without modifying them.
2. Build expected 1-minute coverage from the exchange calendar.
3. Respect exchange early closes when defining regular/aftermarket sessions.
4. Treat source zero-volume / zero-bar-count rows as observed rows, not missing.
5. Validate optional IBKR WAP and bar_count fields.
6. Keep gap classification session-local: leading/internal/trailing/full_session.
7. Write compact audit reports for downstream review.

Designed for the current market_ml project, but intentionally self-contained.

Typical use
-----------
python src/validate_v2.py \
  --input data/raw/bootstrap/AVGO/avgo_1min_extended.csv \
  --symbol AVGO \
  --source-id primary_extended

For the controlled one-year audit:
python src/validate_v2.py \
  --input data/raw/audit/avgo_2025_2026/semiconductor/avgo_1min_extended.csv \
  --symbol AVGO \
  --source-id primary_extended
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import yaml


CORE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
MICROSTRUCTURE_COLUMNS = ["wap", "bar_count"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_CALENDAR = "XNYS"
DEFAULT_FREQUENCY = "1min"
DEFAULT_BAR_LABEL = "start"

DEFAULT_SESSIONS = {
    "premarket": {
        "start": "04:00",
        "end_exclusive": "09:30",
        "crosses_midnight": False,
    },
    "regular": {
        "start": "09:30",
        "end_exclusive": "16:00",
        "crosses_midnight": False,
    },
    "aftermarket": {
        "start": "16:00",
        "end_exclusive": "20:00",
        "crosses_midnight": False,
    },
    "overnight": {
        "start": "20:00",
        "end_exclusive": "04:00",
        "crosses_midnight": True,
    },
}

DEFAULT_COVERED_SESSIONS = ["premarket", "regular", "aftermarket"]

# IBKR WAP can carry half-cent precision while displayed OHLC values are
# rounded to cents. Preserve source WAP exactly and reject only deviations
# beyond this source-precision tolerance.
WAP_OHLC_TOLERANCE = 0.005
FLOAT_EPSILON = 1e-12


@dataclass(frozen=True)
class ValidationConfig:
    timezone: str
    calendar: str
    frequency: str
    bar_label: str
    sessions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    symbol: str
    coverage_start: pd.Timestamp | None
    coverage_end: pd.Timestamp | None
    coverage_sessions: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate source OHLCV bars without modifying source data."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--source-id", default="primary_extended")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Pipeline YAML. Defaults are used if the file does not exist.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("config/sources.csv"),
        help="Source metadata CSV. Defaults are used if the file does not exist.",
    )
    parser.add_argument(
        "--instruments",
        type=Path,
        default=Path("config/instruments.csv"),
        help="Instrument metadata CSV; used for calendar override if available.",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports/validation"),
    )
    parser.add_argument(
        "--calendar",
        default=None,
        help="Optional exchange-calendar override, e.g. XNYS.",
    )
    parser.add_argument(
        "--timezone",
        default=None,
        help="Optional timezone override, e.g. America/New_York.",
    )
    parser.add_argument(
        "--frequency",
        default=None,
        help="Optional bar frequency override; V2 currently expects 1min.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_clock(value: str) -> time:
    return pd.Timestamp(str(value)).time()


def _clock_on_date(day: pd.Timestamp, clock_text: str, timezone: str) -> pd.Timestamp:
    base = pd.Timestamp(day).tz_localize(None).normalize()
    clock = _parse_clock(clock_text)
    naive = base + pd.Timedelta(
        hours=clock.hour,
        minutes=clock.minute,
        seconds=clock.second,
    )
    return naive.tz_localize(timezone)


def _as_local_timestamp(value: Any, timezone: str) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None

    timestamp = pd.Timestamp(text)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def load_pipeline_config(
    path: Path,
    calendar_override: str | None,
    timezone_override: str | None,
    frequency_override: str | None,
) -> ValidationConfig:
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

    sessions = raw.get("sessions") or DEFAULT_SESSIONS
    source_cfg = raw.get("source") or {}

    timezone = (
        timezone_override
        or raw.get("timezone")
        or DEFAULT_TIMEZONE
    )
    calendar = (
        calendar_override
        or raw.get("calendar")
        or DEFAULT_CALENDAR
    )
    frequency = (
        frequency_override
        or source_cfg.get("frequency")
        or raw.get("frequency")
        or DEFAULT_FREQUENCY
    )
    bar_label = (
        source_cfg.get("bar_label")
        or raw.get("bar_label")
        or DEFAULT_BAR_LABEL
    )

    return ValidationConfig(
        timezone=str(timezone),
        calendar=str(calendar),
        frequency=str(frequency),
        bar_label=str(bar_label),
        sessions=sessions,
    )


def _split_sessions(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return list(DEFAULT_COVERED_SESSIONS)

    if isinstance(value, (list, tuple)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return list(DEFAULT_COVERED_SESSIONS)
        for separator in ("|", ",", ";"):
            if separator in text:
                values = text.split(separator)
                break
        else:
            values = [text]

    return [str(item).strip().lower() for item in values if str(item).strip()]


def load_source_config(
    path: Path,
    source_id: str,
    symbol: str,
    timezone: str,
) -> SourceConfig:
    symbol = symbol.upper()

    if not path.exists():
        return SourceConfig(
            source_id=source_id,
            symbol=symbol,
            coverage_start=None,
            coverage_end=None,
            coverage_sessions=list(DEFAULT_COVERED_SESSIONS),
        )

    table = pd.read_csv(path)
    table.columns = table.columns.str.strip().str.lower()

    required = {"source_id", "symbol"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    mask = (
        table["source_id"].astype(str).eq(source_id)
        & table["symbol"].astype(str).str.upper().eq(symbol)
    )
    matches = table.loc[mask]
    if matches.empty:
        raise ValueError(
            f"No source row for source_id={source_id!r}, symbol={symbol!r} in {path}"
        )

    row = matches.iloc[0]
    return SourceConfig(
        source_id=source_id,
        symbol=symbol,
        coverage_start=_as_local_timestamp(
            row.get("coverage_start"),
            timezone,
        ),
        coverage_end=_as_local_timestamp(
            row.get("coverage_end"),
            timezone,
        ),
        coverage_sessions=_split_sessions(row.get("coverage_sessions")),
    )


def instrument_calendar_override(
    path: Path,
    symbol: str,
) -> str | None:
    if not path.exists():
        return None

    table = pd.read_csv(path)
    table.columns = table.columns.str.strip().str.lower()
    if "symbol" not in table.columns:
        return None

    matches = table[
        table["symbol"].astype(str).str.upper().eq(symbol.upper())
    ]
    if matches.empty:
        return None

    row = matches.iloc[0]
    for column in ("calendar", "exchange_calendar"):
        if column in matches.columns:
            value = str(row[column]).strip()
            if value and value.lower() not in {"nan", "none"}:
                return value
    return None


def read_source_csv(
    path: Path,
    timezone: str,
) -> tuple[pd.DataFrame, dict[str, int], bool]:
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip().str.lower()

    missing = [column for column in CORE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    has_wap = "wap" in raw.columns
    has_bar_count = "bar_count" in raw.columns
    if has_wap != has_bar_count:
        raise ValueError(
            "WAP and bar_count must either both be present or both be absent."
        )
    microstructure_available = has_wap and has_bar_count

    metrics: dict[str, int] = {
        "input_rows": int(len(raw)),
    }

    parsed_datetime = pd.to_datetime(raw["datetime"], errors="coerce")
    metrics["invalid_datetime_rows"] = int(parsed_datetime.isna().sum())

    # The corrected scraper writes New York wall time without timezone suffix.
    # If a timezone-aware input is supplied, convert it to the requested zone.
    if parsed_datetime.notna().any():
        try:
            if parsed_datetime.dt.tz is None:
                local = parsed_datetime.dt.tz_localize(
                    timezone,
                    ambiguous="raise",
                    nonexistent="raise",
                )
            else:
                local = parsed_datetime.dt.tz_convert(timezone)
        except AttributeError:
            # Mixed timezone-aware/naive strings: parse row-wise.
            local_values = []
            for value in raw["datetime"]:
                try:
                    ts = pd.Timestamp(value)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize(timezone)
                    else:
                        ts = ts.tz_convert(timezone)
                    local_values.append(ts)
                except Exception:
                    local_values.append(pd.NaT)
            local = pd.Series(local_values, index=raw.index)
    else:
        local = parsed_datetime

    raw["datetime"] = parsed_datetime
    raw["_dt_local"] = local

    numeric_columns = PRICE_COLUMNS + ["volume"]
    if microstructure_available:
        numeric_columns += MICROSTRUCTURE_COLUMNS

    for column in numeric_columns:
        converted = pd.to_numeric(raw[column], errors="coerce")
        metrics[f"invalid_{column}_rows"] = int(converted.isna().sum())
        raw[column] = converted

    valid = raw[raw["_dt_local"].notna()].copy()
    valid = valid.sort_values("_dt_local").reset_index(drop=True)

    return valid, metrics, microstructure_available


def duplicate_audit(
    df: pd.DataFrame,
    microstructure_available: bool,
) -> tuple[pd.DataFrame, int, int]:
    payload = PRICE_COLUMNS + ["volume"]
    if microstructure_available:
        payload += MICROSTRUCTURE_COLUMNS

    duplicate_rows = df[df.duplicated("_dt_local", keep=False)].copy()
    if duplicate_rows.empty:
        return pd.DataFrame(), 0, 0

    records: list[dict[str, Any]] = []
    exact_timestamp_count = 0
    conflicting_timestamp_count = 0

    for timestamp, group in duplicate_rows.groupby("_dt_local", sort=True):
        unique_payloads = group[payload].drop_duplicates()
        conflict = len(unique_payloads) > 1

        if conflict:
            conflicting_timestamp_count += 1
            duplicate_type = "conflicting"
        else:
            exact_timestamp_count += 1
            duplicate_type = "exact"

        for index in group.index:
            record = {
                "datetime": timestamp,
                "duplicate_type": duplicate_type,
                "source_row_index": int(index),
            }
            for column in payload:
                record[column] = group.loc[index, column]
            records.append(record)

    return (
        pd.DataFrame(records),
        exact_timestamp_count,
        conflicting_timestamp_count,
    )


def wap_ohlc_diagnostics(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Measure WAP distance outside displayed OHLC without altering source data."""
    checkable = (
        df["wap"].notna()
        & df["low"].notna()
        & df["high"].notna()
    )

    lower_excess = (df["low"] - df["wap"]).clip(lower=0.0)
    upper_excess = (df["wap"] - df["high"]).clip(lower=0.0)
    excess = pd.concat(
        [lower_excess, upper_excess],
        axis=1,
    ).max(axis=1)

    outside_exact = checkable & excess.gt(FLOAT_EPSILON)
    over_tolerance = checkable & excess.gt(
        WAP_OHLC_TOLERANCE + FLOAT_EPSILON
    )

    metrics = {
        "wap_ohlc_tolerance": WAP_OHLC_TOLERANCE,
        "wap_outside_displayed_ohlc_rows": int(outside_exact.sum()),
        "wap_over_tolerance_rows": int(over_tolerance.sum()),
        "wap_max_displayed_ohlc_excess": (
            float(excess.loc[outside_exact].max())
            if outside_exact.any()
            else 0.0
        ),
    }
    return over_tolerance, metrics


def row_integrity_audit(
    df: pd.DataFrame,
    microstructure_available: bool,
) -> pd.DataFrame:
    issue_rows: list[dict[str, Any]] = []

    def add_issues(mask: pd.Series, issue: str) -> None:
        for index in df.index[mask.fillna(False)]:
            row = df.loc[index]
            issue_rows.append(
                {
                    "datetime": row["_dt_local"],
                    "issue": issue,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "wap": row.get("wap", pd.NA),
                    "bar_count": row.get("bar_count", pd.NA),
                }
            )

    price_complete = df[PRICE_COLUMNS].notna().all(axis=1)
    high_too_low = price_complete & (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
    )
    low_too_high = price_complete & (
        (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )

    add_issues(high_too_low, "ohlc_high_integrity")
    add_issues(low_too_high, "ohlc_low_integrity")
    add_issues(df["volume"].lt(0), "negative_volume")

    if microstructure_available:
        add_issues(df["bar_count"].lt(0), "negative_bar_count")

        non_integer_count = (
            df["bar_count"].notna()
            & ~df["bar_count"].eq(df["bar_count"].round())
        )
        add_issues(non_integer_count, "non_integer_bar_count")

        wap_over_tolerance, _ = wap_ohlc_diagnostics(df)
        add_issues(
            wap_over_tolerance,
            "wap_exceeds_displayed_ohlc_tolerance",
        )

        add_issues(
            df["volume"].eq(0) & df["bar_count"].gt(0),
            "zero_volume_positive_bar_count",
        )
        add_issues(
            df["volume"].gt(0) & df["bar_count"].eq(0),
            "positive_volume_zero_bar_count",
        )
        add_issues(
            df["bar_count"].gt(0) & df["wap"].isna(),
            "positive_bar_count_missing_wap",
        )

    if not issue_rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "issue",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "wap",
                "bar_count",
            ]
        )

    return pd.DataFrame(issue_rows).sort_values(
        ["datetime", "issue"]
    ).reset_index(drop=True)


def session_windows_for_trading_date(
    trading_date: pd.Timestamp,
    schedule_row: pd.Series,
    config: ValidationConfig,
) -> tuple[dict[str, tuple[pd.Timestamp, pd.Timestamp]], bool]:
    timezone = config.timezone

    exchange_open = schedule_row["open"].tz_convert(timezone)
    exchange_close = schedule_row["close"].tz_convert(timezone)

    regular_cfg = config.sessions["regular"]
    aftermarket_cfg = config.sessions["aftermarket"]
    premarket_cfg = config.sessions["premarket"]

    nominal_regular_close = _clock_on_date(
        trading_date,
        regular_cfg["end_exclusive"],
        timezone,
    )
    nominal_aftermarket_end = _clock_on_date(
        trading_date,
        aftermarket_cfg["end_exclusive"],
        timezone,
    )

    # Our source-session rule:
    # - exchange_calendars owns the actual regular open/close, including
    #   shortened sessions.
    # - premarket begins at the configured source start and runs to actual open.
    # - aftermarket begins at the actual close.
    # - on an early close, preserve the configured post-close duration
    #   (normally four hours: 16:00->20:00, therefore 13:00->17:00).
    nominal_post_duration = nominal_aftermarket_end - nominal_regular_close
    is_early_close = exchange_close < nominal_regular_close

    if is_early_close:
        aftermarket_end = exchange_close + nominal_post_duration
    else:
        aftermarket_end = nominal_aftermarket_end

    premarket_start = _clock_on_date(
        trading_date,
        premarket_cfg["start"],
        timezone,
    )

    windows = {
        "premarket": (premarket_start, exchange_open),
        "regular": (exchange_open, exchange_close),
        "aftermarket": (exchange_close, aftermarket_end),
    }
    return windows, bool(is_early_close)


def expected_index(
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    frequency: str,
) -> pd.DatetimeIndex:
    if end_exclusive <= start:
        return pd.DatetimeIndex([], tz=start.tz)

    return pd.date_range(
        start=start,
        end=end_exclusive,
        freq=frequency,
        inclusive="left",
    )


def build_expected_coverage(
    df: pd.DataFrame,
    config: ValidationConfig,
    source: SourceConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    pd.DataFrame,
]:
    if config.frequency not in {"1min", "1 min", "1T", "min"}:
        raise ValueError(
            f"Validation V2 currently targets 1-minute canonical data; "
            f"received frequency={config.frequency!r}."
        )

    if df.empty:
        raise ValueError("No valid timestamp rows to validate.")

    first_local = df["_dt_local"].min()
    last_local = df["_dt_local"].max()

    first_date = first_local.tz_localize(None).normalize()
    last_date = last_local.tz_localize(None).normalize()

    calendar = xcals.get_calendar(config.calendar)
    trading_dates = calendar.sessions_in_range(first_date, last_date)

    actual_unique = pd.DatetimeIndex(
        df["_dt_local"].drop_duplicates().sort_values()
    )

    coverage_records: list[dict[str, Any]] = []
    gap_records: list[dict[str, Any]] = []
    special_records: list[dict[str, Any]] = []
    expected_frames: list[pd.DataFrame] = []

    gap_totals = {
        "leading_missing_bars": 0,
        "internal_missing_bars": 0,
        "trailing_missing_bars": 0,
        "full_session_missing_bars": 0,
    }

    for session_label in trading_dates:
        schedule_row = calendar.schedule.loc[session_label]
        trading_date = pd.Timestamp(session_label).tz_localize(None)
        windows, is_early_close = session_windows_for_trading_date(
            trading_date,
            schedule_row,
            config,
        )

        if is_early_close:
            special_records.append(
                {
                    "trading_date": trading_date.date(),
                    "exchange_open": schedule_row["open"].tz_convert(
                        config.timezone
                    ),
                    "exchange_close": schedule_row["close"].tz_convert(
                        config.timezone
                    ),
                    "derived_aftermarket_end": windows["aftermarket"][1],
                    "special_type": "early_close",
                }
            )

        for session_name in source.coverage_sessions:
            if session_name not in windows:
                # Overnight is intentionally not fabricated by this source-aware
                # P/R/A validator. A future overnight source needs its own
                # trading-date convention and expected window.
                continue

            start, end = windows[session_name]

            if source.coverage_start is not None:
                if end <= source.coverage_start:
                    continue
                start = max(start, source.coverage_start)

            if source.coverage_end is not None:
                if start > source.coverage_end:
                    continue
                end = min(
                    end,
                    source.coverage_end + pd.Timedelta(minutes=1),
                )

            expected = expected_index(start, end, "1min")
            if expected.empty:
                continue

            expected_frame = pd.DataFrame(
                {
                    "datetime": expected,
                    "trading_date": trading_date.date(),
                    "session": session_name,
                }
            )
            expected_frames.append(expected_frame)

            actual_in_session = actual_unique[
                (actual_unique >= start) & (actual_unique < end)
            ]
            missing = expected.difference(actual_in_session)

            expected_count = len(expected)
            actual_count = len(expected.intersection(actual_unique))
            missing_count = len(missing)

            coverage_records.append(
                {
                    "trading_date": trading_date.date(),
                    "session": session_name,
                    "expected_start": expected[0],
                    "expected_end": expected[-1],
                    "expected_bars": expected_count,
                    "actual_bars": actual_count,
                    "missing_bars": missing_count,
                    "coverage_percent": round(
                        actual_count / expected_count * 100,
                        6,
                    )
                    if expected_count
                    else 0.0,
                    "is_early_close": is_early_close,
                }
            )

            if missing_count == 0:
                continue

            if actual_in_session.empty:
                gap_totals["full_session_missing_bars"] += missing_count
                gap_records.append(
                    {
                        "trading_date": trading_date.date(),
                        "session": session_name,
                        "gap_type": "full_session",
                        "gap_start": missing[0],
                        "gap_end": missing[-1],
                        "missing_bars": missing_count,
                        "first_observed_in_session": pd.NaT,
                        "last_observed_in_session": pd.NaT,
                    }
                )
                continue

            first_actual = actual_in_session.min()
            last_actual = actual_in_session.max()

            missing_series = pd.Series(missing)
            run_id = (
                missing_series.diff().ne(pd.Timedelta(minutes=1)).cumsum()
            )

            for _, run in missing_series.groupby(run_id):
                run_start = run.iloc[0]
                run_end = run.iloc[-1]
                run_count = len(run)

                if run_end < first_actual:
                    gap_type = "leading"
                    gap_totals["leading_missing_bars"] += run_count
                elif run_start > last_actual:
                    gap_type = "trailing"
                    gap_totals["trailing_missing_bars"] += run_count
                else:
                    gap_type = "internal"
                    gap_totals["internal_missing_bars"] += run_count

                gap_records.append(
                    {
                        "trading_date": trading_date.date(),
                        "session": session_name,
                        "gap_type": gap_type,
                        "gap_start": run_start,
                        "gap_end": run_end,
                        "missing_bars": run_count,
                        "first_observed_in_session": first_actual,
                        "last_observed_in_session": last_actual,
                    }
                )

    coverage = pd.DataFrame(coverage_records)
    gaps = pd.DataFrame(gap_records)
    special = pd.DataFrame(special_records)

    if expected_frames:
        expected_all = pd.concat(expected_frames, ignore_index=True)
    else:
        expected_all = pd.DataFrame(
            columns=["datetime", "trading_date", "session"]
        )

    return coverage, gaps, special, gap_totals, expected_all


def label_observed_rows(
    df: pd.DataFrame,
    expected_all: pd.DataFrame,
) -> pd.DataFrame:
    labels = expected_all.drop_duplicates("datetime").set_index("datetime")
    result = df.copy()

    result["session"] = result["_dt_local"].map(labels["session"])
    result["trading_date"] = result["_dt_local"].map(labels["trading_date"])

    result["calendar_date"] = result["_dt_local"].dt.date
    return result


def outside_expected_report(
    labeled: pd.DataFrame,
    source: SourceConfig,
) -> pd.DataFrame:
    outside = labeled[labeled["session"].isna()].copy()
    if outside.empty:
        return pd.DataFrame(
            columns=CORE_COLUMNS
            + MICROSTRUCTURE_COLUMNS
            + ["calendar_date", "outside_reason"]
        )

    reason = pd.Series(
        "outside_expected_session",
        index=outside.index,
        dtype="object",
    )

    if source.coverage_start is not None:
        reason.loc[
            outside["_dt_local"] < source.coverage_start
        ] = "before_source_coverage_start"

    if source.coverage_end is not None:
        reason.loc[
            outside["_dt_local"] > source.coverage_end
        ] = "after_source_coverage_end"

    outside["outside_reason"] = reason

    columns = [
        "_dt_local",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    for optional in MICROSTRUCTURE_COLUMNS:
        if optional in outside.columns:
            columns.append(optional)
    columns.extend(["calendar_date", "outside_reason"])

    result = outside[columns].rename(
        columns={"_dt_local": "datetime"}
    )
    return result.reset_index(drop=True)


def microstructure_reports(
    labeled: pd.DataFrame,
    microstructure_available: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not microstructure_available:
        return (
            {"available": False},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    data = labeled.copy()
    data["bar_count"] = data["bar_count"].astype("Float64")

    zero_volume = data["volume"].eq(0)
    zero_count = data["bar_count"].eq(0)
    positive_volume = data["volume"].gt(0)
    positive_count = data["bar_count"].gt(0)

    flat_ohlc = (
        data["open"].eq(data["high"])
        & data["open"].eq(data["low"])
        & data["open"].eq(data["close"])
    )

    previous_time = data["_dt_local"].shift(1)
    previous_close = data["close"].shift(1)
    previous_session = data["session"].shift(1)

    contiguous_previous_minute = (
        data["_dt_local"] - previous_time
    ).eq(pd.Timedelta(minutes=1))
    same_session = data["session"].eq(previous_session)

    close_carried = (
        zero_volume
        & zero_count
        & contiguous_previous_minute
        & same_session
        & data["close"].eq(previous_close)
    )

    positive_trade_mask = positive_count & data["volume"].notna()
    volume_per_count = (
        data.loc[positive_trade_mask, "volume"]
        / data.loc[positive_trade_mask, "bar_count"]
    )

    summary: dict[str, Any] = {
        "available": True,
        "observed_rows": int(len(data)),
        "zero_volume_rows": int(zero_volume.sum()),
        "zero_bar_count_rows": int(zero_count.sum()),
        "zero_volume_zero_bar_count_rows": int(
            (zero_volume & zero_count).sum()
        ),
        "zero_volume_positive_bar_count_rows": int(
            (zero_volume & positive_count).sum()
        ),
        "positive_volume_zero_bar_count_rows": int(
            (positive_volume & zero_count).sum()
        ),
        "positive_volume_positive_bar_count_rows": int(
            (positive_volume & positive_count).sum()
        ),
        "zero_volume_zero_count_flat_ohlc_rows": int(
            (zero_volume & zero_count & flat_ohlc).sum()
        ),
        "zero_volume_zero_count_carried_close_rows": int(close_carried.sum()),
        "missing_wap_rows": int(data["wap"].isna().sum()),
        **wap_ohlc_diagnostics(data)[1],
        "mean_volume_per_count_when_count_positive": (
            round(float(volume_per_count.mean()), 6)
            if not volume_per_count.empty
            else None
        ),
        "median_volume_per_count_when_count_positive": (
            round(float(volume_per_count.median()), 6)
            if not volume_per_count.empty
            else None
        ),
    }

    session_records: list[dict[str, Any]] = []
    for session_name, group in data.groupby("session", dropna=False):
        group_zero_volume = group["volume"].eq(0)
        group_zero_count = group["bar_count"].eq(0)
        group_positive_count = group["bar_count"].gt(0)
        ratio = (
            group.loc[group_positive_count, "volume"]
            / group.loc[group_positive_count, "bar_count"]
        )

        session_records.append(
            {
                "session": (
                    str(session_name)
                    if pd.notna(session_name)
                    else "outside_expected"
                ),
                "rows": int(len(group)),
                "zero_volume_rows": int(group_zero_volume.sum()),
                "zero_volume_percent": round(
                    group_zero_volume.mean() * 100,
                    6,
                ),
                "zero_bar_count_rows": int(group_zero_count.sum()),
                "zero_volume_zero_bar_count_rows": int(
                    (group_zero_volume & group_zero_count).sum()
                ),
                "median_volume": round(
                    float(group["volume"].median()),
                    6,
                ),
                "median_bar_count": round(
                    float(group["bar_count"].median()),
                    6,
                ),
                "median_volume_per_count_when_count_positive": (
                    round(float(ratio.median()), 6)
                    if not ratio.empty
                    else None
                ),
            }
        )

    by_session = pd.DataFrame(session_records)

    data["hour"] = data["_dt_local"].dt.hour
    hour_records: list[dict[str, Any]] = []
    for hour, group in data.groupby("hour"):
        zero = group["volume"].eq(0)
        hour_records.append(
            {
                "hour": int(hour),
                "rows": int(len(group)),
                "zero_volume_rows": int(zero.sum()),
                "zero_volume_percent": round(zero.mean() * 100, 6),
                "median_volume": round(
                    float(group["volume"].median()),
                    6,
                ),
                "median_bar_count": round(
                    float(group["bar_count"].median()),
                    6,
                ),
            }
        )

    by_hour = pd.DataFrame(hour_records)
    return summary, by_session, by_hour


def aggregate_session_coverage(
    daily_coverage: pd.DataFrame,
) -> pd.DataFrame:
    if daily_coverage.empty:
        return pd.DataFrame()

    result = (
        daily_coverage.groupby("session", as_index=False)
        .agg(
            expected_bars=("expected_bars", "sum"),
            actual_bars=("actual_bars", "sum"),
            missing_bars=("missing_bars", "sum"),
            trading_dates=("trading_date", "nunique"),
        )
    )
    result["coverage_percent"] = (
        result["actual_bars"] / result["expected_bars"] * 100
    ).round(6)
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (dict, list)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            json_ready(payload),
            stream,
            indent=2,
            sort_keys=False,
            default=str,
        )


def validate_file(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    config = load_pipeline_config(
        args.config,
        args.calendar,
        args.timezone,
        args.frequency,
    )

    instrument_calendar = instrument_calendar_override(
        args.instruments,
        args.symbol,
    )
    if args.calendar is None and instrument_calendar:
        config = ValidationConfig(
            timezone=config.timezone,
            calendar=instrument_calendar,
            frequency=config.frequency,
            bar_label=config.bar_label,
            sessions=config.sessions,
        )

    source = load_source_config(
        args.sources,
        args.source_id,
        args.symbol,
        config.timezone,
    )

    df, parse_metrics, microstructure_available = read_source_csv(
        input_path,
        config.timezone,
    )

    duplicate_rows, duplicate_exact, duplicate_conflicting = duplicate_audit(
        df,
        microstructure_available,
    )
    row_issues = row_integrity_audit(
        df,
        microstructure_available,
    )

    (
        daily_coverage,
        gaps,
        special_sessions,
        gap_totals,
        expected_all,
    ) = build_expected_coverage(
        df,
        config,
        source,
    )

    labeled = label_observed_rows(df, expected_all)
    outside_expected = outside_expected_report(labeled, source)

    session_coverage = aggregate_session_coverage(daily_coverage)

    micro_summary, micro_by_session, micro_by_hour = microstructure_reports(
        labeled,
        microstructure_available,
    )

    if microstructure_available:
        _, wap_metrics = wap_ohlc_diagnostics(df)
    else:
        wap_metrics = {
            "wap_ohlc_tolerance": WAP_OHLC_TOLERANCE,
            "wap_outside_displayed_ohlc_rows": 0,
            "wap_over_tolerance_rows": 0,
            "wap_max_displayed_ohlc_excess": 0.0,
        }

    if outside_expected.empty:
        outside_reason_counts: dict[str, int] = {}
    else:
        outside_reason_counts = {
            str(key): int(value)
            for key, value in (
                outside_expected["outside_reason"]
                .value_counts()
                .sort_index()
                .items()
            )
        }

    rows_before_coverage_start = int(
        outside_reason_counts.get("before_source_coverage_start", 0)
    )
    rows_after_coverage_end = int(
        outside_reason_counts.get("after_source_coverage_end", 0)
    )
    true_outside_expected_sessions = int(
        outside_reason_counts.get("outside_expected_session", 0)
    )

    expected_bars = int(daily_coverage["expected_bars"].sum())
    actual_bars = int(daily_coverage["actual_bars"].sum())
    missing_bars = int(daily_coverage["missing_bars"].sum())

    structural_failures = (
        parse_metrics["invalid_datetime_rows"]
        + sum(
            value
            for key, value in parse_metrics.items()
            if key.startswith("invalid_") and key != "invalid_datetime_rows"
        )
        + duplicate_conflicting
        + len(row_issues)
    )

    if structural_failures:
        status = "FAIL"
    elif (
        missing_bars
        or rows_before_coverage_start
        or rows_after_coverage_end
        or true_outside_expected_sessions
    ):
        status = "WARN"
    else:
        status = "PASS"

    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = (
        args.reports
        / args.symbol.lower()
        / args.source_id
        / run_id
    )
    report_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "validator_version": "2.1.0",
        "validation_status": status,
        "symbol": args.symbol.upper(),
        "source_id": args.source_id,
        "input_file": str(input_path),
        "input_sha256": sha256_file(input_path),
        "timezone": config.timezone,
        "calendar": config.calendar,
        "frequency": config.frequency,
        "bar_label": config.bar_label,
        "covered_sessions": source.coverage_sessions,
        "source_coverage_start": source.coverage_start,
        "source_coverage_end": source.coverage_end,
        "first_observed_datetime": (
            df["_dt_local"].min() if not df.empty else None
        ),
        "last_observed_datetime": (
            df["_dt_local"].max() if not df.empty else None
        ),
        **parse_metrics,
        "analyzed_rows": int(len(df)),
        "expected_bars": expected_bars,
        "actual_bars": actual_bars,
        "missing_bars": missing_bars,
        "coverage_percent": (
            round(actual_bars / expected_bars * 100, 6)
            if expected_bars
            else 0.0
        ),
        "session_totals": (
            {
                row["session"]: {
                    "expected_bars": int(row["expected_bars"]),
                    "actual_bars": int(row["actual_bars"]),
                    "missing_bars": int(row["missing_bars"]),
                    "coverage_percent": float(row["coverage_percent"]),
                }
                for _, row in session_coverage.iterrows()
            }
            if not session_coverage.empty
            else {}
        ),
        **gap_totals,
        "gap_records": int(len(gaps)),
        "duplicate_exact_timestamps": int(duplicate_exact),
        "duplicate_conflicting_timestamps": int(duplicate_conflicting),
        "row_issue_records": int(len(row_issues)),
        **wap_metrics,
        "rows_before_source_coverage_start": rows_before_coverage_start,
        "rows_after_source_coverage_end": rows_after_coverage_end,
        "rows_outside_expected_sessions": true_outside_expected_sessions,
        "outside_expected_reason_counts": outside_reason_counts,
        "early_close_trading_dates": int(len(special_sessions)),
        "zero_volume_observed_rows": int(df["volume"].eq(0).sum()),
        "microstructure_available": bool(microstructure_available),
        "report_directory": str(report_dir.resolve()),
    }

    summary_json = report_dir / "summary.json"
    write_json(summary_json, summary)

    daily_coverage.to_csv(
        report_dir / "daily_coverage.csv",
        index=False,
    )
    session_coverage.to_csv(
        report_dir / "session_coverage.csv",
        index=False,
    )
    gaps.to_csv(
        report_dir / "gaps.csv",
        index=False,
    )
    special_sessions.to_csv(
        report_dir / "special_sessions.csv",
        index=False,
    )
    row_issues.to_csv(
        report_dir / "row_issues.csv",
        index=False,
    )
    duplicate_rows.to_csv(
        report_dir / "duplicates.csv",
        index=False,
    )
    outside_expected.to_csv(
        report_dir / "outside_expected_sessions.csv",
        index=False,
    )

    write_json(
        report_dir / "volume_summary.json",
        micro_summary,
    )
    if microstructure_available:
        micro_by_session.to_csv(
            report_dir / "volume_by_session.csv",
            index=False,
        )
        micro_by_hour.to_csv(
            report_dir / "volume_by_hour.csv",
            index=False,
        )

    return summary


def main() -> int:
    args = parse_args()
    result = validate_file(args)
    print(json.dumps(json_ready(result), indent=2, default=str))
    return 1 if result["validation_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())