#!/usr/bin/env python3
"""
IBKR historical-data scraper V3.

Core design
-----------
The filesystem, not checkpoint.json, is the source of truth.

Each successfully downloaded trading day is stored at one deterministic,
lower-case path:

    <output_root>/<symbol>/<bar>/<what_to_show>/<session>/<YYYY>/<YYYY-MM-DD>.csv

Example:

    data/raw/ibkr/avgo/1min/trades/extended/2026/2026-01-15.csv

Symbols are case-insensitive in YAML/CLI. They are normalized to upper case only
for the IBKR contract/request and normalized to lower case for filesystem names.

This makes historical requests idempotent:

    requested exchange dates
        - valid daily files already on disk
        = dates that still need downloading

If a process/network failure occurs, rerunning the same or an overlapping
request rescans the daily store and downloads only missing dates.

checkpoint.json is retained only as a progress hint/audit record. It is never
trusted to decide which date should be downloaded next.

Raw daily files are never overwritten automatically. If an existing daily
file is invalid or an imported file conflicts with an existing valid file,
the scraper stops and requires explicit human review.

Examples
--------
Batch YAML:
    python ibq_scraper.py --config config/scraper-config-avgo.yml

Single symbol:
    python ibq_scraper.py \
        -s AVGO \
        --session extended \
        -bs "1 min" \
        --start-date 2021-08-08 \
        --end-date 2025-08-07 \
        --output-root data/raw/ibkr

Seed the V3 daily store from an already corrected combined CSV:
    python ibq_scraper.py \
        --config config/scraper-config-avgo-current.yml \
        --import-csv data/raw/bootstrap/AVGO/avgo_1min_extended.csv

Snapshots are generated automatically after a successful download/import.
For TRADES + extended + 1 min, AVGO becomes:

    data/raw/bootstrap/avgo/avgo_1min_extended.csv

For maintenance, all deterministic snapshots can be rebuilt offline:

    python ibq_scraper.py \
        --config config/scraper-config-semicond.yml \
        --rebuild-snapshots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import yaml
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

import config as app_config
from utils.appenv import APPENV


# Prefer BASE_DIR. Fall back to the parent of the older DATA_DIR setting.
if hasattr(app_config, "BASE_DIR"):
    BASE_DIR = Path(app_config.BASE_DIR)
elif hasattr(app_config, "DATA_DIR"):
    BASE_DIR = Path(app_config.DATA_DIR).parent
else:
    BASE_DIR = Path(__file__).resolve().parent


MARKET_TIMEZONE = "America/New_York"
IB_REQUEST_TIMEZONE = "US/Eastern"
DEFAULT_CALENDAR = "XNYS"

RAW_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "wap",
    "bar_count",
]

# IBKR historical WAP can carry more decimal precision than the displayed
# OHLC values. For stock bars, allow up to half a cent outside the displayed
# OHLC range so harmless rounding differences do not stop historical scraping.
WAP_OHLC_TOLERANCE = 0.005

INFO_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}


@dataclass(frozen=True)
class DownloadJob:
    symbol: str
    group: str
    duration: str
    bar_size: str
    session: str = "rth"
    what_to_show: str = "TRADES"
    exchange: str | None = None
    primary_exchange: str | None = None
    security_type: str = "STK"
    currency: str = "USD"
    start_date: str | None = None
    end_date: str | None = None
    calendar: str = DEFAULT_CALENDAR


@dataclass(frozen=True)
class RuntimeConfig:
    jobs: list[DownloadJob]
    max_workers: int
    output_root: Path
    snapshot_root: Path
    auto_snapshot: bool
    request_pause_seconds: float
    timeout_seconds: int


@dataclass(frozen=True)
class DailyValidation:
    path: Path
    trading_date: date
    row_count: int
    first_datetime: pd.Timestamp
    last_datetime: pd.Timestamp
    sha256: str


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    requested_dates: int
    already_present: int
    downloaded: int
    output_directory: Path


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def safe_name(value: str) -> str:
    """Return one normalized lower-case filesystem component."""
    cleaned = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        str(value).strip(),
    ).strip("._")
    return cleaned.lower()


def parse_sessions(settings: dict[str, Any]) -> list[str]:
    raw_sessions = settings.get("sessions")
    if raw_sessions is None:
        return [
            "overnight"
            if parse_bool(settings.get("overnight", False))
            else "rth"
        ]

    if isinstance(raw_sessions, str):
        raw_sessions = [raw_sessions]

    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError("'sessions' must be a non-empty string or list.")

    aliases = {
        "rth": "rth",
        "regular": "rth",
        "regular_hours": "rth",
        "extended": "extended",
        "extended_hours": "extended",
        "pre_post": "extended",
        "pre-market": "extended",
        "post-market": "extended",
        "overnight": "overnight",
    }

    result: list[str] = []
    for value in raw_sessions:
        normalized = str(value).strip().lower().replace(" ", "_")
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported session {value!r}; use rth, extended, or overnight."
            )
        canonical = aliases[normalized]
        if canonical not in result:
            result.append(canonical)
    return result


def normalize_bar_size(bar_size: str) -> str:
    value = " ".join(str(bar_size).strip().lower().split())

    aliases = {
        "1 sec": "1 secs",
        "1 second": "1 secs",
        "1 seconds": "1 secs",
        "5 sec": "5 secs",
        "5 second": "5 secs",
        "5 seconds": "5 secs",
        "10 sec": "10 secs",
        "10 second": "10 secs",
        "10 seconds": "10 secs",
        "15 sec": "15 secs",
        "15 second": "15 secs",
        "15 seconds": "15 secs",
        "30 sec": "30 secs",
        "30 second": "30 secs",
        "30 seconds": "30 secs",
        "1 mins": "1 min",
        "1 minute": "1 min",
        "1 minutes": "1 min",
        "2 min": "2 mins",
        "2 minute": "2 mins",
        "2 minutes": "2 mins",
        "3 min": "3 mins",
        "3 minute": "3 mins",
        "3 minutes": "3 mins",
        "5 min": "5 mins",
        "5 minute": "5 mins",
        "5 minutes": "5 mins",
        "10 min": "10 mins",
        "10 minute": "10 mins",
        "10 minutes": "10 mins",
        "15 min": "15 mins",
        "15 minute": "15 mins",
        "15 minutes": "15 mins",
        "20 min": "20 mins",
        "20 minute": "20 mins",
        "20 minutes": "20 mins",
        "30 min": "30 mins",
        "30 minute": "30 mins",
        "30 minutes": "30 mins",
        "1 hours": "1 hour",
        "2 hour": "2 hours",
        "3 hour": "3 hours",
        "4 hour": "4 hours",
        "8 hour": "8 hours",
        "1 days": "1 day",
        "1 week": "1 week",
        "1 weeks": "1 week",
        "1 month": "1 month",
        "1 months": "1 month",
    }

    return aliases.get(value, value)


def is_intraday_bar_size(bar_size: str) -> bool:
    return normalize_bar_size(bar_size) not in {"1 day", "1 week", "1 month"}


def bar_storage_name(bar_size: str) -> str:
    return safe_name(normalize_bar_size(bar_size).replace(" ", ""))


def parse_config_date(value: str | None, field_name: str) -> date | None:
    if value is None or not str(value).strip():
        return None

    timestamp = pd.Timestamp(str(value).strip())
    if pd.isna(timestamp):
        raise ValueError(f"Invalid {field_name}: {value!r}")

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(MARKET_TIMEZONE).tz_localize(None)

    return timestamp.normalize().date()


def duration_start_date(end_date: date, duration: str) -> date:
    match = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", duration.upper())
    if not match:
        raise ValueError(
            f"Unsupported duration {duration!r}; "
            "use forms such as '30 D', '12 M', or '5 Y'."
        )

    value, unit = int(match.group(1)), match.group(2)
    end_ts = pd.Timestamp(end_date)

    if unit == "D":
        start_ts = end_ts - pd.Timedelta(days=value)
    elif unit == "W":
        start_ts = end_ts - pd.Timedelta(weeks=value)
    elif unit == "M":
        start_ts = end_ts - pd.DateOffset(months=value)
    else:
        start_ts = end_ts - pd.DateOffset(years=value)

    return start_ts.date()


def resolve_date_range(job: DownloadJob) -> tuple[date, date]:
    end_date = parse_config_date(job.end_date, "end_date")
    if end_date is None:
        end_date = pd.Timestamp.now(tz=MARKET_TIMEZONE).date()

    start_date = parse_config_date(job.start_date, "start_date")
    if start_date is None:
        start_date = duration_start_date(end_date, job.duration)

    if start_date > end_date:
        raise ValueError(
            f"{job.symbol}: start_date {start_date} is after end_date {end_date}"
        )

    return start_date, end_date


def trading_dates_for_job(job: DownloadJob) -> list[date]:
    start_date, end_date = resolve_date_range(job)

    calendar = xcals.get_calendar(job.calendar)
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )

    return [pd.Timestamp(session).date() for session in sessions]


def load_jobs(config_path: Path) -> RuntimeConfig:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    defaults = config.get("defaults", {})
    groups = config.get("groups", {})

    if not isinstance(groups, dict) or not groups:
        raise ValueError("Config must contain a non-empty 'groups' mapping.")

    # Dataset identity deliberately excludes `group`. A context symbol such as
    # SPY can appear in several groups without creating several copies of the
    # same raw market data.
    jobs_by_key: dict[tuple[Any, ...], DownloadJob] = {}
    referenced_groups: dict[tuple[Any, ...], set[str]] = {}

    for group, group_config in groups.items():
        if isinstance(group_config, list):
            group_config = {"symbols": group_config}

        if not isinstance(group_config, dict):
            raise ValueError(f"Group {group!r} must be a list or mapping.")

        merged = {**defaults, **group_config}

        symbols = merged.get("symbols", []) or []
        context_symbols = merged.get("context_symbols", []) or []

        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(context_symbols, str):
            context_symbols = [context_symbols]

        symbol_entries = list(symbols) + list(context_symbols)
        if not symbol_entries:
            raise ValueError(
                f"Group {group!r} has neither symbols nor context_symbols."
            )

        for symbol_entry in symbol_entries:
            overrides = (
                symbol_entry
                if isinstance(symbol_entry, dict)
                else {}
            )
            symbol = (
                overrides.get("symbol")
                if overrides
                else symbol_entry
            )

            if not symbol:
                raise ValueError(
                    f"Invalid symbol entry in group {group!r}."
                )

            settings = {**merged, **overrides}

            for session in parse_sessions(settings):
                # Internal/IBKR identity is upper case. YAML may freely use
                # avgo, AVGO, AvGo, etc.
                job = DownloadJob(
                    symbol=str(symbol).strip().upper(),
                    group=str(group).strip().lower(),
                    duration=str(
                        settings.get("duration", "365 D")
                    ),
                    bar_size=str(
                        settings.get("bar_size", "1 day")
                    ),
                    session=session,
                    what_to_show=str(
                        settings.get("what_to_show", "TRADES")
                    ).upper(),
                    exchange=settings.get("exchange"),
                    primary_exchange=settings.get(
                        "primary_exchange"
                    ),
                    security_type=str(
                        settings.get("security_type", "STK")
                    ).upper(),
                    currency=str(
                        settings.get("currency", "USD")
                    ).upper(),
                    start_date=(
                        str(settings["start_date"])
                        if settings.get("start_date") is not None
                        else None
                    ),
                    end_date=(
                        str(settings["end_date"])
                        if settings.get("end_date") is not None
                        else None
                    ),
                    calendar=str(
                        settings.get(
                            "calendar",
                            DEFAULT_CALENDAR,
                        )
                    ),
                )

                key = (
                    job.symbol,
                    normalize_bar_size(job.bar_size),
                    job.session,
                    job.what_to_show,
                    job.exchange,
                    job.primary_exchange,
                    job.security_type,
                    job.currency,
                    job.calendar,
                )

                referenced_groups.setdefault(key, set()).add(
                    str(group).strip().lower()
                )

                if key not in jobs_by_key:
                    jobs_by_key[key] = job

    jobs = list(jobs_by_key.values())

    for key, groups_for_dataset in referenced_groups.items():
        if len(groups_for_dataset) > 1:
            job = jobs_by_key[key]
            print(
                f"[config] deduplicated {job.symbol} "
                f"{normalize_bar_size(job.bar_size)} "
                f"{job.session}/{job.what_to_show}; "
                f"referenced by groups={sorted(groups_for_dataset)}"
            )

    max_workers = int(config.get("max_workers", 3))
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    output_root = Path(
        config.get(
            "output_root",
            Path(BASE_DIR) / "data" / "raw" / "ibkr",
        )
    )
    if not output_root.is_absolute():
        output_root = Path(BASE_DIR) / output_root

    snapshot_root = Path(
        config.get(
            "snapshot_root",
            Path(BASE_DIR) / "data" / "raw" / "bootstrap",
        )
    )
    if not snapshot_root.is_absolute():
        snapshot_root = Path(BASE_DIR) / snapshot_root

    auto_snapshot = parse_bool(
        config.get("auto_snapshot", True)
    )

    request_pause_seconds = float(
        config.get("request_pause_seconds", 1.5)
    )
    timeout_seconds = int(
        config.get("timeout_seconds", 120)
    )

    return RuntimeConfig(
        jobs=jobs,
        max_workers=max_workers,
        output_root=output_root,
        snapshot_root=snapshot_root,
        auto_snapshot=auto_snapshot,
        request_pause_seconds=request_pause_seconds,
        timeout_seconds=timeout_seconds,
    )

def job_storage_root(job: DownloadJob, output_root: Path) -> Path:
    return (
        output_root
        / safe_name(job.symbol)
        / bar_storage_name(job.bar_size)
        / safe_name(job.what_to_show)
        / safe_name(job.session)
    )


def snapshot_filename(job: DownloadJob) -> str:
    """Return the deterministic combined-history filename for one dataset.

    Current TRADES examples:
        avgo_1min_extended.csv
        spy_1min_extended.csv
        avgo_1min_rth.csv

    For non-TRADES data, append what_to_show to avoid a collision with TRADES:
        avgo_1min_extended_midpoint.csv
    """
    parts = [
        safe_name(job.symbol),
        bar_storage_name(job.bar_size),
        safe_name(job.session),
    ]

    if job.what_to_show.upper() != "TRADES":
        parts.append(safe_name(job.what_to_show))

    return "_".join(parts) + ".csv"


def snapshot_output_path(
    job: DownloadJob,
    snapshot_root: Path,
) -> Path:
    return (
        snapshot_root
        / safe_name(job.symbol)
        / snapshot_filename(job)
    )


def daily_file_path(
    job: DownloadJob,
    output_root: Path,
    trading_date: date,
) -> Path:
    return (
        job_storage_root(job, output_root)
        / f"{trading_date:%Y}"
        / f"{trading_date:%Y-%m-%d}.csv"
    )


def job_checkpoint_path(
    job: DownloadJob,
    output_root: Path,
) -> Path:
    start_date, end_date = resolve_date_range(job)
    filename = (
        f"{start_date:%Y%m%d}_{end_date:%Y%m%d}.checkpoint.json"
    )
    return job_storage_root(job, output_root) / ".state" / filename


def manifest_path(job: DownloadJob, output_root: Path) -> Path:
    return job_storage_root(job, output_root) / "manifest.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temp_path, path)


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in RAW_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Downloaded/imported chunk is missing columns: {missing_columns}"
        )

    result = df.loc[:, RAW_COLUMNS].copy()

    numeric_datetime = pd.to_numeric(
        result["datetime"],
        errors="coerce",
    )
    epoch_mask = numeric_datetime.between(
        1_000_000_000,
        4_000_000_000,
    )

    parsed = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns]",
    )

    if epoch_mask.any():
        epoch_values = pd.to_datetime(
            numeric_datetime.loc[epoch_mask],
            unit="s",
            utc=True,
            errors="coerce",
        )
        parsed.loc[epoch_mask] = (
            epoch_values
            .dt.tz_convert(MARKET_TIMEZONE)
            .dt.tz_localize(None)
        )

    text_mask = ~epoch_mask
    if text_mask.any():
        text_values = pd.to_datetime(
            result.loc[text_mask, "datetime"],
            errors="coerce",
        )

        # Corrected raw files are New York wall time with no timezone suffix.
        # If a timezone-aware series is encountered, normalize it explicitly.
        try:
            if text_values.dt.tz is not None:
                text_values = (
                    text_values
                    .dt.tz_convert(MARKET_TIMEZONE)
                    .dt.tz_localize(None)
                )
        except AttributeError:
            pass

        parsed.loc[text_mask] = text_values

    result["datetime"] = parsed

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "bar_count",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = (
        result
        .dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return result


def validate_daily_frame(
    df: pd.DataFrame,
    requested_trading_date: date,
    job: DownloadJob,
) -> None:
    if df.empty:
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: daily chunk is empty."
        )

    missing_columns = [
        column for column in RAW_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: "
            f"missing columns {missing_columns}"
        )

    if df["datetime"].isna().any():
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: invalid datetime value."
        )

    if df["datetime"].duplicated().any():
        duplicates = df.loc[
            df["datetime"].duplicated(keep=False),
            "datetime",
        ].head(5).tolist()
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: duplicate timestamps "
            f"found, examples={duplicates}"
        )

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: "
            "timestamps are not monotonically increasing."
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "bar_count",
    ]
    invalid_numeric = df[numeric_columns].isna()
    if invalid_numeric.any().any():
        bad = invalid_numeric.sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: "
            f"invalid numeric values {bad}"
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: negative volume."
        )

    if (df["bar_count"] < 0).any():
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: negative bar_count."
        )

    non_integer_bar_count = ~df["bar_count"].eq(df["bar_count"].round())
    if non_integer_bar_count.any():
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: "
            "bar_count contains non-integer values."
        )

    high_bad = df["high"] < df[
        ["open", "low", "close"]
    ].max(axis=1)
    low_bad = df["low"] > df[
        ["open", "high", "close"]
    ].min(axis=1)

    if high_bad.any() or low_bad.any():
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: OHLC integrity failure."
        )

    # WAP may have finer precision than displayed OHLC. Measure only the
    # amount by which WAP falls outside the displayed low/high interval.
    wap_distance = pd.concat(
        [
            (df["low"] - df["wap"]).clip(lower=0),
            (df["wap"] - df["high"]).clip(lower=0),
        ],
        axis=1,
    ).max(axis=1)

    wap_bad = (
        wap_distance
        > (WAP_OHLC_TOLERANCE + 1e-12)
    )

    if wap_bad.any():
        max_distance = float(
            wap_distance.loc[wap_bad].max()
        )
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: "
            f"{int(wap_bad.sum())} WAP value(s) exceed "
            "the displayed OHLC range by more than "
            f"${WAP_OHLC_TOLERANCE:.3f}; "
            f"max_excess=${max_distance:.6f}."
        )

    observed_dates = set(df["datetime"].dt.date.unique())

    if job.session in {"extended", "rth"}:
        allowed_dates = {requested_trading_date}
    else:
        # Future overnight data can span the prior evening and current
        # trading date. Overnight semantics will be validated more deeply in
        # the canonical validator.
        allowed_dates = {
            requested_trading_date - timedelta(days=1),
            requested_trading_date,
        }

    unexpected_dates = observed_dates.difference(allowed_dates)
    if unexpected_dates:
        raise ValueError(
            f"{job.symbol} {requested_trading_date}: "
            f"response contains unexpected calendar dates "
            f"{sorted(unexpected_dates)}"
        )


def load_and_validate_daily_file(
    path: Path,
    trading_date: date,
    job: DownloadJob,
) -> DailyValidation:
    frame = prepare_chunk(pd.read_csv(path))
    validate_daily_frame(frame, trading_date, job)

    return DailyValidation(
        path=path,
        trading_date=trading_date,
        row_count=len(frame),
        first_datetime=frame["datetime"].iloc[0],
        last_datetime=frame["datetime"].iloc[-1],
        sha256=sha256_file(path),
    )


def frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_norm = (
        prepare_chunk(left)
        .reset_index(drop=True)
    )
    right_norm = (
        prepare_chunk(right)
        .reset_index(drop=True)
    )

    if len(left_norm) != len(right_norm):
        return False

    return left_norm.equals(right_norm)


def write_daily_frame_atomic(
    frame: pd.DataFrame,
    path: Path,
    trading_date: date,
    job: DownloadJob,
) -> tuple[DailyValidation, bool]:
    """
    Write one daily raw file atomically.

    Returns (validation, created).

    Existing valid raw files are never overwritten. If the proposed content
    differs from an existing valid file, raise a conflict.
    """
    prepared = prepare_chunk(frame)
    validate_daily_frame(prepared, trading_date, job)

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path)
        existing_prepared = prepare_chunk(existing)
        validate_daily_frame(existing_prepared, trading_date, job)

        if frames_equal(existing_prepared, prepared):
            return (
                DailyValidation(
                    path=path,
                    trading_date=trading_date,
                    row_count=len(existing_prepared),
                    first_datetime=existing_prepared["datetime"].iloc[0],
                    last_datetime=existing_prepared["datetime"].iloc[-1],
                    sha256=sha256_file(path),
                ),
                False,
            )

        raise RuntimeError(
            f"RAW CONFLICT: {path} already exists and differs from the "
            "new data. Raw daily files are never overwritten automatically."
        )

    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    prepared.to_csv(temp_path, index=False)

    # Make sure bytes reach disk before the atomic rename.
    with temp_path.open("rb") as stream:
        os.fsync(stream.fileno())

    os.replace(temp_path, path)

    validation = load_and_validate_daily_file(
        path,
        trading_date,
        job,
    )
    return validation, True


def cleanup_stale_temp_files(root: Path) -> int:
    if not root.exists():
        return 0

    removed = 0
    for path in root.rglob("*.tmp"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def job_identity(job: DownloadJob) -> dict[str, Any]:
    return {
        "symbol": job.symbol,
        "bar_size": normalize_bar_size(job.bar_size),
        "session": job.session,
        "what_to_show": job.what_to_show,
        "exchange": job.exchange,
        "primary_exchange": job.primary_exchange,
        "security_type": job.security_type,
        "currency": job.currency,
        "calendar": job.calendar,
    }


def save_progress_state(
    job: DownloadJob,
    output_root: Path,
    requested_dates: list[date],
    complete_dates: set[date],
) -> None:
    remaining = [
        day for day in requested_dates if day not in complete_dates
    ]

    payload = {
        "version": 3,
        "checkpoint_role": (
            "progress_hint_only_filesystem_daily_files_are_authoritative"
        ),
        "job": job_identity(job),
        "requested_start_date": (
            requested_dates[0].isoformat()
            if requested_dates
            else None
        ),
        "requested_end_date": (
            requested_dates[-1].isoformat()
            if requested_dates
            else None
        ),
        "expected_trading_dates": len(requested_dates),
        "valid_daily_files": len(complete_dates),
        "remaining_dates": len(remaining),
        "last_completed_trading_date": (
            max(complete_dates).isoformat()
            if complete_dates
            else None
        ),
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    write_json_atomic(
        job_checkpoint_path(job, output_root),
        payload,
    )


def scan_existing_daily_files(
    job: DownloadJob,
    output_root: Path,
    requested_dates: list[date],
) -> dict[date, DailyValidation]:
    """
    Rescan disk on every run.

    This function is the resume mechanism. checkpoint.json is not consulted.
    """
    valid: dict[date, DailyValidation] = {}

    for trading_date in requested_dates:
        path = daily_file_path(
            job,
            output_root,
            trading_date,
        )

        if not path.exists():
            continue

        try:
            validation = load_and_validate_daily_file(
                path,
                trading_date,
                job,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Existing raw daily file is invalid: {path}\n"
                f"Reason: {exc}\n"
                "The scraper will not overwrite it automatically. "
                "Inspect/remove the bad file explicitly, then rerun."
            ) from exc

        valid[trading_date] = validation

    return valid


def request_end_datetime(trading_date: date) -> str:
    # Explicit named timezone avoids the old YYYYMMDD-HH:mm:ss UTC/DST bug.
    return (
        f"{trading_date:%Y%m%d} "
        f"23:59:59 {IB_REQUEST_TIMEZONE}"
    )


def format_date_mode(bar_size: str) -> int:
    # Intraday epoch seconds remove Gateway/TWS login-timezone ambiguity.
    return 2 if is_intraday_bar_size(bar_size) else 1


class IBClient(EWrapper, EClient, APPENV):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        APPENV.__init__(self)

        self.connected_event = threading.Event()

        self.historical_data: dict[int, list[dict[str, Any]]] = {}
        self.request_events: dict[int, threading.Event] = {}
        self.request_errors: dict[int, str] = {}

        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._request_id = int(time.time()) % 1_000_000

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        print(f"IB connected. Next valid order ID: {orderId}")
        self.connected_event.set()

    def error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        if errorCode in INFO_ERROR_CODES:
            print(f"IB INFO {errorCode}: {errorString}")
            return

        print(
            f"IB ERROR request={reqId} "
            f"code={errorCode}: {errorString}"
        )

        with self._state_lock:
            event = self.request_events.get(reqId)
            if reqId >= 0 and event is not None:
                self.request_errors[reqId] = (
                    f"{errorCode}: {errorString}"
                )
                event.set()

    def connect_ib(self) -> None:
        print(f"Connecting IB {self.host}:{self.port}")
        self.connect(
            self.host,
            self.port,
            self.client_id,
        )

        threading.Thread(
            target=self.run,
            daemon=True,
            name="ib-api-loop",
        ).start()

        if not self.connected_event.wait(timeout=10):
            self.disconnect()
            raise ConnectionError("IB API handshake failed")

        print("IB API ready")

    def next_request_id(self) -> int:
        with self._state_lock:
            request_id = self._request_id
            self._request_id += 1
            return request_id

    @staticmethod
    def stock_contract(job: DownloadJob) -> Contract:
        contract = Contract()
        contract.symbol = job.symbol
        contract.secType = job.security_type
        contract.currency = job.currency

        if job.exchange:
            contract.exchange = job.exchange
        elif job.symbol in {"BIT", "BITCOIN"}:
            contract.exchange = "PAXOS"
        elif job.session == "overnight":
            contract.exchange = "OVERNIGHT"
        else:
            contract.exchange = "SMART"

        if job.primary_exchange:
            contract.primaryExchange = job.primary_exchange
        elif job.symbol == "VIX":
            contract.primaryExchange = "CBOE"

        return contract

    def historicalData(
        self,
        reqId: int,
        bar: BarData,
    ) -> None:  # noqa: N802
        wap = getattr(
            bar,
            "wap",
            getattr(bar, "average", None),
        )
        bar_count = getattr(
            bar,
            "barCount",
            getattr(bar, "count", None),
        )

        with self._state_lock:
            if reqId not in self.historical_data:
                return

            self.historical_data[reqId].append(
                {
                    "datetime": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": float(bar.volume),
                    "wap": (
                        float(wap)
                        if wap is not None
                        else None
                    ),
                    "bar_count": (
                        int(bar_count)
                        if bar_count is not None
                        else None
                    ),
                }
            )

    def historicalDataEnd(
        self,
        reqId: int,
        start: str,
        end: str,
    ) -> None:  # noqa: N802
        with self._state_lock:
            event = self.request_events.get(reqId)

        if event:
            event.set()

    def _cleanup_request(self, req_id: int) -> None:
        with self._state_lock:
            self.historical_data.pop(req_id, None)
            self.request_events.pop(req_id, None)
            self.request_errors.pop(req_id, None)

    def request_one_trading_day(
        self,
        job: DownloadJob,
        trading_date: date,
        timeout_seconds: int,
    ) -> pd.DataFrame:
        req_id = self.next_request_id()
        event = threading.Event()

        with self._state_lock:
            self.historical_data[req_id] = []
            self.request_events[req_id] = event

        end_datetime = request_end_datetime(trading_date)
        normalized_bar_size = normalize_bar_size(job.bar_size)

        print(
            f"[{job.symbol}] request {req_id}: "
            f"{trading_date} ending '{end_datetime}'"
        )

        with self._send_lock:
            self.reqHistoricalData(
                reqId=req_id,
                contract=self.stock_contract(job),
                endDateTime=end_datetime,
                durationStr="1 D",
                barSizeSetting=normalized_bar_size,
                whatToShow=job.what_to_show,
                useRTH=1 if job.session == "rth" else 0,
                formatDate=format_date_mode(
                    normalized_bar_size
                ),
                keepUpToDate=False,
                chartOptions=[],
            )

        if not event.wait(timeout=timeout_seconds):
            self.cancelHistoricalData(req_id)
            self._cleanup_request(req_id)
            raise TimeoutError(
                f"{job.symbol} {trading_date}: "
                f"request {req_id} timed out after "
                f"{timeout_seconds}s"
            )

        with self._state_lock:
            rows = list(
                self.historical_data.get(req_id, [])
            )
            request_error = self.request_errors.get(req_id)

        self._cleanup_request(req_id)

        if request_error:
            raise RuntimeError(
                f"{job.symbol} {trading_date}: "
                f"IB request failed ({request_error})"
            )

        if not rows:
            raise RuntimeError(
                f"{job.symbol} {trading_date}: "
                "IB returned no historical rows for an "
                f"{job.calendar} trading date."
            )

        chunk = prepare_chunk(pd.DataFrame(rows))
        validate_daily_frame(
            chunk,
            trading_date,
            job,
        )
        return chunk

    def download_job(
        self,
        job: DownloadJob,
        output_root: Path,
        request_pause_seconds: float,
        timeout_seconds: int,
    ) -> DownloadResult:
        storage_root = job_storage_root(
            job,
            output_root,
        )
        storage_root.mkdir(parents=True, exist_ok=True)

        removed_temps = cleanup_stale_temp_files(storage_root)
        if removed_temps:
            print(
                f"[{job.symbol}] removed "
                f"{removed_temps} stale temp file(s)"
            )

        requested_dates = trading_dates_for_job(job)
        if not requested_dates:
            raise RuntimeError(
                f"{job.symbol}: requested range contains "
                f"no {job.calendar} trading dates."
            )

        start_date, end_date = resolve_date_range(job)
        print(
            f"[{job.symbol}] requested calendar range "
            f"{start_date} -> {end_date}; "
            f"{len(requested_dates)} exchange sessions"
        )

        # Authoritative resume step: scan the actual daily raw files.
        existing = scan_existing_daily_files(
            job,
            output_root,
            requested_dates,
        )
        complete_dates = set(existing)

        save_progress_state(
            job,
            output_root,
            requested_dates,
            complete_dates,
        )

        missing_dates = [
            day
            for day in requested_dates
            if day not in complete_dates
        ]

        print(
            f"[{job.symbol}] valid daily files={len(existing)}, "
            f"need_download={len(missing_dates)}"
        )

        downloaded = 0

        # Newest to oldest keeps behavior familiar and tends to be useful when
        # a long bootstrap is interrupted intentionally.
        for trading_date in reversed(missing_dates):
            chunk = self.request_one_trading_day(
                job,
                trading_date,
                timeout_seconds,
            )

            path = daily_file_path(
                job,
                output_root,
                trading_date,
            )

            validation, created = write_daily_frame_atomic(
                chunk,
                path,
                trading_date,
                job,
            )

            if not created:
                # Normally impossible because the pre-scan already found all
                # files. This protects against another process writing the same
                # day between scan and commit.
                print(
                    f"[{job.symbol}] race-safe skip "
                    f"{trading_date}: already present"
                )
            else:
                downloaded += 1
                complete_dates.add(trading_date)

                append_manifest(
                    manifest_path(job, output_root),
                    {
                        "event": "download",
                        "timestamp_utc": (
                            pd.Timestamp.now(tz="UTC").isoformat()
                        ),
                        "job": job_identity(job),
                        "trading_date": trading_date.isoformat(),
                        "request_end_datetime": (
                            request_end_datetime(trading_date)
                        ),
                        "duration_str": "1 D",
                        "path": str(validation.path.resolve()),
                        "sha256": validation.sha256,
                        "row_count": validation.row_count,
                        "first_datetime": (
                            validation.first_datetime.isoformat()
                        ),
                        "last_datetime": (
                            validation.last_datetime.isoformat()
                        ),
                    },
                )

                print(
                    f"[{job.symbol}] saved {trading_date}: "
                    f"{validation.row_count} rows "
                    f"({validation.first_datetime} -> "
                    f"{validation.last_datetime})"
                )

            # checkpoint is audit/progress only; if this write is lost, the
            # next run still rescans the actual CSV files.
            save_progress_state(
                job,
                output_root,
                requested_dates,
                complete_dates,
            )

            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)

        print(
            f"[{job.symbol}] complete: "
            f"requested={len(requested_dates)}, "
            f"already_present={len(existing)}, "
            f"downloaded={downloaded}"
        )

        return DownloadResult(
            symbol=job.symbol,
            requested_dates=len(requested_dates),
            already_present=len(existing),
            downloaded=downloaded,
            output_directory=storage_root,
        )


def import_existing_csv(
    job: DownloadJob,
    output_root: Path,
    csv_path: Path,
) -> int:
    """
    Seed the date-addressed raw store from an already-corrected combined CSV.

    This is useful for the corrected Aug-2025 -> Aug-2026 AVGO file so a later
    overlapping request can see those days on disk and skip them.
    """
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print(
        f"[{job.symbol}] importing existing CSV into daily raw store: "
        f"{csv_path}"
    )

    combined = prepare_chunk(pd.read_csv(csv_path))
    if combined.empty:
        raise RuntimeError(f"{csv_path} contains no valid rows.")

    imported = 0
    skipped_identical = 0

    combined["_calendar_date"] = combined["datetime"].dt.date

    for trading_date, group in combined.groupby(
        "_calendar_date",
        sort=True,
    ):
        frame = group.drop(
            columns=["_calendar_date"]
        ).reset_index(drop=True)

        path = daily_file_path(
            job,
            output_root,
            trading_date,
        )

        validation, created = write_daily_frame_atomic(
            frame,
            path,
            trading_date,
            job,
        )

        if created:
            imported += 1
            event = "import"
        else:
            skipped_identical += 1
            event = "import_skip_identical"

        append_manifest(
            manifest_path(job, output_root),
            {
                "event": event,
                "timestamp_utc": (
                    pd.Timestamp.now(tz="UTC").isoformat()
                ),
                "job": job_identity(job),
                "trading_date": trading_date.isoformat(),
                "source_csv": str(csv_path),
                "path": str(validation.path.resolve()),
                "sha256": validation.sha256,
                "row_count": validation.row_count,
                "first_datetime": (
                    validation.first_datetime.isoformat()
                ),
                "last_datetime": (
                    validation.last_datetime.isoformat()
                ),
            },
        )

    print(
        f"[{job.symbol}] import complete: "
        f"created={imported}, "
        f"already_identical={skipped_identical}"
    )
    return imported


def all_daily_files(
    job: DownloadJob,
    output_root: Path,
) -> list[Path]:
    root = job_storage_root(job, output_root)
    if not root.exists():
        return []

    files: list[Path] = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        if not re.fullmatch(r"\d{4}", year_dir.name):
            continue

        files.extend(sorted(year_dir.glob("????-??-??.csv")))

    return sorted(files)


def build_snapshot(
    job: DownloadJob,
    output_root: Path,
    output_path: Path,
) -> int:
    """
    Build one deterministic combined snapshot from all stored daily raw files.

    The daily files remain authoritative. The snapshot is replaceable and can
    always be regenerated.
    """
    paths = all_daily_files(job, output_root)
    if not paths:
        raise RuntimeError(
            f"{job.symbol}: no daily raw files found to assemble."
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        trading_date = date.fromisoformat(path.stem)
        frame = prepare_chunk(pd.read_csv(path))
        validate_daily_frame(frame, trading_date, job)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    duplicate_rows = combined[
        combined.duplicated("datetime", keep=False)
    ]
    if not duplicate_rows.empty:
        conflicting = (
            duplicate_rows
            .groupby("datetime")[RAW_COLUMNS[1:]]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicting.any():
            bad_times = conflicting[
                conflicting
            ].index[:5].tolist()
            raise RuntimeError(
                f"{job.symbol}: conflicting timestamps while assembling, "
                f"examples={bad_times}"
            )

        combined = combined.drop_duplicates(
            "datetime",
            keep="first",
        )

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    combined.to_csv(temp_path, index=False)

    with temp_path.open("rb") as stream:
        os.fsync(stream.fileno())

    os.replace(temp_path, output_path)

    print(
        f"[{job.symbol}] snapshot {len(combined)} rows "
        f"from {len(paths)} daily files -> {output_path}"
    )
    return len(combined)


def _download_and_snapshot(
    app: IBClient,
    job: DownloadJob,
    runtime: RuntimeConfig,
) -> DownloadResult:
    result = app.download_job(
        job,
        runtime.output_root,
        runtime.request_pause_seconds,
        runtime.timeout_seconds,
    )

    if runtime.auto_snapshot:
        snapshot_path = snapshot_output_path(
            job,
            runtime.snapshot_root,
        )
        build_snapshot(
            job,
            runtime.output_root,
            snapshot_path,
        )

    return result


def rebuild_all_snapshots(runtime: RuntimeConfig) -> int:
    """Offline maintenance: regenerate deterministic snapshots for all jobs."""
    failures = 0

    for job in runtime.jobs:
        try:
            path = snapshot_output_path(
                job,
                runtime.snapshot_root,
            )
            build_snapshot(
                job,
                runtime.output_root,
                path,
            )
        except Exception as exc:
            failures += 1
            print(
                f"[{job.symbol}] SNAPSHOT FAILED: {exc}"
            )

    return failures


def run_batch(
    app: IBClient,
    runtime: RuntimeConfig,
) -> int:
    failures = 0

    with ThreadPoolExecutor(
        max_workers=min(
            runtime.max_workers,
            len(runtime.jobs),
        ),
        thread_name_prefix="ib-download",
    ) as pool:
        futures = {
            pool.submit(
                _download_and_snapshot,
                app,
                job,
                runtime,
            ): job
            for job in runtime.jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures += 1
                print(
                    f"[{job.symbol}] FAILED: {exc}"
                )

    return failures

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download IBKR historical data into an idempotent "
            "date-addressed raw store."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML batch configuration file.",
    )

    parser.add_argument(
        "-s",
        "--symbol",
        help="Single-symbol mode.",
    )
    parser.add_argument(
        "--session",
        choices=["rth", "extended", "overnight"],
        default=None,
    )
    parser.add_argument(
        "-o",
        "--overnight",
        type=parse_bool,
        default=False,
        help="Legacy alias for --session overnight.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        default="365 D",
    )
    parser.add_argument(
        "-bs",
        "--bar-size",
        "--bar_size",
        dest="bar_size",
        default="1 hour",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive YYYY-MM-DD.",
    )
    parser.add_argument(
        "--calendar",
        default=DEFAULT_CALENDAR,
    )
    parser.add_argument(
        "--group",
        default="ungrouped",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override authoritative daily raw root. "
            "Default: <BASE_DIR>/data/raw/ibkr"
        ),
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=None,
        help=(
            "Override generated combined-snapshot root. "
            "Default: <BASE_DIR>/data/raw/bootstrap"
        ),
    )

    parser.add_argument(
        "--import-csv",
        type=Path,
        default=None,
        help=(
            "Seed the V3 daily store from one corrected combined CSV. "
            "Requires exactly one configured job and does not connect to IB."
        ),
    )
    parser.add_argument(
        "--rebuild-snapshots",
        action="store_true",
        help=(
            "Offline maintenance: regenerate the automatically named "
            "combined snapshot for every configured unique dataset."
        ),
    )

    return parser.parse_args()


def runtime_from_args(args: argparse.Namespace) -> RuntimeConfig:
    if args.config is not None:
        runtime = load_jobs(args.config)

        if args.output_root is not None:
            output_root = args.output_root
            if not output_root.is_absolute():
                output_root = Path(BASE_DIR) / output_root

            runtime = RuntimeConfig(
                jobs=runtime.jobs,
                max_workers=runtime.max_workers,
                output_root=output_root,
                snapshot_root=runtime.snapshot_root,
                auto_snapshot=runtime.auto_snapshot,
                request_pause_seconds=runtime.request_pause_seconds,
                timeout_seconds=runtime.timeout_seconds,
            )

        if args.snapshot_root is not None:
            snapshot_root = args.snapshot_root
            if not snapshot_root.is_absolute():
                snapshot_root = Path(BASE_DIR) / snapshot_root

            runtime = RuntimeConfig(
                jobs=runtime.jobs,
                max_workers=runtime.max_workers,
                output_root=runtime.output_root,
                snapshot_root=snapshot_root,
                auto_snapshot=runtime.auto_snapshot,
                request_pause_seconds=runtime.request_pause_seconds,
                timeout_seconds=runtime.timeout_seconds,
            )

        return runtime

    if not args.symbol:
        default_config = (
            Path(BASE_DIR)
            / "config"
            / "scraper-config-semicond.yml"
        )
        return load_jobs(default_config)

    if args.session is not None:
        session = args.session
    elif args.overnight:
        session = "overnight"
    else:
        session = "rth"

    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(BASE_DIR) / "data" / "raw" / "ibkr"
    )
    if not output_root.is_absolute():
        output_root = Path(BASE_DIR) / output_root

    snapshot_root = (
        args.snapshot_root
        if args.snapshot_root is not None
        else Path(BASE_DIR) / "data" / "raw" / "bootstrap"
    )
    if not snapshot_root.is_absolute():
        snapshot_root = Path(BASE_DIR) / snapshot_root

    job = DownloadJob(
        symbol=args.symbol.upper(),
        group=args.group,
        duration=args.duration,
        bar_size=args.bar_size,
        session=session,
        start_date=args.start_date,
        end_date=args.end_date,
        calendar=args.calendar,
    )

    return RuntimeConfig(
        jobs=[job],
        max_workers=max(1, args.workers),
        output_root=output_root,
        snapshot_root=snapshot_root,
        auto_snapshot=True,
        request_pause_seconds=1.5,
        timeout_seconds=120,
    )


def require_single_job(
    runtime: RuntimeConfig,
    operation: str,
) -> DownloadJob:
    if len(runtime.jobs) != 1:
        raise ValueError(
            f"{operation} requires exactly one configured job; "
            f"found {len(runtime.jobs)}."
        )
    return runtime.jobs[0]


def main() -> int:
    args = parse_args()
    runtime = runtime_from_args(args)

    # Offline migration: split the already-corrected combined CSV into the
    # authoritative daily raw store.
    if args.import_csv is not None:
        job = require_single_job(
            runtime,
            "--import-csv",
        )
        import_existing_csv(
            job,
            runtime.output_root,
            args.import_csv,
        )

        if runtime.auto_snapshot:
            build_snapshot(
                job,
                runtime.output_root,
                snapshot_output_path(
                    job,
                    runtime.snapshot_root,
                ),
            )
        return 0

    # Offline maintenance only. Normal download/import runs build snapshots
    # automatically and do not require a user-supplied output filename.
    if args.rebuild_snapshots:
        failures = rebuild_all_snapshots(runtime)
        return 1 if failures else 0

    app = IBClient()

    try:
        app.connect_ib()
        failures = run_batch(app, runtime)
        print(
            f"Completed: "
            f"{len(runtime.jobs) - failures} succeeded, "
            f"{failures} failed"
        )
        return 1 if failures else 0
    finally:
        if app.isConnected():
            app.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
