from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


OHLCV = ["open", "high", "low", "close", "volume"]
MICROSTRUCTURE_COLUMNS = ["wap", "bar_count"]
SOURCE_VALUE_COLUMNS = [*OHLCV, *MICROSTRUCTURE_COLUMNS]
REQUIRED_SOURCE_COLUMNS = ["datetime", *OHLCV]

CANONICAL_COLUMNS = [
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
]

DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_CALENDAR = "XNYS"
DEFAULT_FREQUENCY = "1min"
DEFAULT_BAR_LABEL = "start"


@dataclass(frozen=True)
class SessionDefinition:
    name: str
    start: str
    end_exclusive: str
    crosses_midnight: bool


@dataclass(frozen=True)
class PipelineConfig:
    timezone: str
    calendar: str
    frequency: str
    bar_label: str
    sessions: dict[str, SessionDefinition]
    fill_internal_gaps: bool
    max_internal_gap_minutes: int
    fill_leading_gaps: bool
    fill_trailing_gaps: bool
    fill_fully_missing_sessions: bool
    synthetic_volume: float
    parquet_compression: str


def load_config(path: Path) -> PipelineConfig:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(
        path.read_text(encoding="utf-8")
    ) or {}

    raw_sessions = raw.get("sessions") or {}
    if not raw_sessions:
        raise ValueError(
            f"{path} does not define any sessions."
        )

    sessions = {
        str(name).strip().lower(): SessionDefinition(
            name=str(name).strip().lower(),
            start=str(values["start"]),
            end_exclusive=str(
                values["end_exclusive"]
            ),
            crosses_midnight=bool(
                values.get(
                    "crosses_midnight",
                    False,
                )
            ),
        )
        for name, values in raw_sessions.items()
    }

    source_cfg = raw.get("source") or {}
    regularization_cfg = (
        raw.get("regularization") or {}
    )
    parquet_cfg = raw.get("parquet") or {}

    return PipelineConfig(
        timezone=str(
            raw.get(
                "timezone",
                DEFAULT_TIMEZONE,
            )
        ),
        calendar=str(
            raw.get(
                "calendar",
                DEFAULT_CALENDAR,
            )
        ),
        frequency=str(
            source_cfg.get(
                "frequency",
                raw.get(
                    "frequency",
                    DEFAULT_FREQUENCY,
                ),
            )
        ),
        bar_label=str(
            source_cfg.get(
                "bar_label",
                raw.get(
                    "bar_label",
                    DEFAULT_BAR_LABEL,
                ),
            )
        ),
        sessions=sessions,
        fill_internal_gaps=bool(
            regularization_cfg.get(
                "fill_internal_gaps",
                True,
            )
        ),
        max_internal_gap_minutes=int(
            regularization_cfg.get(
                "max_internal_gap_minutes",
                5,
            )
        ),
        fill_leading_gaps=bool(
            regularization_cfg.get(
                "fill_leading_gaps",
                False,
            )
        ),
        fill_trailing_gaps=bool(
            regularization_cfg.get(
                "fill_trailing_gaps",
                False,
            )
        ),
        fill_fully_missing_sessions=bool(
            regularization_cfg.get(
                "fill_fully_missing_sessions",
                False,
            )
        ),
        synthetic_volume=float(
            regularization_cfg.get(
                "synthetic_volume",
                0.0,
            )
        ),
        parquet_compression=str(
            parquet_cfg.get(
                "compression",
                "zstd",
            )
        ),
    )


def load_instrument(
    path: Path,
    symbol: str,
) -> dict:
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "symbol" not in df.columns:
        raise ValueError(
            f"{path} is missing a symbol column."
        )

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    match = df.loc[
        df["symbol"].eq(
            symbol.strip().upper()
        )
    ]
    if match.empty:
        raise ValueError(
            f"{symbol.upper()} is not defined "
            f"in {path}"
        )

    return match.iloc[0].to_dict()


def load_source(
    path: Path,
    source_id: str,
    symbol: str,
) -> dict:
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required = {
        "symbol",
        "source_id",
    }
    missing = required.difference(
        df.columns
    )
    if missing:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{sorted(missing)}"
        )

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["source_id"] = (
        df["source_id"]
        .astype(str)
        .str.strip()
    )

    match = df.loc[
        df["source_id"].eq(
            str(source_id).strip()
        )
        & df["symbol"].eq(
            symbol.strip().upper()
        )
    ]

    if match.empty:
        raise ValueError(
            "No source config for "
            f"source_id={source_id!r}, "
            f"symbol={symbol.upper()!r}"
        )

    row = match.iloc[0].to_dict()

    raw_sessions = row.get(
        "coverage_sessions"
    )
    if raw_sessions is None or (
        isinstance(
            raw_sessions,
            float,
        )
        and pd.isna(raw_sessions)
    ):
        row["coverage_sessions"] = [
            "premarket",
            "regular",
            "aftermarket",
        ]
    else:
        text = str(
            raw_sessions
        ).strip()

        separator = "|"
        for candidate in (
            "|",
            ",",
            ";",
        ):
            if candidate in text:
                separator = candidate
                break

        row["coverage_sessions"] = [
            value.strip().lower()
            for value in text.split(
                separator
            )
            if value.strip()
        ]

    return row


def parse_market_timestamp(
    value: Any,
    timezone: str,
) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize(
            timezone,
            ambiguous="raise",
            nonexistent="raise",
        )

    return ts.tz_convert(timezone)


def _optional_market_timestamp(
    value: Any,
    timezone: str,
) -> pd.Timestamp | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {
        "nan",
        "nat",
        "none",
    }:
        return None

    return parse_market_timestamp(
        value,
        timezone,
    )


def effective_coverage_start(
    instrument: dict,
    source: dict,
    timezone: str,
) -> pd.Timestamp:
    """
    Use the later of instrument Big Bang and source coverage start.

    If one metadata value is absent, use the one that is present.
    """
    instrument_start = (
        _optional_market_timestamp(
            instrument.get("bigbang"),
            timezone,
        )
    )
    source_start = (
        _optional_market_timestamp(
            source.get(
                "coverage_start"
            ),
            timezone,
        )
    )

    starts = [
        value
        for value in (
            instrument_start,
            source_start,
        )
        if value is not None
    ]

    if not starts:
        raise ValueError(
            "Neither instrument.bigbang nor "
            "source.coverage_start defines "
            "a usable coverage boundary."
        )

    return max(starts)


def _parse_datetime_series(
    values: pd.Series,
    timezone: str,
) -> pd.Series:
    """
    Return one timezone-aware market-time Series without mixed-DST warnings.

    Raw corrected IBKR CSVs use naive New York wall time.
    Canonical CSVs use explicit offsets such as -04:00 and -05:00.
    """
    text = values.astype("string").str.strip()

    explicit_offset = text.str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})$",
        case=False,
        regex=True,
        na=False,
    )

    if explicit_offset.any() and not explicit_offset.all():
        raise ValueError(
            "Datetime column mixes timezone-aware and naive values."
        )

    if explicit_offset.all():
        parsed = pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        return parsed.dt.tz_convert(
            timezone
        )

    parsed = pd.to_datetime(
        values,
        errors="coerce",
    )

    if pd.api.types.is_datetime64_dtype(
        parsed.dtype
    ):
        return parsed.dt.tz_localize(
            timezone,
            ambiguous="raise",
            nonexistent="raise",
        )

    result = []

    for value in values:
        try:
            ts = pd.Timestamp(value)

            if ts.tzinfo is not None:
                raise ValueError(
                    "Unexpected timezone-aware value "
                    "inside a naive datetime column."
                )

            ts = ts.tz_localize(
                timezone,
                ambiguous="raise",
                nonexistent="raise",
            )
            result.append(ts)

        except Exception:
            result.append(pd.NaT)

    return pd.Series(
        result,
        index=values.index,
    )


def load_ohlcv_csv(
    path: Path,
    timezone: str,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Load OHLCV plus optional IBKR wap/bar_count.

    wap and bar_count are a pair: both must be present or both absent.
    This function parses only; it never fills or modifies market values.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)
    raw.columns = (
        raw.columns
        .str.strip()
        .str.lower()
    )

    missing = [
        column
        for column
        in REQUIRED_SOURCE_COLUMNS
        if column not in raw.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{missing}"
        )

    has_wap = "wap" in raw.columns
    has_bar_count = (
        "bar_count" in raw.columns
    )

    if has_wap != has_bar_count:
        raise ValueError(
            "wap and bar_count must either "
            "both be present or both be absent."
        )

    metrics: dict[str, Any] = {
        "input_rows": int(len(raw)),
        "microstructure_available": bool(
            has_wap
            and has_bar_count
        ),
    }

    parsed = _parse_datetime_series(
        raw["datetime"],
        timezone,
    )
    metrics[
        "invalid_datetime_rows"
    ] = int(
        parsed.isna().sum()
    )

    valid = parsed.notna()

    df = raw.loc[
        valid
    ].copy()
    df["datetime"] = parsed.loc[
        valid
    ]

    numeric_columns = list(
        OHLCV
    )
    if has_wap and has_bar_count:
        numeric_columns += (
            MICROSTRUCTURE_COLUMNS
        )

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )
        metrics[
            f"invalid_{column}_rows"
        ] = int(
            values.isna().sum()
        )
        df[column] = values

    return (
        df
        .sort_values("datetime")
        .reset_index(drop=True),
        metrics,
    )


def _minutes(
    value: str,
) -> int:
    hour, minute = map(
        int,
        value.split(":"),
    )
    return (
        hour * 60
        + minute
    )


def _timedelta(
    value: str,
) -> pd.Timedelta:
    hour, minute = map(
        int,
        value.split(":"),
    )
    return pd.Timedelta(
        hours=hour,
        minutes=minute,
    )


def _calendar():
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise RuntimeError(
            "exchange-calendars is required "
            "for exchange-session calendar logic."
        ) from exc

    return xcals


@lru_cache(maxsize=16)
def get_exchange_calendar(
    calendar_name: str,
):
    return _calendar().get_calendar(
        calendar_name
    )


def _naive_normalized_date(
    value: Any,
) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts.normalize()


def exchange_session_dates(
    calendar_name: str,
    start_date: Any,
    end_date: Any,
) -> pd.DatetimeIndex:
    calendar = get_exchange_calendar(
        calendar_name
    )

    start = _naive_normalized_date(
        start_date
    )
    end = _naive_normalized_date(
        end_date
    )

    return calendar.sessions_in_range(
        start,
        end,
    )


def _clock_on_date(
    day: Any,
    clock_text: str,
    timezone: str,
) -> pd.Timestamp:
    base = _naive_normalized_date(
        day
    )

    return (
        base.tz_localize(timezone)
        + _timedelta(clock_text)
    )


def session_windows_for_trading_date(
    trading_date: Any,
    config: PipelineConfig,
) -> tuple[
    dict[
        str,
        tuple[
            pd.Timestamp,
            pd.Timestamp,
        ],
    ],
    bool,
]:
    """
    Calendar-aware windows for one exchange trading date.

    Premarket:
        configured start -> actual exchange open

    Regular:
        actual exchange open -> actual exchange close

    Aftermarket:
        actual exchange close -> nominal configured end on normal days.
        On an early close, preserve the configured post-close duration.
        Example: normal postmarket 16:00-20:00 and 13:00 close -> 13:00-17:00.

    Overnight:
        previous calendar date configured start -> trading-date configured end.
    """
    trading_date = (
        _naive_normalized_date(
            trading_date
        )
    )

    calendar = get_exchange_calendar(
        config.calendar
    )

    labels = calendar.sessions_in_range(
        trading_date,
        trading_date,
    )

    if len(labels) != 1:
        raise ValueError(
            f"{trading_date.date()} "
            f"is not a {config.calendar} "
            "trading session."
        )

    session_label = labels[0]
    schedule_row = (
        calendar.schedule.loc[
            session_label
        ]
    )

    exchange_open = (
        schedule_row["open"]
        .tz_convert(
            config.timezone
        )
    )
    exchange_close = (
        schedule_row["close"]
        .tz_convert(
            config.timezone
        )
    )

    windows: dict[
        str,
        tuple[
            pd.Timestamp,
            pd.Timestamp,
        ],
    ] = {}

    regular_cfg = (
        config.sessions.get(
            "regular"
        )
    )
    premarket_cfg = (
        config.sessions.get(
            "premarket"
        )
    )
    aftermarket_cfg = (
        config.sessions.get(
            "aftermarket"
        )
    )

    if regular_cfg is not None:
        windows["regular"] = (
            exchange_open,
            exchange_close,
        )

    if premarket_cfg is not None:
        premarket_start = (
            _clock_on_date(
                trading_date,
                premarket_cfg.start,
                config.timezone,
            )
        )

        windows["premarket"] = (
            premarket_start,
            exchange_open,
        )

    is_early_close = False

    if aftermarket_cfg is not None:
        if regular_cfg is None:
            raise ValueError(
                "aftermarket requires a "
                "regular session definition."
            )

        nominal_regular_close = (
            _clock_on_date(
                trading_date,
                regular_cfg.end_exclusive,
                config.timezone,
            )
        )
        nominal_aftermarket_end = (
            _clock_on_date(
                trading_date,
                aftermarket_cfg.end_exclusive,
                config.timezone,
            )
        )

        nominal_post_duration = (
            nominal_aftermarket_end
            - nominal_regular_close
        )

        is_early_close = (
            exchange_close
            < nominal_regular_close
        )

        if is_early_close:
            aftermarket_end = (
                exchange_close
                + nominal_post_duration
            )
        else:
            aftermarket_end = (
                nominal_aftermarket_end
            )

        windows["aftermarket"] = (
            exchange_close,
            aftermarket_end,
        )

    overnight_cfg = (
        config.sessions.get(
            "overnight"
        )
    )

    if overnight_cfg is not None:
        if not (
            overnight_cfg
            .crosses_midnight
        ):
            raise ValueError(
                "overnight session must have "
                "crosses_midnight: true"
            )

        previous_day = (
            trading_date
            - pd.Timedelta(days=1)
        )

        overnight_start = (
            _clock_on_date(
                previous_day,
                overnight_cfg.start,
                config.timezone,
            )
        )
        overnight_end = (
            _clock_on_date(
                trading_date,
                overnight_cfg.end_exclusive,
                config.timezone,
            )
        )

        windows["overnight"] = (
            overnight_start,
            overnight_end,
        )

    return (
        windows,
        bool(is_early_close),
    )


def next_exchange_session_date(
    date: Any,
    calendar_name: str,
    *,
    strictly_after: bool,
):
    date = _naive_normalized_date(
        date
    )

    start = (
        date
        + pd.Timedelta(days=1)
        if strictly_after
        else date
    )
    end = (
        start
        + pd.Timedelta(days=14)
    )

    sessions = exchange_session_dates(
        calendar_name,
        start,
        end,
    )

    if len(sessions) == 0:
        raise RuntimeError(
            f"No {calendar_name} session "
            f"found near {date.date()}"
        )

    return pd.Timestamp(
        sessions[0]
    ).date()


def trading_date_for_timestamp(
    timestamp: pd.Timestamp,
    session: str,
    config: PipelineConfig,
):
    """
    P/R/A -> local calendar date.

    Overnight:
        20:00-23:59 -> next exchange trading date.
        00:00-03:59 -> exchange trading date on/after that calendar date.
    """
    local = parse_market_timestamp(
        timestamp,
        config.timezone,
    )

    if session != "overnight":
        return local.date()

    overnight_cfg = (
        config.sessions.get(
            "overnight"
        )
    )
    if overnight_cfg is None:
        raise ValueError(
            "Timestamp is overnight but no "
            "overnight session is configured."
        )

    start_minutes = _minutes(
        overnight_cfg.start
    )
    minute = (
        local.hour * 60
        + local.minute
    )

    return next_exchange_session_date(
        local.date(),
        config.calendar,
        strictly_after=(
            minute
            >= start_minutes
        ),
    )


def session_for_timestamp(
    timestamp: pd.Timestamp,
    config: PipelineConfig,
) -> str:
    """
    Calendar-aware single-timestamp session lookup.

    For bulk data use add_market_labels(), which labels by trading day
    rather than performing a calendar lookup for every row.
    """
    local = parse_market_timestamp(
        timestamp,
        config.timezone,
    )
    local_date = pd.Timestamp(
        local.date()
    )

    same_day_sessions = (
        exchange_session_dates(
            config.calendar,
            local_date,
            local_date,
        )
    )

    if len(same_day_sessions) == 1:
        windows, _ = (
            session_windows_for_trading_date(
                local_date,
                config,
            )
        )

        for session_name in (
            "premarket",
            "regular",
            "aftermarket",
        ):
            if session_name not in windows:
                continue

            start, end = windows[
                session_name
            ]

            if start <= local < end:
                return session_name

    if "overnight" in config.sessions:
        candidate_trading_date = (
            trading_date_for_timestamp(
                local,
                "overnight",
                config,
            )
        )

        windows, _ = (
            session_windows_for_trading_date(
                candidate_trading_date,
                config,
            )
        )

        if "overnight" in windows:
            start, end = (
                windows["overnight"]
            )

            if start <= local < end:
                return "overnight"

    return "unknown"


def add_market_labels(
    df: pd.DataFrame,
    config: PipelineConfig,
    *,
    source_id: str,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Add calendar_date, trading_date, session and source metadata.

    This bulk implementation uses one calendar-window calculation per
    trading date, so it remains practical for hundreds of thousands of rows.
    """
    result = df.copy()

    if "datetime" not in result.columns:
        raise ValueError(
            "datetime column is required."
        )

    local = _parse_datetime_series(
        result["datetime"],
        config.timezone,
    )

    if local.isna().any():
        raise ValueError(
            "Invalid datetime exists while "
            "adding market labels."
        )

    result["datetime"] = local
    result["calendar_date"] = (
        local.dt.date
    )
    result["session"] = "unknown"
    # trading_date is a logical exchange-session date, not a timestamp.
    # Use object dtype so assigning datetime.date values is type-correct.
    result["trading_date"] = pd.Series(
        [None] * len(result),
        index=result.index,
        dtype="object",
    )

    if result.empty:
        result["source_id"] = (
            str(source_id)
            .strip()
            .lower()
        )
        if symbol is not None:
            result["symbol"] = (
                str(symbol)
                .strip()
                .lower()
            )
        return result

    min_date = pd.Timestamp(
        local.min().date()
    )
    max_date = pd.Timestamp(
        local.max().date()
    )

    # Include one following exchange session so a final evening overnight
    # timestamp can map to its next logical trading date.
    calendar_sessions = (
        exchange_session_dates(
            config.calendar,
            min_date,
            max_date
            + pd.Timedelta(days=14),
        )
    )

    for session_label in calendar_sessions:
        trading_date = pd.Timestamp(
            session_label
        ).date()

        windows, _ = (
            session_windows_for_trading_date(
                session_label,
                config,
            )
        )

        for session_name in (
            "premarket",
            "regular",
            "aftermarket",
            "overnight",
        ):
            if session_name not in windows:
                continue

            start, end = (
                windows[
                    session_name
                ]
            )

            mask = (
                result["datetime"].ge(
                    start
                )
                & result["datetime"].lt(
                    end
                )
            )

            if not mask.any():
                continue

            # Do not let a later pass relabel a row that already belongs to
            # another explicit session.
            mask = (
                mask
                & result["session"].eq(
                    "unknown"
                )
            )

            result.loc[
                mask,
                "session",
            ] = session_name

            result.loc[
                mask,
                "trading_date",
            ] = trading_date

    result["source_id"] = (
        str(source_id)
        .strip()
        .lower()
    )

    if symbol is not None:
        result["symbol"] = (
            str(symbol)
            .strip()
            .lower()
        )

    return result


def expected_index_for_trading_date(
    trading_date: Any,
    config: PipelineConfig,
    covered_sessions: Iterable[str],
    *,
    lower_bound: pd.Timestamp | None = None,
    upper_bound: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """
    Build expected source timestamps using the same calendar-aware windows
    used by add_market_labels(), including shortened exchange sessions.
    """
    if (
        str(config.bar_label)
        .strip()
        .lower()
        != "start"
    ):
        raise ValueError(
            "Current canonical pipeline "
            "expects bar_label='start'."
        )

    windows, _ = (
        session_windows_for_trading_date(
            trading_date,
            config,
        )
    )

    pieces: list[
        pd.DatetimeIndex
    ] = []

    for raw_session_name in (
        covered_sessions
    ):
        session_name = (
            str(raw_session_name)
            .strip()
            .lower()
        )

        if session_name not in windows:
            if session_name not in (
                config.sessions
            ):
                raise ValueError(
                    "Unknown session "
                    f"{session_name!r}"
                )
            continue

        start, end_exclusive = (
            windows[
                session_name
            ]
        )

        if lower_bound is not None:
            lower = (
                parse_market_timestamp(
                    lower_bound,
                    config.timezone,
                )
            )
            start = max(
                start,
                lower,
            )

        if upper_bound is not None:
            upper = (
                parse_market_timestamp(
                    upper_bound,
                    config.timezone,
                )
            )
            end_exclusive = min(
                end_exclusive,
                upper
                + pd.Timedelta(
                    config.frequency
                ),
            )

        if end_exclusive <= start:
            continue

        pieces.append(
            pd.date_range(
                start=start,
                end=end_exclusive,
                freq=config.frequency,
                inclusive="left",
                name="datetime",
            )
        )

    if not pieces:
        return pd.DatetimeIndex(
            [],
            tz=config.timezone,
            name="datetime",
        )

    combined = pieces[0]

    for piece in pieces[1:]:
        combined = combined.append(
            piece
        )

    return (
        combined
        .sort_values()
        .unique()
    )


def split_consecutive(
    index: pd.DatetimeIndex,
    frequency: str,
) -> list[pd.DatetimeIndex]:
    if len(index) == 0:
        return []

    delta = pd.Timedelta(
        frequency
    )
    windows = []
    start = 0

    for position in range(
        1,
        len(index),
    ):
        if (
            index[position]
            - index[
                position - 1
            ]
            != delta
        ):
            windows.append(
                index[
                    start:position
                ]
            )
            start = position

    windows.append(
        index[start:]
    )

    return windows


def sha256_file(
    path: Path,
) -> str:
    import hashlib

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()