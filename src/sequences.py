from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_dataset
import yaml

from common import load_config


SEQUENCE_VERSION = "1.1.0"

# features.py v2.1 deliberately caps exported contiguous_history_minutes at
# 512 and documents 512 as "at least 512". Therefore an endpoint test using
# contiguous_history_minutes >= sequence_length is exact for lengths <= 512.
MAX_SUPPORTED_SEQUENCE_LENGTH = 512

DEFAULT_SEQUENCE_LENGTH = 120
DEFAULT_TARGET_HORIZON = 15
DEFAULT_STRIDE_MINUTES = 1
DEFAULT_SCOPE = "continuous"
DEFAULT_SESSIONS = (
    "premarket",
    "regular",
    "aftermarket",
)
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15

VALID_SCOPES = {
    "continuous",
    "same_session",
}


FEATURE_COLUMNS = [
    "datetime",
    "prediction_time",
    "feature_available_at",
    "calendar_date",
    "trading_date",
    "session",
    "symbol",
    "source_id",
    "is_current_bar_usable",
    "is_contiguous_from_previous",
    "continuity_break_type",
    "contiguous_history_minutes",
]

# Structural columns carry split identity, keys, and timestamps only — no
# target/label VALUES. Safe to write for every split, including TEST, since
# nothing here reveals a feature/target/label outcome.
SEQUENCE_INDEX_STRUCTURAL_COLUMNS = [
    "sample_index",
    "symbol",
    "source_id",
    "sequence_length",
    "target_horizon_minutes",
    "stride_minutes",
    "sequence_scope",
    "window_start_datetime",
    "window_end_datetime",
    "prediction_time",
    "target_realized_at",
    "calendar_date",
    "trading_date",
    "session",
    "split",
    "contiguous_history_minutes",
    "scope_history_minutes",
]

# Target VALUE columns. Must never be written for a TEST-split row in any
# trainer-facing (trainval) artifact.
SEQUENCE_INDEX_VALUE_COLUMNS = [
    "target_return_bps",
    "target_log_return",
    "target_direction",
    "target_same_session",
    "target_same_trading_date",
]

SEQUENCE_INDEX_COLUMNS = (
    SEQUENCE_INDEX_STRUCTURAL_COLUMNS
    + SEQUENCE_INDEX_VALUE_COLUMNS
)


@dataclass(frozen=True)
class SplitPlan:
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    train_last_trading_date: object
    validation_last_trading_date: object
    train_cutoff_prediction_time: pd.Timestamp
    validation_cutoff_prediction_time: pd.Timestamp
    trading_dates: int


@dataclass(frozen=True)
class SequenceSettings:
    sequence_length: int
    target_horizon: int
    stride_minutes: int
    scope: str
    sessions: tuple[str, ...]
    train_fraction: float
    validation_fraction: float

    @property
    def test_fraction(self) -> float:
        return (
            1.0
            - self.train_fraction
            - self.validation_fraction
        )


def _coalesce_setting(
    cli_value,
    yaml_value,
    default_value,
):
    if cli_value is not None:
        return cli_value, "cli"
    if yaml_value is not None:
        return yaml_value, "yaml"
    return default_value, "script_default"


def _normalize_sessions(value) -> tuple[str, ...]:
    if value is None:
        return tuple(DEFAULT_SESSIONS)

    if isinstance(value, str):
        raw_items = [
            item
            for item in re.split(r"[\\s,]+", value)
            if item
        ]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raise ValueError(
            "sequences.sessions must be a list or string."
        )

    normalized = []
    seen = set()

    for item in raw_items:
        session = str(item).strip().lower()
        if not session:
            continue
        if session not in seen:
            normalized.append(session)
            seen.add(session)

    if not normalized:
        raise ValueError(
            "At least one sequence session is required."
        )

    return tuple(normalized)


def load_sequence_yaml(
    path: Path,
) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {path}"
        )

    raw = (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8",
            )
        )
        or {}
    )

    pipeline = raw.get(
        "pipeline",
        {},
    )
    if not isinstance(
        pipeline,
        dict,
    ):
        raise ValueError(
            "pipeline must be a YAML mapping."
        )

    sequences = pipeline.get(
        "sequences",
        {},
    )
    if sequences is None:
        sequences = {}
    if not isinstance(
        sequences,
        dict,
    ):
        raise ValueError(
            "pipeline.sequences must be a YAML mapping."
        )

    split = sequences.get(
        "split",
        {},
    )
    if split is None:
        split = {}
    if not isinstance(
        split,
        dict,
    ):
        raise ValueError(
            "pipeline.sequences.split must be a YAML mapping."
        )

    sequence_length = sequences.get(
        "sequence_length",
        sequences.get(
            "length",
        ),
    )

    values = {
        "sequence_length": sequence_length,
        "target_horizon": sequences.get(
            "target_horizon_minutes"
        ),
        "stride_minutes": sequences.get(
            "stride_minutes"
        ),
        "scope": sequences.get(
            "scope"
        ),
        "sessions": sequences.get(
            "sessions"
        ),
        "train_fraction": split.get(
            "train_fraction"
        ),
        "validation_fraction": split.get(
            "validation_fraction"
        ),
        "test_fraction": split.get(
            "test_fraction"
        ),
    }

    return raw, values


def resolve_sequence_settings(
    args: argparse.Namespace,
    *,
    config_path: Path,
) -> tuple[SequenceSettings, dict[str, str]]:
    raw, yaml_values = (
        load_sequence_yaml(
            config_path
        )
    )

    sources: dict[str, str] = {}

    sequence_length, sources[
        "sequence_length"
    ] = _coalesce_setting(
        getattr(
            args,
            "sequence_length",
            None,
        ),
        yaml_values[
            "sequence_length"
        ],
        DEFAULT_SEQUENCE_LENGTH,
    )

    target_horizon, sources[
        "target_horizon_minutes"
    ] = _coalesce_setting(
        getattr(
            args,
            "target_horizon",
            None,
        ),
        yaml_values[
            "target_horizon"
        ],
        DEFAULT_TARGET_HORIZON,
    )

    stride_minutes, sources[
        "stride_minutes"
    ] = _coalesce_setting(
        getattr(
            args,
            "stride_minutes",
            None,
        ),
        yaml_values[
            "stride_minutes"
        ],
        DEFAULT_STRIDE_MINUTES,
    )

    scope, sources[
        "scope"
    ] = _coalesce_setting(
        getattr(
            args,
            "scope",
            None,
        ),
        yaml_values[
            "scope"
        ],
        DEFAULT_SCOPE,
    )

    sessions_value, sources[
        "sessions"
    ] = _coalesce_setting(
        getattr(
            args,
            "sessions",
            None,
        ),
        yaml_values[
            "sessions"
        ],
        DEFAULT_SESSIONS,
    )

    train_fraction, sources[
        "train_fraction"
    ] = _coalesce_setting(
        getattr(
            args,
            "train_frac",
            None,
        ),
        yaml_values[
            "train_fraction"
        ],
        DEFAULT_TRAIN_FRACTION,
    )

    validation_fraction, sources[
        "validation_fraction"
    ] = _coalesce_setting(
        getattr(
            args,
            "validation_frac",
            None,
        ),
        yaml_values[
            "validation_fraction"
        ],
        DEFAULT_VALIDATION_FRACTION,
    )

    try:
        sequence_length = int(
            sequence_length
        )
        target_horizon = int(
            target_horizon
        )
        stride_minutes = int(
            stride_minutes
        )
        train_fraction = float(
            train_fraction
        )
        validation_fraction = float(
            validation_fraction
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Sequence numeric configuration contains "
            "an invalid value."
        ) from exc

    scope = str(
        scope
    ).strip().lower()
    sessions = _normalize_sessions(
        sessions_value
    )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )
    if (
        sequence_length
        > MAX_SUPPORTED_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"sequence_length={sequence_length} exceeds "
            f"the feature continuity cap of "
            f"{MAX_SUPPORTED_SEQUENCE_LENGTH}."
        )
    if target_horizon < 1:
        raise ValueError(
            "target_horizon_minutes must be >= 1."
        )
    if stride_minutes < 1:
        raise ValueError(
            "stride_minutes must be >= 1."
        )
    if scope not in VALID_SCOPES:
        raise ValueError(
            "scope must be one of: "
            + ", ".join(
                sorted(
                    VALID_SCOPES
                )
            )
        )

    configured_sessions = raw.get(
        "sessions",
        {},
    )
    if not isinstance(
        configured_sessions,
        dict,
    ):
        raise ValueError(
            "Top-level sessions must be a YAML mapping."
        )

    known_sessions = set(
        configured_sessions
    )
    if not known_sessions:
        known_sessions = {
            *DEFAULT_SESSIONS,
            "overnight",
        }

    unknown_sessions = [
        session
        for session in sessions
        if session not in known_sessions
    ]
    if unknown_sessions:
        raise ValueError(
            "Unknown sequence sessions: "
            f"{unknown_sessions}. "
            f"Known sessions: {sorted(known_sessions)}"
        )

    if not (
        0.0
        < train_fraction
        < 1.0
    ):
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )
    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )
    if (
        train_fraction
        + validation_fraction
        >= 1.0
    ):
        raise ValueError(
            "train_fraction + validation_fraction "
            "must be < 1."
        )

    yaml_test = yaml_values[
        "test_fraction"
    ]
    if (
        yaml_test is not None
        and sources[
            "train_fraction"
        ] != "cli"
        and sources[
            "validation_fraction"
        ] != "cli"
    ):
        yaml_test = float(
            yaml_test
        )
        derived_test = (
            1.0
            - train_fraction
            - validation_fraction
        )
        if not np.isclose(
            yaml_test,
            derived_test,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "pipeline.sequences.split.test_fraction "
                "does not equal 1 - train_fraction - "
                "validation_fraction."
            )

    settings = SequenceSettings(
        sequence_length=(
            sequence_length
        ),
        target_horizon=(
            target_horizon
        ),
        stride_minutes=(
            stride_minutes
        ),
        scope=scope,
        sessions=sessions,
        train_fraction=(
            train_fraction
        ),
        validation_fraction=(
            validation_fraction
        ),
    )

    return settings, sources


def sequence_settings_payload(
    settings: SequenceSettings,
    sources: dict[str, str],
) -> dict:
    return {
        "effective": {
            "sequence_length": (
                settings.sequence_length
            ),
            "target_horizon_minutes": (
                settings.target_horizon
            ),
            "stride_minutes": (
                settings.stride_minutes
            ),
            "scope": settings.scope,
            "sessions": list(
                settings.sessions
            ),
            "split": {
                "train_fraction": (
                    settings.train_fraction
                ),
                "validation_fraction": (
                    settings.validation_fraction
                ),
                "test_fraction": (
                    settings.test_fraction
                ),
            },
        },
        "sources": sources,
        "precedence": (
            "CLI override > pipeline.yaml > script default"
        ),
    }


def normalize_symbol(value: str) -> str:
    return str(value).strip().lower()


def ensure_timezone(
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

    return pd.to_datetime(
        values,
        errors="raise",
        utc=True,
    ).dt.tz_convert(timezone)


def normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values,
        errors="raise",
    ).dt.date


def parse_bool(values: pd.Series, column: str) -> pd.Series:
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

    parsed = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

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


def discover_parts(
    root: Path,
    *,
    symbol: str,
) -> dict[tuple[int, int], Path]:
    base = root / f"symbol={normalize_symbol(symbol)}"

    if not base.exists():
        raise FileNotFoundError(
            f"Partition root does not exist: {base}"
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
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid partition path: {path}"
            ) from exc

        key = (year, month)
        if key in result:
            raise ValueError(
                f"Duplicate partition {key}: "
                f"{result[key]} and {path}"
            )

        result[key] = path

    if not result:
        raise FileNotFoundError(
            f"No monthly Parquet parts under {base}"
        )

    return result


def target_columns(horizon: int) -> list[str]:
    suffix = f"{int(horizon)}m"

    return [
        "datetime",
        "prediction_time",
        "trading_date",
        "session",
        "symbol",
        "source_id",
        f"target_valid_{suffix}",
        f"target_realized_at_{suffix}",
        f"target_return_bps_{suffix}",
        f"target_log_return_{suffix}",
        f"target_direction_{suffix}",
        f"target_same_session_{suffix}",
        f"target_same_trading_date_{suffix}",
        f"target_invalid_reason_{suffix}",
    ]


def target_structural_columns(horizon: int) -> list[str]:
    """
    v2 build-path contract: no future target OUTCOME value is read by this
    column set. It exposes only the minimum eligibility/timing metadata
    required to reproduce the existing sequence universe (window
    eligibility, split assignment, and the boundary purge) — `target_valid`
    and `target_realized_at` are retained because both are load-bearing for
    that reproduction. `target_invalid_reason` is intentionally omitted
    here: it is diagnostic-only and unused by `process_symbol_v2` (v1's
    `target_columns`/`normalize_targets` path is unchanged and still
    includes it). `target_same_session`/`target_same_trading_date` are
    deliberately excluded here (see `target_outcome_columns`) even though
    they are booleans, not prices — they are still derived from the
    realized future bar, so they belong with the outcome columns, not this
    eligibility set.
    """
    suffix = f"{int(horizon)}m"

    return [
        "datetime",
        "prediction_time",
        "trading_date",
        "session",
        "symbol",
        "source_id",
        f"target_valid_{suffix}",
        f"target_realized_at_{suffix}",
    ]


def target_outcome_columns(horizon: int) -> list[str]:
    """
    Future-target-derived VALUE columns (matches SEQUENCE_INDEX_VALUE_COLUMNS).
    `datetime`/`source_id` are included only as the join key back onto the
    structural row set — they are already-known structural identifiers, not
    outcomes themselves.
    """
    suffix = f"{int(horizon)}m"

    return [
        "datetime",
        "source_id",
        f"target_return_bps_{suffix}",
        f"target_log_return_{suffix}",
        f"target_direction_{suffix}",
        f"target_same_session_{suffix}",
        f"target_same_trading_date_{suffix}",
    ]


def normalize_features(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timezone: str,
) -> pd.DataFrame:
    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Feature partition missing columns: {missing}"
        )

    out = frame[FEATURE_COLUMNS].copy()

    for column in (
        "datetime",
        "prediction_time",
        "feature_available_at",
    ):
        out[column] = ensure_timezone(
            out[column],
            timezone=timezone,
            column=column,
        )

    for column in (
        "calendar_date",
        "trading_date",
    ):
        out[column] = normalize_date(
            out[column]
        )

    for column in (
        "is_current_bar_usable",
        "is_contiguous_from_previous",
    ):
        out[column] = parse_bool(
            out[column],
            column,
        )

    out["contiguous_history_minutes"] = (
        pd.to_numeric(
            out["contiguous_history_minutes"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int32")
    )

    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["source_id"] = (
        out["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["session"] = (
        out["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected_symbol = normalize_symbol(symbol)
    found = out["symbol"].dropna().unique()
    if (
        len(found) != 1
        or found[0] != expected_symbol
    ):
        raise ValueError(
            f"Expected symbol={expected_symbol!r}; "
            f"found={found.tolist()}"
        )

    if out.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            "Duplicate feature (datetime, source_id) keys."
        )

    expected_prediction_time = (
        out["datetime"]
        + timedelta(minutes=1)
    )
    if (
        out["prediction_time"]
        != expected_prediction_time
    ).any():
        raise ValueError(
            "Feature timing violation: expected "
            "prediction_time = datetime + 1 minute."
        )

    leakage = (
        out["feature_available_at"]
        > out["prediction_time"]
    )
    if leakage.any():
        examples = (
            out.loc[
                leakage,
                [
                    "datetime",
                    "feature_available_at",
                    "prediction_time",
                ],
            ]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Feature availability leakage detected: "
            f"{examples}"
        )

    return (
        out.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def normalize_targets(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timezone: str,
    horizon: int,
) -> pd.DataFrame:
    required = target_columns(horizon)
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Target partition missing columns: {missing}"
        )

    out = frame[required].copy()
    suffix = f"{int(horizon)}m"

    for column in (
        "datetime",
        "prediction_time",
        f"target_realized_at_{suffix}",
    ):
        out[column] = ensure_timezone(
            out[column],
            timezone=timezone,
            column=column,
        )

    out["trading_date"] = normalize_date(
        out["trading_date"]
    )

    for column in (
        f"target_valid_{suffix}",
        f"target_same_session_{suffix}",
        f"target_same_trading_date_{suffix}",
    ):
        out[column] = parse_bool(
            out[column],
            column,
        )

    for column in (
        f"target_return_bps_{suffix}",
        f"target_log_return_{suffix}",
        f"target_direction_{suffix}",
    ):
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["source_id"] = (
        out["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["session"] = (
        out["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected_symbol = normalize_symbol(symbol)
    found = out["symbol"].dropna().unique()
    if (
        len(found) != 1
        or found[0] != expected_symbol
    ):
        raise ValueError(
            f"Expected target symbol={expected_symbol!r}; "
            f"found={found.tolist()}"
        )

    if out.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            "Duplicate target (datetime, source_id) keys."
        )

    expected_realized = (
        out["prediction_time"]
        + timedelta(minutes=int(horizon))
    )
    if (
        out[f"target_realized_at_{suffix}"]
        != expected_realized
    ).any():
        raise ValueError(
            f"Target timing violation for {suffix}: "
            "target_realized_at must equal "
            "prediction_time + horizon."
        )

    valid = out[f"target_valid_{suffix}"]

    finite_required = [
        f"target_return_bps_{suffix}",
        f"target_log_return_{suffix}",
        f"target_direction_{suffix}",
    ]
    values = out.loc[
        valid,
        finite_required,
    ].to_numpy(
        dtype="float64",
        na_value=np.nan,
    )
    if (
        len(values)
        and not np.isfinite(values).all()
    ):
        raise ValueError(
            f"Valid {suffix} targets contain NaN/inf."
        )

    return (
        out.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def normalize_target_structural(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timezone: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Structural-only counterpart to `normalize_targets`: validates and
    normalizes eligibility/timing columns without ever touching a target
    outcome VALUE column.
    """
    required = target_structural_columns(
        horizon
    )
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Target partition missing structural columns: {missing}"
        )

    out = frame[required].copy()
    suffix = f"{int(horizon)}m"

    for column in (
        "datetime",
        "prediction_time",
        f"target_realized_at_{suffix}",
    ):
        out[column] = ensure_timezone(
            out[column],
            timezone=timezone,
            column=column,
        )

    out["trading_date"] = normalize_date(
        out["trading_date"]
    )

    out[f"target_valid_{suffix}"] = parse_bool(
        out[f"target_valid_{suffix}"],
        f"target_valid_{suffix}",
    )

    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["source_id"] = (
        out["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    out["session"] = (
        out["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected_symbol = normalize_symbol(symbol)
    found = out["symbol"].dropna().unique()
    if (
        len(found) != 1
        or found[0] != expected_symbol
    ):
        raise ValueError(
            f"Expected target symbol={expected_symbol!r}; "
            f"found={found.tolist()}"
        )

    if out.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            "Duplicate target (datetime, source_id) keys."
        )

    expected_realized = (
        out["prediction_time"]
        + timedelta(minutes=int(horizon))
    )
    if (
        out[f"target_realized_at_{suffix}"]
        != expected_realized
    ).any():
        raise ValueError(
            f"Target timing violation for {suffix}: "
            "target_realized_at must equal "
            "prediction_time + horizon."
        )

    return (
        out.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def read_feature_partition(
    path: Path,
    *,
    symbol: str,
    timezone: str,
) -> pd.DataFrame:
    return normalize_features(
        pd.read_parquet(
            path,
            engine="pyarrow",
            columns=FEATURE_COLUMNS,
        ),
        symbol=symbol,
        timezone=timezone,
    )


def read_target_partition(
    path: Path,
    *,
    symbol: str,
    timezone: str,
    horizon: int,
) -> pd.DataFrame:
    columns = target_columns(horizon)

    return normalize_targets(
        pd.read_parquet(
            path,
            engine="pyarrow",
            columns=columns,
        ),
        symbol=symbol,
        timezone=timezone,
        horizon=horizon,
    )


def read_target_structural_partition(
    path: Path,
    *,
    symbol: str,
    timezone: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Reads only eligibility/timing columns from a monthly target partition —
    never a target outcome VALUE column. Safe for every row, every split,
    every month, including TEST-period months, for the v2 build path.
    """
    columns = target_structural_columns(
        horizon
    )

    return normalize_target_structural(
        pd.read_parquet(
            path,
            engine="pyarrow",
            columns=columns,
        ),
        symbol=symbol,
        timezone=timezone,
        horizon=horizon,
    )


def read_target_outcome_values(
    path: Path,
    *,
    timezone: str,
    horizon: int,
    trainval_max_datetime: pd.Timestamp,
    validation_last_trading_date,
) -> pd.DataFrame:
    """
    Reads target outcome VALUE columns for this file's rows satisfying
    BOTH:

        trading_date <= validation_last_trading_date   (authoritative TEST exclusion)
        datetime     <= trainval_max_datetime           (additionally excludes
                                                          boundary-purged
                                                          VALIDATION rows)

    via a single Arrow-level predicate-pushdown filter
    (`pyarrow.dataset` `filter=`). `trading_date` is the authoritative
    split-boundary predicate — it alone is sufficient to exclude every
    TEST-split row, since TEST is defined as `trading_date >
    validation_last_trading_date` (`assign_split`). `trainval_max_datetime`
    is supplied by the caller as the maximum `datetime` (window endpoint)
    among the sequence endpoints already retained for TRAIN+VALIDATION
    *after* both split assignment and the existing `target_realized_at`
    boundary purge — computed from structural/timing metadata only, before
    this function is ever called — and additionally excludes rows that,
    while chronologically on a TRAIN/VALIDATION trading date, were purged
    for crossing the split boundary within their own target horizon.

    Both filter scalars are constructed from the dataset's own physical
    schema field types (`dataset.schema.field(...).type`), never guessed,
    to avoid any timezone/unit/type mismatch with the stored Parquet data.

    Rows failing either predicate are excluded by the Arrow compute kernel
    inside the scanner — the pandas DataFrame this function returns never
    contains them, even transiently, even when the underlying Parquet file
    mixes retained and excluded rows in a single row group. Internal
    Arrow/Parquet decoding of excluded rows' column bytes may occur to
    evaluate the filter; that decoding stays inside the Arrow C++ scanner
    and is never returned to this or any other Python caller.

    `trading_date` is read only to enforce and then fail-closed-verify the
    filter; it is dropped before this function returns, since callers join
    on `(source_id, datetime)` and do not need it.

    Normalization below operates only on this already-filtered table — the
    source file is never re-read. Finite-value validation against the
    `target_valid` flag is intentionally left to the caller, to be applied
    after this frame is joined onto the retained structural endpoint set
    (see `process_symbol_v2`), since `target_valid` is not available here.
    """
    columns = target_outcome_columns(
        horizon
    )
    suffix = f"{int(horizon)}m"

    dataset = pa_dataset.dataset(
        path,
        format="parquet",
        partitioning=None,
    )

    datetime_bound = pa.scalar(
        trainval_max_datetime.to_pydatetime(),
        type=dataset.schema.field(
            "datetime"
        ).type,
    )
    trading_date_bound = pa.scalar(
        validation_last_trading_date,
        type=dataset.schema.field(
            "trading_date"
        ).type,
    )

    table = dataset.to_table(
        columns=columns + ["trading_date"],
        filter=(
            (
                pc.field("trading_date")
                <= trading_date_bound
            )
            & (
                pc.field("datetime")
                <= datetime_bound
            )
        ),
    )

    out = table.to_pandas()

    # Fail-closed defense in depth: confirm the Arrow filter actually
    # enforced BOTH predicates before this data is treated as safe to hold
    # in a Python object at all.
    if len(out) and (
        out["datetime"].max()
        > trainval_max_datetime
    ):
        raise AssertionError(
            f"{path}: predicate-pushdown filter failed to exclude "
            "rows beyond the TRAIN+VALIDATION datetime bound."
        )
    if len(out) and (
        out["trading_date"].max()
        > validation_last_trading_date
    ):
        raise AssertionError(
            f"{path}: predicate-pushdown filter failed to exclude "
            "rows beyond validation_last_trading_date — a TEST-split "
            "row would otherwise have entered this frame."
        )

    out = out.drop(
        columns=["trading_date"]
    )

    out["datetime"] = ensure_timezone(
        out["datetime"],
        timezone=timezone,
        column="datetime",
    )
    out["source_id"] = (
        out["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in (
        f"target_same_session_{suffix}",
        f"target_same_trading_date_{suffix}",
    ):
        out[column] = parse_bool(
            out[column],
            column,
        )

    for column in (
        f"target_return_bps_{suffix}",
        f"target_log_return_{suffix}",
        f"target_direction_{suffix}",
    ):
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    if out.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            f"{path}: duplicate target-outcome "
            "(datetime, source_id) keys."
        )

    return (
        out.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def verify_aligned_month(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    key: tuple[int, int],
) -> None:
    if len(features) != len(targets):
        raise ValueError(
            f"{key}: feature/target row count mismatch: "
            f"{len(features)} != {len(targets)}"
        )

    for column in (
        "datetime",
        "prediction_time",
        "symbol",
        "source_id",
        "trading_date",
        "session",
    ):
        left = features[column].reset_index(drop=True)
        right = targets[column].reset_index(drop=True)

        if not left.equals(right):
            examples = pd.DataFrame(
                {
                    "feature": left,
                    "target": right,
                }
            ).loc[
                lambda x: x["feature"] != x["target"]
            ].head(5)

            raise ValueError(
                f"{key}: feature/target {column} mismatch. "
                f"Examples={examples.to_dict(orient='records')}"
            )


def collect_trading_date_cutoffs(
    feature_parts: dict[tuple[int, int], Path],
    *,
    symbol: str,
    timezone: str,
) -> dict[object, pd.Timestamp]:
    """
    Read only trading_date + prediction_time from the feature store and retain
    the latest prediction_time for every exchange trading date.

    Splitting by trading date prevents a train/validation/test boundary from
    landing in the middle of a market day.
    """
    cutoffs: dict[object, pd.Timestamp] = {}

    for key in sorted(feature_parts):
        path = feature_parts[key]
        small = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=[
                "trading_date",
                "prediction_time",
                "symbol",
            ],
        )

        small["symbol"] = (
            small["symbol"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        expected_symbol = normalize_symbol(symbol)
        found = small["symbol"].dropna().unique()
        if (
            len(found) != 1
            or found[0] != expected_symbol
        ):
            raise ValueError(
                f"{key}: unexpected symbol(s) "
                f"{found.tolist()}"
            )

        small["trading_date"] = normalize_date(
            small["trading_date"]
        )
        small["prediction_time"] = ensure_timezone(
            small["prediction_time"],
            timezone=timezone,
            column="prediction_time",
        )

        grouped = (
            small.groupby(
                "trading_date",
                sort=False,
            )["prediction_time"]
            .max()
        )

        for trading_date, cutoff in grouped.items():
            existing = cutoffs.get(
                trading_date
            )
            if (
                existing is None
                or cutoff > existing
            ):
                cutoffs[
                    trading_date
                ] = cutoff

    if not cutoffs:
        raise ValueError(
            "No trading dates found in feature store."
        )

    return cutoffs


def build_fraction_split_plan(
    trading_date_cutoffs: dict[object, pd.Timestamp],
    *,
    train_fraction: float,
    validation_fraction: float,
) -> SplitPlan:
    if not (
        0.0 < train_fraction < 1.0
    ):
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )

    if not (
        0.0 < validation_fraction < 1.0
    ):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if (
        train_fraction
        + validation_fraction
        >= 1.0
    ):
        raise ValueError(
            "train_fraction + validation_fraction "
            "must be < 1."
        )

    dates = sorted(
        trading_date_cutoffs
    )
    n = len(dates)

    if n < 3:
        raise ValueError(
            "At least 3 trading dates are required "
            "for train/validation/test."
        )

    train_count = int(
        np.floor(
            n * train_fraction
        )
    )
    validation_count = int(
        np.floor(
            n * validation_fraction
        )
    )

    train_count = max(
        1,
        min(train_count, n - 2),
    )
    validation_count = max(
        1,
        min(
            validation_count,
            n - train_count - 1,
        ),
    )

    train_last = dates[
        train_count - 1
    ]
    validation_last = dates[
        train_count
        + validation_count
        - 1
    ]

    return SplitPlan(
        train_fraction=float(
            train_fraction
        ),
        validation_fraction=float(
            validation_fraction
        ),
        test_fraction=float(
            1.0
            - train_fraction
            - validation_fraction
        ),
        train_last_trading_date=(
            train_last
        ),
        validation_last_trading_date=(
            validation_last
        ),
        train_cutoff_prediction_time=(
            trading_date_cutoffs[
                train_last
            ]
        ),
        validation_cutoff_prediction_time=(
            trading_date_cutoffs[
                validation_last
            ]
        ),
        trading_dates=n,
    )


def assign_split(
    trading_date: pd.Series,
    *,
    plan: SplitPlan,
) -> pd.Series:
    train = (
        trading_date
        <= plan.train_last_trading_date
    )
    validation = (
        (trading_date
         > plan.train_last_trading_date)
        & (
            trading_date
            <= plan.validation_last_trading_date
        )
    )

    return pd.Series(
        np.select(
            [
                train,
                validation,
            ],
            [
                "train",
                "validation",
            ],
            default="test",
        ),
        index=trading_date.index,
        dtype="object",
    )


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

    temp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        frame.to_parquet(
            temp,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        os.replace(
            temp,
            path,
        )
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        temp.write_text(
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
            temp,
            path,
        )
    finally:
        if temp.exists():
            temp.unlink()



def compute_scope_history(
    features: pd.DataFrame,
    *,
    allowed_sessions: tuple[str, ...],
    scope: str,
    state: dict[str, dict],
) -> pd.Series:
    """
    Count consecutive usable input rows that satisfy the configured session
    scope. State is carried across monthly partitions so future overnight
    coverage can cross a month boundary without losing valid history.

    Physical continuity remains anchored to features.py's
    contiguous_history_minutes. A value > 1 means the current usable row
    continued from the previous usable one-minute row; 1 marks a reset.
    """
    allowed = set(
        allowed_sessions
    )
    result = pd.Series(
        0,
        index=features.index,
        dtype="int64",
    )

    for source_id, group in features.groupby(
        "source_id",
        sort=False,
    ):
        idx = group.index
        usable = (
            group[
                "is_current_bar_usable"
            ].to_numpy(
                dtype=bool
            )
        )
        in_session = (
            group[
                "session"
            ]
            .isin(
                allowed
            )
            .to_numpy(
                dtype=bool
            )
        )
        raw_history = (
            group[
                "contiguous_history_minutes"
            ]
            .to_numpy(
                dtype="int64"
            )
        )
        sessions = (
            group[
                "session"
            ]
            .astype(str)
            .to_numpy()
        )

        prior = state.get(
            str(source_id),
            {
                "history": 0,
                "session": None,
                "eligible": False,
            },
        )

        history = np.zeros(
            len(group),
            dtype="int64",
        )

        for position in range(
            len(group)
        ):
            eligible = bool(
                usable[position]
                and in_session[position]
            )

            if not eligible:
                history[position] = 0
                prior_history = 0
                prior_session = (
                    sessions[position]
                )
                prior_eligible = False
                continue

            if position == 0:
                previous_history = int(
                    prior[
                        "history"
                    ]
                )
                previous_session = prior[
                    "session"
                ]
                previous_eligible = bool(
                    prior[
                        "eligible"
                    ]
                )
            else:
                previous_history = int(
                    history[
                        position - 1
                    ]
                )
                previous_session = (
                    sessions[
                        position - 1
                    ]
                )
                previous_eligible = bool(
                    history[
                        position - 1
                    ]
                    > 0
                )

            physical_continuation = bool(
                raw_history[
                    position
                ]
                > 1
            )

            session_continuation = (
                scope
                == "continuous"
                or (
                    previous_session
                    == sessions[
                        position
                    ]
                )
            )

            if (
                previous_eligible
                and physical_continuation
                and session_continuation
            ):
                history[
                    position
                ] = (
                    previous_history
                    + 1
                )
            else:
                history[
                    position
                ] = 1

        if len(group):
            state[
                str(source_id)
            ] = {
                "history": int(
                    history[-1]
                ),
                "session": str(
                    sessions[-1]
                ),
                "eligible": bool(
                    history[-1]
                    > 0
                ),
            }

        result.loc[
            idx
        ] = history

    return result


def process_symbol(
    *,
    features_root: Path,
    targets_root: Path,
    output_root: Path,
    symbol: str,
    sequence_length: int,
    target_horizon: int,
    stride_minutes: int,
    scope: str,
    sessions: tuple[str, ...],
    train_fraction: float,
    validation_fraction: float,
    config,
    reports_root: Path | None,
    configuration_sources: dict[str, str] | None = None,
) -> dict:
    symbol = normalize_symbol(
        symbol
    )
    sequence_length = int(
        sequence_length
    )
    target_horizon = int(
        target_horizon
    )
    stride_minutes = int(
        stride_minutes
    )
    scope = str(
        scope
    ).strip().lower()
    sessions = tuple(
        str(item).strip().lower()
        for item in sessions
    )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )
    if (
        sequence_length
        > MAX_SUPPORTED_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"sequence_length={sequence_length} exceeds "
            f"the v2.1 feature continuity cap of "
            f"{MAX_SUPPORTED_SEQUENCE_LENGTH}. "
            "Recompute continuity with a larger cap before "
            "building longer sequences."
        )
    if target_horizon < 1:
        raise ValueError(
            "target_horizon must be >= 1."
        )
    if stride_minutes < 1:
        raise ValueError(
            "stride_minutes must be >= 1."
        )
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"Unsupported scope={scope!r}."
        )
    if not sessions:
        raise ValueError(
            "At least one session is required."
        )

    feature_parts = discover_parts(
        features_root,
        symbol=symbol,
    )
    target_parts = discover_parts(
        targets_root,
        symbol=symbol,
    )

    feature_keys = sorted(
        feature_parts
    )
    target_keys = sorted(
        target_parts
    )

    if feature_keys != target_keys:
        missing_targets = sorted(
            set(feature_keys)
            - set(target_keys)
        )
        extra_targets = sorted(
            set(target_keys)
            - set(feature_keys)
        )

        raise ValueError(
            "Feature/target partition mismatch. "
            f"missing_targets={missing_targets}, "
            f"extra_targets={extra_targets}"
        )

    trading_date_cutoffs = (
        collect_trading_date_cutoffs(
            feature_parts,
            symbol=symbol,
            timezone=config.timezone,
        )
    )

    split_plan = (
        build_fraction_split_plan(
            trading_date_cutoffs,
            train_fraction=train_fraction,
            validation_fraction=(
                validation_fraction
            ),
        )
    )

    suffix = f"{target_horizon}m"

    sessions_tag = "+".join(
        sessions
    )

    parameter_root = (
        output_root
        / f"sequence_length={sequence_length}"
        / f"horizon={suffix}"
        / f"scope={scope}"
        / f"stride={stride_minutes}"
        / f"sessions={sessions_tag}"
    )

    totals = {
        "input_rows": 0,
        "current_usable_rows": 0,
        "window_eligible_rows": 0,
        "stride_rejected_rows": 0,
        "structural_eligible_rows": 0,
        "target_valid_rows": 0,
        "scope_target_rejected_rows": 0,
        "joint_eligible_rows": 0,
        "written_rows": 0,
        "split_candidate_rows": {
            "train": 0,
            "validation": 0,
            "test": 0,
        },
        "split_written_rows": {
            "train": 0,
            "validation": 0,
            "test": 0,
        },
        "split_purged_target_boundary_rows": {
            "train": 0,
            "validation": 0,
            "test": 0,
        },
        "target_invalid_reasons_on_structural_rows": {},
    }

    month_summaries = []
    sample_offset = 0
    scope_state: dict[str, dict] = {}

    for key in feature_keys:
        year, month = key

        features = read_feature_partition(
            feature_parts[key],
            symbol=symbol,
            timezone=config.timezone,
        )
        targets = read_target_partition(
            target_parts[key],
            symbol=symbol,
            timezone=config.timezone,
            horizon=target_horizon,
        )

        verify_aligned_month(
            features,
            targets,
            key=key,
        )

        current_usable = (
            features[
                "is_current_bar_usable"
            ]
        )

        scope_history = (
            compute_scope_history(
                features,
                allowed_sessions=(
                    sessions
                ),
                scope=scope,
                state=scope_state,
            )
        )

        # features.py remains the source of truth for physical one-minute
        # continuity. scope_history only adds session-policy constraints.
        window_eligible = (
            current_usable
            & (
                features[
                    "contiguous_history_minutes"
                ]
                >= sequence_length
            )
            & (
                scope_history
                >= sequence_length
            )
        )

        stride_phase = (
            (
                scope_history
                - sequence_length
            )
            % stride_minutes
            == 0
        )

        structural = (
            window_eligible
            & stride_phase
        )

        target_valid = (
            targets[
                f"target_valid_{suffix}"
            ]
        )

        if scope == "same_session":
            target_scope_ok = (
                targets[
                    f"target_same_session_{suffix}"
                ]
            )
        else:
            target_scope_ok = pd.Series(
                True,
                index=targets.index,
                dtype=bool,
            )

        joint = (
            structural
            & target_valid
            & target_scope_ok
        )

        structural_invalid_reasons = (
            targets.loc[
                structural
                & ~target_valid,
                f"target_invalid_reason_{suffix}",
            ]
            .value_counts()
            .to_dict()
        )

        scope_target_rejected = (
            structural
            & target_valid
            & ~target_scope_ok
        )
        if scope_target_rejected.any():
            structural_invalid_reasons[
                "scope_target_crosses_session"
            ] = int(
                scope_target_rejected.sum()
            )

        candidates = pd.DataFrame(
            {
                "datetime": (
                    features.loc[
                        joint,
                        "datetime",
                    ]
                    .reset_index(drop=True)
                ),
                "prediction_time": (
                    features.loc[
                        joint,
                        "prediction_time",
                    ]
                    .reset_index(drop=True)
                ),
                "calendar_date": (
                    features.loc[
                        joint,
                        "calendar_date",
                    ]
                    .reset_index(drop=True)
                ),
                "trading_date": (
                    features.loc[
                        joint,
                        "trading_date",
                    ]
                    .reset_index(drop=True)
                ),
                "session": (
                    features.loc[
                        joint,
                        "session",
                    ]
                    .reset_index(drop=True)
                ),
                "symbol": (
                    features.loc[
                        joint,
                        "symbol",
                    ]
                    .reset_index(drop=True)
                ),
                "source_id": (
                    features.loc[
                        joint,
                        "source_id",
                    ]
                    .reset_index(drop=True)
                ),
                "contiguous_history_minutes": (
                    features.loc[
                        joint,
                        "contiguous_history_minutes",
                    ]
                    .reset_index(drop=True)
                ),
                "scope_history_minutes": (
                    scope_history.loc[
                        joint
                    ]
                    .reset_index(drop=True)
                ),
                "target_realized_at": (
                    targets.loc[
                        joint,
                        f"target_realized_at_{suffix}",
                    ]
                    .reset_index(drop=True)
                ),
                "target_return_bps": (
                    targets.loc[
                        joint,
                        f"target_return_bps_{suffix}",
                    ]
                    .reset_index(drop=True)
                ),
                "target_log_return": (
                    targets.loc[
                        joint,
                        f"target_log_return_{suffix}",
                    ]
                    .reset_index(drop=True)
                ),
                "target_direction": (
                    targets.loc[
                        joint,
                        f"target_direction_{suffix}",
                    ]
                    .reset_index(drop=True)
                ),
                "target_same_session": (
                    targets.loc[
                        joint,
                        f"target_same_session_{suffix}",
                    ]
                    .reset_index(drop=True)
                ),
                "target_same_trading_date": (
                    targets.loc[
                        joint,
                        f"target_same_trading_date_{suffix}",
                    ]
                    .reset_index(drop=True)
                ),
            }
        )

        candidates[
            "sequence_length"
        ] = sequence_length
        candidates[
            "target_horizon_minutes"
        ] = target_horizon
        candidates[
            "stride_minutes"
        ] = stride_minutes
        candidates[
            "sequence_scope"
        ] = scope

        candidates[
            "window_end_datetime"
        ] = candidates[
            "datetime"
        ]
        candidates[
            "window_start_datetime"
        ] = (
            candidates[
                "window_end_datetime"
            ]
            - timedelta(
                minutes=(
                    sequence_length
                    - 1
                )
            )
        )

        candidates[
            "split"
        ] = assign_split(
            candidates[
                "trading_date"
            ],
            plan=split_plan,
        )

        candidate_counts = (
            candidates[
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        boundary_ok = pd.Series(
            True,
            index=candidates.index,
            dtype=bool,
        )

        train_mask = candidates[
            "split"
        ].eq("train")
        validation_mask = candidates[
            "split"
        ].eq("validation")

        boundary_ok.loc[
            train_mask
        ] = (
            candidates.loc[
                train_mask,
                "target_realized_at",
            ]
            <= split_plan.train_cutoff_prediction_time
        )

        boundary_ok.loc[
            validation_mask
        ] = (
            candidates.loc[
                validation_mask,
                "target_realized_at",
            ]
            <= split_plan.validation_cutoff_prediction_time
        )

        purged = (
            ~boundary_ok
        )

        purged_counts = (
            candidates.loc[
                purged,
                "split",
            ]
            .value_counts()
            .to_dict()
        )

        output = (
            candidates.loc[
                boundary_ok
            ]
            .copy()
            .reset_index(drop=True)
        )

        output[
            "sample_index"
        ] = np.arange(
            sample_offset,
            sample_offset
            + len(output),
            dtype="int64",
        )
        sample_offset += len(
            output
        )

        output = output[
            SEQUENCE_INDEX_COLUMNS
        ]

        if output.duplicated(
            [
                "symbol",
                "source_id",
                "window_end_datetime",
            ]
        ).any():
            raise ValueError(
                f"{key}: duplicate sequence endpoint keys."
            )

        if (
            output[
                "prediction_time"
            ]
            <= output[
                "window_end_datetime"
            ]
        ).any():
            raise AssertionError(
                f"{key}: prediction_time must be "
                "after the final input bar."
            )

        if (
            output[
                "target_realized_at"
            ]
            <= output[
                "prediction_time"
            ]
        ).any():
            raise AssertionError(
                f"{key}: target must be realized "
                "after prediction_time."
            )

        output_path = (
            parameter_root
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

        written_split_counts = (
            output[
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        month_summary = {
            "year": year,
            "month": month,
            "input_rows": int(
                len(features)
            ),
            "current_usable_rows": int(
                current_usable.sum()
            ),
            "window_eligible_rows": int(
                window_eligible.sum()
            ),
            "stride_rejected_rows": int(
                (
                    window_eligible
                    & ~stride_phase
                ).sum()
            ),
            "structural_eligible_rows": int(
                structural.sum()
            ),
            "target_valid_rows": int(
                target_valid.sum()
            ),
            "scope_target_rejected_rows": int(
                scope_target_rejected.sum()
            ),
            "joint_eligible_rows": int(
                joint.sum()
            ),
            "written_rows": int(
                len(output)
            ),
            "split_candidate_rows": {
                name: int(
                    candidate_counts.get(
                        name,
                        0,
                    )
                )
                for name in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "split_written_rows": {
                name: int(
                    written_split_counts.get(
                        name,
                        0,
                    )
                )
                for name in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "split_purged_target_boundary_rows": {
                name: int(
                    purged_counts.get(
                        name,
                        0,
                    )
                )
                for name in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "target_invalid_reasons_on_structural_rows": {
                str(name): int(count)
                for name, count
                in structural_invalid_reasons.items()
            },
            "output_file": str(
                output_path
            ),
        }

        month_summaries.append(
            month_summary
        )

        totals[
            "input_rows"
        ] += len(features)
        totals[
            "current_usable_rows"
        ] += int(
            current_usable.sum()
        )
        totals[
            "window_eligible_rows"
        ] += int(
            window_eligible.sum()
        )
        totals[
            "stride_rejected_rows"
        ] += int(
            (
                window_eligible
                & ~stride_phase
            ).sum()
        )
        totals[
            "structural_eligible_rows"
        ] += int(
            structural.sum()
        )
        totals[
            "target_valid_rows"
        ] += int(
            target_valid.sum()
        )
        totals[
            "scope_target_rejected_rows"
        ] += int(
            scope_target_rejected.sum()
        )
        totals[
            "joint_eligible_rows"
        ] += int(
            joint.sum()
        )
        totals[
            "written_rows"
        ] += len(output)

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            totals[
                "split_candidate_rows"
            ][
                split_name
            ] += int(
                candidate_counts.get(
                    split_name,
                    0,
                )
            )

            totals[
                "split_written_rows"
            ][
                split_name
            ] += int(
                written_split_counts.get(
                    split_name,
                    0,
                )
            )

            totals[
                "split_purged_target_boundary_rows"
            ][
                split_name
            ] += int(
                purged_counts.get(
                    split_name,
                    0,
                )
            )

        for (
            reason,
            count,
        ) in (
            structural_invalid_reasons.items()
        ):
            totals[
                "target_invalid_reasons_on_structural_rows"
            ][
                str(reason)
            ] = (
                totals[
                    "target_invalid_reasons_on_structural_rows"
                ].get(
                    str(reason),
                    0,
                )
                + int(count)
            )

        print(
            f"[{year:04d}-{month:02d}] "
            f"rows={len(features):,}, "
            f"window={int(window_eligible.sum()):,}, "
            f"structural={int(structural.sum()):,}, "
            f"target_valid={int(target_valid.sum()):,}, "
            f"joint={int(joint.sum()):,}, "
            f"written={len(output):,}"
        )

    summary = {
        "sequence_version": (
            SEQUENCE_VERSION
        ),
        "symbol": symbol,
        "sequence_length": (
            sequence_length
        ),
        "target_horizon_minutes": (
            target_horizon
        ),
        "stride_minutes": (
            stride_minutes
        ),
        "scope": scope,
        "sessions": list(
            sessions
        ),
        "effective_configuration": {
            "sequence_length": sequence_length,
            "target_horizon_minutes": target_horizon,
            "stride_minutes": stride_minutes,
            "scope": scope,
            "sessions": list(
                sessions
            ),
            "train_fraction": (
                train_fraction
            ),
            "validation_fraction": (
                validation_fraction
            ),
            "test_fraction": (
                1.0
                - train_fraction
                - validation_fraction
            ),
        },
        "configuration_sources": (
            configuration_sources
            or {}
        ),
        "configuration_precedence": (
            "CLI override > pipeline.yaml > script default"
        ),
        "sequence_semantics": (
            "A sequence contains sequence_length consecutive usable "
            "1-minute feature rows ending at window_end_datetime. "
            "The endpoint row is available at prediction_time. "
            + (
                "Session boundaries may be crossed when physical one-minute "
                "continuity is preserved and all input rows belong to the "
                "configured sessions. "
                if scope == "continuous"
                else
                "The full input window and exact-horizon target must remain "
                "inside one session. "
            )
            + "Unresolved rows and timestamp gaps break continuity."
        ),
        "session_policy": (
            "Configured sessions constrain the input window. "
            "For same_session scope, target_same_session must also be true. "
            "For continuous scope, the exact target may cross a session "
            "boundary, matching targets.py semantics."
        ),
        "materialization_policy": (
            "This stage writes a compact sequence endpoint/index store, "
            "not a dense 3-D model tensor. Feature selection, finite-value "
            "window checks, train-fitted scaling, and lazy batch "
            "materialization belong to model preparation/training."
        ),
        "feature_continuity_cap_minutes": (
            MAX_SUPPORTED_SEQUENCE_LENGTH
        ),
        "feature_root": str(
            features_root
        ),
        "target_root": str(
            targets_root
        ),
        "output_root": str(
            parameter_root
        ),
        "months": len(
            feature_keys
        ),
        "split_strategy": {
            "unit": "trading_date",
            "train_fraction_requested": (
                split_plan.train_fraction
            ),
            "validation_fraction_requested": (
                split_plan.validation_fraction
            ),
            "test_fraction_requested": (
                split_plan.test_fraction
            ),
            "trading_dates": (
                split_plan.trading_dates
            ),
            "train_last_trading_date": (
                split_plan.train_last_trading_date
            ),
            "validation_last_trading_date": (
                split_plan.validation_last_trading_date
            ),
            "train_cutoff_prediction_time": (
                split_plan.train_cutoff_prediction_time
            ),
            "validation_cutoff_prediction_time": (
                split_plan.validation_cutoff_prediction_time
            ),
            "boundary_policy": (
                "Train/validation samples are retained only when "
                "target_realized_at is no later than that split's final "
                "prediction_time. Validation/test input windows may use "
                "earlier historical context because it was already known "
                "at prediction time."
            ),
        },
        **totals,
        "month_summaries": (
            month_summaries
        ),
    }

    if reports_root is not None:
        report = (
            reports_root
            / symbol
            / (
                "sequence_summary_"
                f"v{SEQUENCE_VERSION}_"
                f"L{sequence_length}_"
                f"H{target_horizon}m_"
                f"{scope}_"
                f"S{stride_minutes}_"
                f"{sessions_tag}.json"
            )
        )

        atomic_write_json(
            summary,
            report,
        )

        summary[
            "report_file"
        ] = str(
            report
        )

    return summary


def process_symbol_v2(
    *,
    features_root: Path,
    targets_root: Path,
    structural_output_root: Path,
    trainval_output_root: Path,
    symbol: str,
    sequence_length: int,
    target_horizon: int,
    stride_minutes: int,
    scope: str,
    sessions: tuple[str, ...],
    train_fraction: float,
    validation_fraction: float,
    config,
    reports_root: Path | None,
    configuration_sources: dict[str, str] | None = None,
) -> dict:
    """
    Canonical Phase-3 v2 build path. Produces, directly from raw
    features_root/targets_root — never from a pre-existing v1 artifact —
    two additive outputs:

      structural_output_root: every split, structural columns only
        (SEQUENCE_INDEX_STRUCTURAL_COLUMNS) — no target outcome VALUE for
        any row, including TRAIN/VALIDATION.

      trainval_output_root: TRAIN+VALIDATION rows only, full columns
        (SEQUENCE_INDEX_COLUMNS).

    Two passes over the same monthly partitions:

      Pass 1 (structural): for every month, reads only structural/timing
      columns (never a target outcome VALUE), assigns splits via
      `assign_split`, applies the existing `target_realized_at` boundary
      purge, and writes the structural output. This pass alone determines
      `trainval_max_datetime` — the maximum window endpoint among rows
      actually retained for TRAIN+VALIDATION after split assignment AND the
      purge — before any outcome value is read.

      Pass 2 (trainval values): only for months that Pass 1 found to have
      at least one retained TRAIN/VALIDATION row, reads target outcome
      VALUE columns via `read_target_outcome_values`'s Arrow-level
      predicate-pushdown filter (bounded by `trainval_max_datetime`), joins
      them onto that month's retained structural rows, and writes the
      trainval output. A month with zero retained rows (i.e. entirely
      TEST-period) never has its outcome columns read at all.
    """
    symbol = normalize_symbol(
        symbol
    )
    sequence_length = int(
        sequence_length
    )
    target_horizon = int(
        target_horizon
    )
    stride_minutes = int(
        stride_minutes
    )
    scope = str(
        scope
    ).strip().lower()
    sessions = tuple(
        str(item).strip().lower()
        for item in sessions
    )

    if scope == "same_session":
        raise NotImplementedError(
            "process_symbol_v2 does not support scope='same_session': "
            "target_same_session is an outcome-VALUE column (derived from "
            "the realized future bar) and cannot be read structurally for "
            "every row without materializing TEST outcomes. The active "
            "project configuration uses scope='continuous', which does "
            "not need this column for eligibility."
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )
    if (
        sequence_length
        > MAX_SUPPORTED_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"sequence_length={sequence_length} exceeds "
            f"the v2.1 feature continuity cap of "
            f"{MAX_SUPPORTED_SEQUENCE_LENGTH}."
        )
    if target_horizon < 1:
        raise ValueError(
            "target_horizon must be >= 1."
        )
    if stride_minutes < 1:
        raise ValueError(
            "stride_minutes must be >= 1."
        )
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"Unsupported scope={scope!r}."
        )
    if not sessions:
        raise ValueError(
            "At least one session is required."
        )

    feature_parts = discover_parts(
        features_root,
        symbol=symbol,
    )
    target_parts = discover_parts(
        targets_root,
        symbol=symbol,
    )

    feature_keys = sorted(
        feature_parts
    )
    target_keys = sorted(
        target_parts
    )

    if feature_keys != target_keys:
        missing_targets = sorted(
            set(feature_keys)
            - set(target_keys)
        )
        extra_targets = sorted(
            set(target_keys)
            - set(feature_keys)
        )

        raise ValueError(
            "Feature/target partition mismatch. "
            f"missing_targets={missing_targets}, "
            f"extra_targets={extra_targets}"
        )

    trading_date_cutoffs = (
        collect_trading_date_cutoffs(
            feature_parts,
            symbol=symbol,
            timezone=config.timezone,
        )
    )

    split_plan = (
        build_fraction_split_plan(
            trading_date_cutoffs,
            train_fraction=train_fraction,
            validation_fraction=(
                validation_fraction
            ),
        )
    )

    suffix = f"{target_horizon}m"
    sessions_tag = "+".join(
        sessions
    )

    shape_suffix = (
        f"sequence_length={sequence_length}"
        f"/horizon={suffix}"
        f"/scope={scope}"
        f"/stride={stride_minutes}"
        f"/sessions={sessions_tag}"
    )

    structural_parameter_root = (
        structural_output_root
        / shape_suffix
    )
    trainval_parameter_root = (
        trainval_output_root
        / shape_suffix
    )

    structural_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }
    trainval_counts = {
        "train": 0,
        "validation": 0,
    }

    sample_offset = 0
    scope_state: dict[str, dict] = {}
    retained_trainval_by_month: dict[
        tuple[int, int], pd.DataFrame
    ] = {}
    trainval_max_datetime: pd.Timestamp | None = None
    month_summaries = []

    # --- Pass 1: structural, every month, no outcome VALUE ever read. ---
    for key in feature_keys:
        year, month = key

        features = read_feature_partition(
            feature_parts[key],
            symbol=symbol,
            timezone=config.timezone,
        )
        targets_structural = (
            read_target_structural_partition(
                target_parts[key],
                symbol=symbol,
                timezone=config.timezone,
                horizon=target_horizon,
            )
        )

        verify_aligned_month(
            features,
            targets_structural,
            key=key,
        )

        current_usable = features[
            "is_current_bar_usable"
        ]

        scope_history = compute_scope_history(
            features,
            allowed_sessions=sessions,
            scope=scope,
            state=scope_state,
        )

        window_eligible = (
            current_usable
            & (
                features[
                    "contiguous_history_minutes"
                ]
                >= sequence_length
            )
            & (
                scope_history
                >= sequence_length
            )
        )

        stride_phase = (
            (
                scope_history
                - sequence_length
            )
            % stride_minutes
            == 0
        )

        structural_mask = (
            window_eligible
            & stride_phase
        )

        target_valid = targets_structural[
            f"target_valid_{suffix}"
        ]

        # scope='same_session' is rejected above; continuous scope never
        # needs target_same_session for eligibility.
        target_scope_ok = pd.Series(
            True,
            index=targets_structural.index,
            dtype=bool,
        )

        joint = (
            structural_mask
            & target_valid
            & target_scope_ok
        )

        candidates = pd.DataFrame(
            {
                "datetime": (
                    features.loc[
                        joint, "datetime"
                    ].reset_index(drop=True)
                ),
                "prediction_time": (
                    features.loc[
                        joint, "prediction_time"
                    ].reset_index(drop=True)
                ),
                "calendar_date": (
                    features.loc[
                        joint, "calendar_date"
                    ].reset_index(drop=True)
                ),
                "trading_date": (
                    features.loc[
                        joint, "trading_date"
                    ].reset_index(drop=True)
                ),
                "session": (
                    features.loc[
                        joint, "session"
                    ].reset_index(drop=True)
                ),
                "symbol": (
                    features.loc[
                        joint, "symbol"
                    ].reset_index(drop=True)
                ),
                "source_id": (
                    features.loc[
                        joint, "source_id"
                    ].reset_index(drop=True)
                ),
                "contiguous_history_minutes": (
                    features.loc[
                        joint,
                        "contiguous_history_minutes",
                    ].reset_index(drop=True)
                ),
                "scope_history_minutes": (
                    scope_history.loc[
                        joint
                    ].reset_index(drop=True)
                ),
                "target_realized_at": (
                    targets_structural.loc[
                        joint,
                        f"target_realized_at_{suffix}",
                    ].reset_index(drop=True)
                ),
            }
        )

        candidates["sequence_length"] = sequence_length
        candidates["target_horizon_minutes"] = target_horizon
        candidates["stride_minutes"] = stride_minutes
        candidates["sequence_scope"] = scope

        candidates["window_end_datetime"] = candidates["datetime"]
        candidates["window_start_datetime"] = (
            candidates["window_end_datetime"]
            - timedelta(
                minutes=(sequence_length - 1)
            )
        )

        candidates["split"] = assign_split(
            candidates["trading_date"],
            plan=split_plan,
        )

        boundary_ok = pd.Series(
            True,
            index=candidates.index,
            dtype=bool,
        )
        train_mask = candidates["split"].eq("train")
        validation_mask = candidates["split"].eq(
            "validation"
        )

        boundary_ok.loc[train_mask] = (
            candidates.loc[
                train_mask, "target_realized_at"
            ]
            <= split_plan.train_cutoff_prediction_time
        )
        boundary_ok.loc[validation_mask] = (
            candidates.loc[
                validation_mask, "target_realized_at"
            ]
            <= split_plan.validation_cutoff_prediction_time
        )

        structural_month = (
            candidates.loc[boundary_ok]
            .copy()
            .reset_index(drop=True)
        )

        structural_month["sample_index"] = np.arange(
            sample_offset,
            sample_offset + len(structural_month),
            dtype="int64",
        )
        sample_offset += len(structural_month)

        structural_month = structural_month[
            SEQUENCE_INDEX_STRUCTURAL_COLUMNS
        ]

        if structural_month.duplicated(
            ["symbol", "source_id", "window_end_datetime"]
        ).any():
            raise ValueError(
                f"{key}: duplicate sequence endpoint keys."
            )
        if (
            structural_month["prediction_time"]
            <= structural_month["window_end_datetime"]
        ).any():
            raise AssertionError(
                f"{key}: prediction_time must be "
                "after the final input bar."
            )
        if (
            structural_month["target_realized_at"]
            <= structural_month["prediction_time"]
        ).any():
            raise AssertionError(
                f"{key}: target must be realized "
                "after prediction_time."
            )

        observed_structural_splits = set(
            structural_month["split"].unique()
        )
        if not observed_structural_splits <= {
            "train",
            "validation",
            "test",
        }:
            raise AssertionError(
                f"{key}: structural output contains disallowed split "
                "value(s) "
                f"{observed_structural_splits - {'train', 'validation', 'test'}}."
            )

        structural_path = (
            structural_parameter_root
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )
        atomic_write_parquet(
            structural_month,
            structural_path,
            compression=config.parquet_compression,
        )

        for split_name, count in (
            structural_month["split"]
            .value_counts()
            .to_dict()
            .items()
        ):
            structural_counts[split_name] = (
                structural_counts.get(split_name, 0)
                + int(count)
            )

        retained = structural_month.loc[
            structural_month["split"] != "test"
        ].copy()

        if len(retained):
            retained_trainval_by_month[key] = retained
            month_max = retained[
                "window_end_datetime"
            ].max()
            if (
                trainval_max_datetime is None
                or month_max > trainval_max_datetime
            ):
                trainval_max_datetime = month_max

        month_summaries.append(
            {
                "year": year,
                "month": month,
                "structural_rows_written": int(
                    len(structural_month)
                ),
                "trainval_rows_retained": int(
                    len(retained)
                ),
            }
        )

    if trainval_max_datetime is None:
        raise ValueError(
            "No TRAIN/VALIDATION rows survived split assignment "
            "and the boundary purge — cannot derive "
            "trainval_max_datetime."
        )

    # --- Pass 2: trainval outcome values, only for months with retained rows. ---
    for key, retained in sorted(
        retained_trainval_by_month.items()
    ):
        year, month = key

        outcome = read_target_outcome_values(
            target_parts[key],
            timezone=config.timezone,
            horizon=target_horizon,
            trainval_max_datetime=trainval_max_datetime,
            validation_last_trading_date=(
                split_plan.validation_last_trading_date
            ),
        )

        suffix_cols = {
            f"target_return_bps_{suffix}": "target_return_bps",
            f"target_log_return_{suffix}": "target_log_return",
            f"target_direction_{suffix}": "target_direction",
            f"target_same_session_{suffix}": "target_same_session",
            f"target_same_trading_date_{suffix}": (
                "target_same_trading_date"
            ),
        }
        outcome = outcome.rename(columns=suffix_cols)

        merged = retained.merge(
            outcome,
            how="left",
            left_on=["source_id", "window_end_datetime"],
            right_on=["source_id", "datetime"],
            validate="one_to_one",
        )

        missing = merged[
            "target_return_bps"
        ].isna()
        if missing.any():
            raise ValueError(
                f"{key}: {int(missing.sum())} retained TRAIN/VALIDATION "
                "row(s) had no matching target-outcome row within "
                "trainval_max_datetime."
            )

        finite_required = [
            "target_return_bps",
            "target_log_return",
            "target_direction",
        ]
        finite_values = merged[
            finite_required
        ].to_numpy(dtype="float64", na_value=np.nan)
        if not np.isfinite(finite_values).all():
            raise ValueError(
                f"{key}: non-finite target outcome values in retained "
                "TRAIN/VALIDATION rows."
            )

        merged = merged[
            SEQUENCE_INDEX_COLUMNS
        ]

        # Fail-closed, explicit — do not rely on `split != "test"` filtering
        # upstream as an implicit guarantee.
        observed_splits = set(
            merged["split"].unique()
        )
        if not observed_splits <= {
            "train",
            "validation",
        }:
            raise AssertionError(
                f"{key}: trainval output contains disallowed split "
                f"value(s) {observed_splits - {'train', 'validation'}}."
            )
        if (
            merged["split"] == "test"
        ).any():
            raise AssertionError(
                f"{key}: trainval output contains "
                "split == 'test' rows immediately before write."
            )

        trainval_path = (
            trainval_parameter_root
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )
        atomic_write_parquet(
            merged,
            trainval_path,
            compression=config.parquet_compression,
        )

        for split_name, count in (
            merged["split"]
            .value_counts()
            .to_dict()
            .items()
        ):
            trainval_counts[split_name] = (
                trainval_counts.get(split_name, 0)
                + int(count)
            )

    contract_hash = hashlib.sha256(
        json.dumps(
            {
                "structural_row_counts": structural_counts,
                "trainval_row_counts": trainval_counts,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    common_manifest_fields = {
        "artifact_contract_version": "v2",
        "sequence_version": SEQUENCE_VERSION,
        "symbol": symbol,
        "features_root": str(features_root),
        "targets_root": str(targets_root),
        "sequence_length": sequence_length,
        "target_horizon_minutes": target_horizon,
        "stride_minutes": stride_minutes,
        "scope": scope,
        "sessions": list(sessions),
        "train_fraction": split_plan.train_fraction,
        "validation_fraction": split_plan.validation_fraction,
        "test_fraction": split_plan.test_fraction,
        "train_last_trading_date": str(
            split_plan.train_last_trading_date
        ),
        "validation_last_trading_date": str(
            split_plan.validation_last_trading_date
        ),
        "train_cutoff_prediction_time": str(
            split_plan.train_cutoff_prediction_time
        ),
        "validation_cutoff_prediction_time": str(
            split_plan.validation_cutoff_prediction_time
        ),
        "trainval_max_datetime": str(
            trainval_max_datetime
        ),
        "structural_row_counts": structural_counts,
        "trainval_row_counts": trainval_counts,
        "configuration_sources": (
            configuration_sources or {}
        ),
        "generated_at": generated_at,
        "artifact_contract_hash": contract_hash,
    }

    structural_manifest = {
        "artifact_scope": "structural_all_splits",
        **common_manifest_fields,
    }
    trainval_manifest = {
        "artifact_scope": "trainval",
        **common_manifest_fields,
    }

    atomic_write_json(
        structural_manifest,
        structural_parameter_root
        / f"symbol={symbol}"
        / "manifest.json",
    )
    atomic_write_json(
        trainval_manifest,
        trainval_parameter_root
        / f"symbol={symbol}"
        / "manifest.json",
    )

    summary = {
        "sequence_version": SEQUENCE_VERSION,
        "artifact_contract_version": "v2",
        "artifact_contract_hash": contract_hash,
        "symbol": symbol,
        "structural_row_counts": structural_counts,
        "trainval_row_counts": trainval_counts,
        "trainval_max_datetime": str(
            trainval_max_datetime
        ),
        "month_summaries": month_summaries,
        "structural_output_root": str(
            structural_parameter_root
        ),
        "trainval_output_root": str(
            trainval_parameter_root
        ),
    }

    if reports_root is not None:
        report = (
            reports_root
            / symbol
            / (
                "sequence_summary_v2_"
                f"L{sequence_length}_"
                f"H{target_horizon}m_"
                f"{scope}_"
                f"S{stride_minutes}_"
                f"{sessions_tag}.json"
            )
        )
        atomic_write_json(
            summary,
            report,
        )
        summary["report_file"] = str(report)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe sequence endpoint/index store "
            "from causal feature and exact-horizon target Parquet. "
            "Sequence settings resolve as CLI > pipeline.yaml > "
            "script defaults."
        )
    )

    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--targets-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "v1 legacy output root. Requires --legacy-v1. Writes the "
            "full-value, all-splits sequence-index artifact — do not use "
            "for a trainer-facing build."
        ),
    )
    parser.add_argument(
        "--structural-output-root",
        type=Path,
        default=None,
        help=(
            "v2 (default/canonical) structural output root: every split, "
            "structural columns only, no target outcome VALUE for any row."
        ),
    )
    parser.add_argument(
        "--trainval-output-root",
        type=Path,
        default=None,
        help=(
            "v2 (default/canonical) TRAIN+VALIDATION-only value output "
            "root."
        ),
    )
    parser.add_argument(
        "--legacy-v1",
        action="store_true",
        help=(
            "Explicit opt-in to run the frozen v1 build path "
            "(process_symbol), which writes full target-outcome VALUES "
            "for every split, including TEST, into a single artifact. "
            "Requires --output-root. Without this flag, the v2 build path "
            "(process_symbol_v2) runs by default and requires "
            "--structural-output-root/--trainval-output-root instead."
        ),
    )
    parser.add_argument(
        "--symbol",
        default=None,
    )

    # None is intentional. It lets us distinguish an explicit CLI override
    # from a value that should come from pipeline.yaml or script defaults.
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help=(
            "CLI override for pipeline.sequences.sequence_length."
        ),
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=None,
        help=(
            "CLI override for "
            "pipeline.sequences.target_horizon_minutes."
        ),
    )
    parser.add_argument(
        "--stride-minutes",
        type=int,
        default=None,
        help=(
            "CLI override for pipeline.sequences.stride_minutes."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=sorted(
            VALID_SCOPES
        ),
        default=None,
        help=(
            "CLI override for pipeline.sequences.scope."
        ),
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help=(
            "CLI override for pipeline.sequences.sessions, "
            "for example: --sessions premarket regular aftermarket"
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=None,
        help=(
            "CLI override for "
            "pipeline.sequences.split.train_fraction."
        ),
    )
    parser.add_argument(
        "--validation-frac",
        type=float,
        default=None,
        help=(
            "CLI override for "
            "pipeline.sequences.split.validation_fraction."
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
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help=(
            "Resolve and print sequence configuration, then exit "
            "without reading feature/target data."
        ),
    )

    return parser.parse_args()


def _require_runtime_arg(
    value,
    name: str,
):
    if value is None:
        raise ValueError(
            f"{name} is required unless --show-config is used."
        )
    return value


def main() -> int:
    args = parse_args()

    settings, setting_sources = (
        resolve_sequence_settings(
            args,
            config_path=args.config,
        )
    )

    resolved_payload = (
        sequence_settings_payload(
            settings,
            setting_sources,
        )
    )

    if args.show_config:
        print(
            json.dumps(
                resolved_payload,
                indent=2,
                default=str,
            )
        )
        return 0

    features_root = _require_runtime_arg(
        args.features_root,
        "--features-root",
    )
    targets_root = _require_runtime_arg(
        args.targets_root,
        "--targets-root",
    )
    symbol = _require_runtime_arg(
        args.symbol,
        "--symbol",
    )

    config = load_config(
        args.config
    )

    print(
        "Resolved sequence configuration:"
    )
    print(
        json.dumps(
            resolved_payload,
            indent=2,
            default=str,
        )
    )

    if args.legacy_v1:
        output_root = _require_runtime_arg(
            args.output_root,
            "--output-root",
        )
        print(
            "--legacy-v1 explicitly requested: running the frozen v1 "
            "build path. This writes target-outcome VALUES for every "
            "split, including TEST. Do not point a trainer at this "
            "output."
        )
        summary = process_symbol(
            features_root=features_root,
            targets_root=targets_root,
            output_root=output_root,
            symbol=symbol,
            sequence_length=(
                settings.sequence_length
            ),
            target_horizon=(
                settings.target_horizon
            ),
            stride_minutes=(
                settings.stride_minutes
            ),
            scope=settings.scope,
            sessions=settings.sessions,
            train_fraction=(
                settings.train_fraction
            ),
            validation_fraction=(
                settings.validation_fraction
            ),
            config=config,
            reports_root=(
                args.reports
            ),
            configuration_sources=(
                setting_sources
            ),
        )
    else:
        structural_output_root = (
            _require_runtime_arg(
                args.structural_output_root,
                "--structural-output-root",
            )
        )
        trainval_output_root = (
            _require_runtime_arg(
                args.trainval_output_root,
                "--trainval-output-root",
            )
        )
        summary = process_symbol_v2(
            features_root=features_root,
            targets_root=targets_root,
            structural_output_root=(
                structural_output_root
            ),
            trainval_output_root=(
                trainval_output_root
            ),
            symbol=symbol,
            sequence_length=(
                settings.sequence_length
            ),
            target_horizon=(
                settings.target_horizon
            ),
            stride_minutes=(
                settings.stride_minutes
            ),
            scope=settings.scope,
            sessions=settings.sessions,
            train_fraction=(
                settings.train_fraction
            ),
            validation_fraction=(
                settings.validation_fraction
            ),
            config=config,
            reports_root=(
                args.reports
            ),
            configuration_sources=(
                setting_sources
            ),
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
    raise SystemExit(
        main()
    )