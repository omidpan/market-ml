from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
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

        ordered_columns = [
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
            "target_return_bps",
            "target_log_return",
            "target_direction",
            "target_same_session",
            "target_same_trading_date",
        ]

        output = output[
            ordered_columns
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
    output_root = _require_runtime_arg(
        args.output_root,
        "--output-root",
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