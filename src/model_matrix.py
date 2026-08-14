from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from common import load_config


MODEL_MATRIX_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Baseline feature policy
# ---------------------------------------------------------------------------
#
# core_v1 intentionally avoids the broadest/sparsest feature families.
#
# In particular, the first baseline does NOT require:
#   - higher-timeframe features,
#   - session rolling z-scores,
#   - session EMAs,
#   - SPY/context-symbol features,
#   - ACD/state features.
#
# Those can be added as controlled feature-set ablations later.
#
# The selected columns below are all causal features already produced by
# features.py v2.1. The model-matrix stage does not recompute indicators.
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_SET = "core_v1"
DEFAULT_SCALER = "standard"
DEFAULT_OUTPUT_DTYPE = "float32"

VALID_SCALERS = {"standard", "none"}
VALID_OUTPUT_DTYPES = {"float32", "float64"}

DEFAULT_FEATURES = (
    # Recent path / momentum at several 1-minute horizons.
    "f_1m_log_return_1",
    "f_1m_log_return_5",
    "f_1m_log_return_15",
    "f_1m_log_return_30",
    "f_1m_log_return_60",

    # Current-bar geometry / volatility.
    "f_1m_true_range_bps",
    "f_1m_body_bps",
    "f_1m_range_bps",
    "f_1m_upper_shadow_fraction",
    "f_1m_lower_shadow_fraction",
    "f_1m_body_to_range",
    "f_1m_close_wap_bps",

    # Activity without unsafe division by thin-session baselines.
    "f_1m_log_volume",
    "f_1m_log_bar_count",
    "f_1m_log_dollar_volume",

    # Session position and calendar-time state.
    "f_session_progress",
    "f_time_minute_sin",
    "f_time_minute_cos",
    "f_time_dow_sin",
    "f_time_dow_cos",
    "f_session_is_premarket",
    "f_session_is_regular",
    "f_session_is_aftermarket",
)

FEATURE_METADATA_COLUMNS = (
    "datetime",
    "prediction_time",
    "trading_date",
    "session",
    "symbol",
    "source_id",
    "is_current_bar_usable",
)

SEQUENCE_REQUIRED_COLUMNS = (
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
)

DEFAULT_SEQUENCE_LENGTH = 120
DEFAULT_TARGET_HORIZON = 15
DEFAULT_STRIDE_MINUTES = 1
DEFAULT_SCOPE = "continuous"
DEFAULT_SESSIONS = (
    "premarket",
    "regular",
    "aftermarket",
)


@dataclass(frozen=True)
class MatrixSettings:
    feature_set: str
    features: tuple[str, ...]
    scaler: str
    output_dtype: str


@dataclass(frozen=True)
class SequenceAddress:
    sequence_length: int
    target_horizon: int
    stride_minutes: int
    scope: str
    sessions: tuple[str, ...]

    @property
    def sessions_tag(self) -> str:
        return "+".join(self.sessions)

    def parameter_root(self, root: Path) -> Path:
        return (
            root
            / f"sequence_length={self.sequence_length}"
            / f"horizon={self.target_horizon}m"
            / f"scope={self.scope}"
            / f"stride={self.stride_minutes}"
            / f"sessions={self.sessions_tag}"
        )


@dataclass
class RunningMoments:
    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, width: int) -> "RunningMoments":
        return cls(
            count=0,
            mean=np.zeros(width, dtype=np.float64),
            m2=np.zeros(width, dtype=np.float64),
        )

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return

        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                f"Expected 2-D scaler batch; received shape={values.shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Scaler batch contains NaN or infinity."
            )

        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean = (
            self.mean
            + delta
            * (batch_count / total_count)
        )

        self.m2 = (
            self.m2
            + batch_m2
            + np.square(delta)
            * (
                self.count
                * batch_count
                / total_count
            )
        )

        self.count = total_count

    def population_variance(self) -> np.ndarray:
        if self.count <= 0:
            raise ValueError(
                "Cannot compute scaler variance with zero training rows."
            )
        return self.m2 / float(self.count)


def normalize_symbol(value: str) -> str:
    return str(value).strip().lower()


def normalize_string_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        items = [
            item.strip()
            for item in value.replace(",", " ").split()
            if item.strip()
        ]
    elif isinstance(value, (list, tuple)):
        items = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    else:
        raise ValueError(
            "Expected a string/list/tuple."
        )

    output = []
    seen = set()

    for item in items:
        normalized = item.lower()
        if normalized not in seen:
            output.append(normalized)
            seen.add(normalized)

    return tuple(output)


def feature_signature(
    feature_set: str,
    features: Iterable[str],
) -> str:
    payload = {
        "feature_set": str(feature_set),
        "features": list(features),
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def ensure_timezone(
    values: pd.Series,
    *,
    timezone: str,
    column: str,
) -> pd.Series:
    if isinstance(
        values.dtype,
        pd.DatetimeTZDtype,
    ):
        return values.dt.tz_convert(
            timezone
        )

    if pd.api.types.is_datetime64_dtype(
        values.dtype
    ):
        return values.dt.tz_localize(
            timezone,
            ambiguous="raise",
            nonexistent="raise",
        )

    return pd.to_datetime(
        values,
        errors="raise",
        utc=True,
    ).dt.tz_convert(
        timezone
    )


def normalize_date(
    values: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        values,
        errors="raise",
    ).dt.date


def parse_bool(
    values: pd.Series,
    column: str,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        values.dtype
    ):
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
            values.loc[
                parsed.isna()
            ]
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Invalid boolean values in {column}: {examples}"
        )

    return parsed.astype(bool)


def load_raw_yaml(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found: {path}"
        )

    return (
        yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )


def _coalesce(
    cli_value,
    yaml_value,
    default_value,
):
    if cli_value is not None:
        return cli_value, "cli"
    if yaml_value is not None:
        return yaml_value, "yaml"
    return default_value, "script_default"


def resolve_matrix_settings(
    args: argparse.Namespace,
    raw_yaml: dict,
) -> tuple[MatrixSettings, dict[str, str]]:
    section = (
        raw_yaml
        .get("pipeline", {})
        .get("model_matrix", {})
        or {}
    )

    feature_set, source_feature_set = _coalesce(
        args.feature_set,
        section.get("feature_set"),
        DEFAULT_FEATURE_SET,
    )

    yaml_features = section.get(
        "features"
    )

    cli_features = (
        tuple(args.features)
        if args.features is not None
        else None
    )

    feature_values, source_features = _coalesce(
        cli_features,
        yaml_features,
        DEFAULT_FEATURES,
    )

    features = tuple(
        str(value).strip()
        for value in feature_values
        if str(value).strip()
    )

    if not features:
        raise ValueError(
            "model_matrix requires at least one feature."
        )

    duplicates = sorted(
        {
            feature
            for feature in features
            if features.count(feature) > 1
        }
    )
    if duplicates:
        raise ValueError(
            f"Duplicate model features: {duplicates}"
        )

    invalid_names = [
        feature
        for feature in features
        if not feature.startswith("f_")
    ]
    if invalid_names:
        raise ValueError(
            "Model feature names must be feature-store columns "
            f"beginning with f_: {invalid_names}"
        )

    scaler, source_scaler = _coalesce(
        args.scaler,
        section.get("scaler"),
        DEFAULT_SCALER,
    )
    scaler = str(scaler).strip().lower()

    if scaler not in VALID_SCALERS:
        raise ValueError(
            f"Unsupported scaler={scaler!r}; "
            f"expected one of {sorted(VALID_SCALERS)}"
        )

    output_dtype, source_dtype = _coalesce(
        args.output_dtype,
        section.get("output_dtype"),
        DEFAULT_OUTPUT_DTYPE,
    )
    output_dtype = str(
        output_dtype
    ).strip().lower()

    if (
        output_dtype
        not in VALID_OUTPUT_DTYPES
    ):
        raise ValueError(
            f"Unsupported output_dtype={output_dtype!r}; "
            f"expected one of {sorted(VALID_OUTPUT_DTYPES)}"
        )

    return (
        MatrixSettings(
            feature_set=(
                str(feature_set)
                .strip()
                .lower()
            ),
            features=features,
            scaler=scaler,
            output_dtype=output_dtype,
        ),
        {
            "feature_set": source_feature_set,
            "features": source_features,
            "scaler": source_scaler,
            "output_dtype": source_dtype,
        },
    )


def resolve_sequence_address(
    raw_yaml: dict,
) -> SequenceAddress:
    section = (
        raw_yaml
        .get("pipeline", {})
        .get("sequences", {})
        or {}
    )

    sequence_length = int(
        section.get(
            "sequence_length",
            DEFAULT_SEQUENCE_LENGTH,
        )
    )
    target_horizon = int(
        section.get(
            "target_horizon_minutes",
            DEFAULT_TARGET_HORIZON,
        )
    )
    stride_minutes = int(
        section.get(
            "stride_minutes",
            DEFAULT_STRIDE_MINUTES,
        )
    )
    scope = str(
        section.get(
            "scope",
            DEFAULT_SCOPE,
        )
    ).strip().lower()

    sessions = normalize_string_tuple(
        section.get(
            "sessions",
            DEFAULT_SESSIONS,
        )
    )

    if sequence_length < 1:
        raise ValueError(
            "pipeline.sequences.sequence_length must be >= 1."
        )
    if target_horizon < 1:
        raise ValueError(
            "pipeline.sequences.target_horizon_minutes must be >= 1."
        )
    if stride_minutes < 1:
        raise ValueError(
            "pipeline.sequences.stride_minutes must be >= 1."
        )
    if not sessions:
        raise ValueError(
            "pipeline.sequences.sessions cannot be empty."
        )

    return SequenceAddress(
        sequence_length=sequence_length,
        target_horizon=target_horizon,
        stride_minutes=stride_minutes,
        scope=scope,
        sessions=sessions,
    )


def discover_parts(
    root: Path,
    *,
    symbol: str,
) -> dict[tuple[int, int], Path]:
    base = (
        root
        / f"symbol={normalize_symbol(symbol)}"
    )

    if not base.exists():
        raise FileNotFoundError(
            f"Partition root does not exist: {base}"
        )

    result: dict[
        tuple[int, int],
        Path,
    ] = {}

    for path in sorted(
        base.glob(
            "year=*/month=*/part.parquet"
        )
    ):
        try:
            year = int(
                path.parent.parent.name
                .split("=", 1)[1]
            )
            month = int(
                path.parent.name
                .split("=", 1)[1]
            )
        except (
            IndexError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid partition path: {path}"
            ) from exc

        key = (
            year,
            month,
        )

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


def validate_feature_schema(
    path: Path,
    *,
    selected_features: tuple[str, ...],
) -> list[str]:
    columns = list(
        pq.read_schema(
            path
        ).names
    )

    required = [
        *FEATURE_METADATA_COLUMNS,
        *selected_features,
    ]

    missing = [
        column
        for column in required
        if column not in columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing model-matrix columns: {missing}"
        )

    return columns


def read_feature_part(
    path: Path,
    *,
    symbol: str,
    timezone: str,
    selected_features: tuple[str, ...],
) -> pd.DataFrame:
    columns = [
        *FEATURE_METADATA_COLUMNS,
        *selected_features,
    ]

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
        columns=columns,
    ).copy()

    frame["datetime"] = (
        ensure_timezone(
            frame["datetime"],
            timezone=timezone,
            column="datetime",
        )
    )
    frame["prediction_time"] = (
        ensure_timezone(
            frame["prediction_time"],
            timezone=timezone,
            column="prediction_time",
        )
    )
    frame["trading_date"] = (
        normalize_date(
            frame["trading_date"]
        )
    )

    frame[
        "is_current_bar_usable"
    ] = parse_bool(
        frame[
            "is_current_bar_usable"
        ],
        "is_current_bar_usable",
    )

    frame["symbol"] = (
        frame["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["source_id"] = (
        frame["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["session"] = (
        frame["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected_symbol = (
        normalize_symbol(symbol)
    )
    found_symbols = (
        frame["symbol"]
        .dropna()
        .unique()
    )
    if (
        len(found_symbols) != 1
        or found_symbols[0]
        != expected_symbol
    ):
        raise ValueError(
            f"Expected symbol={expected_symbol!r}; "
            f"found={found_symbols.tolist()}"
        )

    if frame.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            f"{path}: duplicate feature (datetime, source_id) keys."
        )

    expected_prediction = (
        frame["datetime"]
        + timedelta(minutes=1)
    )
    if (
        frame["prediction_time"]
        != expected_prediction
    ).any():
        raise ValueError(
            f"{path}: prediction_time must equal datetime + 1 minute."
        )

    for feature in selected_features:
        frame[feature] = (
            pd.to_numeric(
                frame[feature],
                errors="coerce",
            )
            .astype(float)
        )

    return (
        frame.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def read_sequence_part(
    path: Path,
    *,
    symbol: str,
    timezone: str,
    address: SequenceAddress,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    ).copy()

    missing = [
        column
        for column in SEQUENCE_REQUIRED_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{path} missing sequence-index columns: {missing}"
        )

    frame = frame[
        list(SEQUENCE_REQUIRED_COLUMNS)
    ].copy()

    for column in (
        "window_start_datetime",
        "window_end_datetime",
        "prediction_time",
        "target_realized_at",
    ):
        frame[column] = (
            ensure_timezone(
                frame[column],
                timezone=timezone,
                column=column,
            )
        )

    for column in (
        "calendar_date",
        "trading_date",
    ):
        frame[column] = (
            normalize_date(
                frame[column]
            )
        )

    for column in (
        "target_same_session",
        "target_same_trading_date",
    ):
        frame[column] = parse_bool(
            frame[column],
            column,
        )

    for column in (
        "target_return_bps",
        "target_log_return",
        "target_direction",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["symbol"] = (
        frame["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["source_id"] = (
        frame["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["session"] = (
        frame["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["split"] = (
        frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["sequence_scope"] = (
        frame["sequence_scope"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected_symbol = (
        normalize_symbol(symbol)
    )
    found_symbols = (
        frame["symbol"]
        .dropna()
        .unique()
    )
    if (
        len(found_symbols) != 1
        or found_symbols[0]
        != expected_symbol
    ):
        raise ValueError(
            f"Expected sequence symbol={expected_symbol!r}; "
            f"found={found_symbols.tolist()}"
        )

    checks = {
        "sequence_length": (
            frame[
                "sequence_length"
            ]
            .eq(address.sequence_length)
        ),
        "target_horizon_minutes": (
            frame[
                "target_horizon_minutes"
            ]
            .eq(address.target_horizon)
        ),
        "stride_minutes": (
            frame[
                "stride_minutes"
            ]
            .eq(address.stride_minutes)
        ),
        "sequence_scope": (
            frame[
                "sequence_scope"
            ]
            .eq(address.scope)
        ),
    }

    failed = [
        name
        for name, ok
        in checks.items()
        if not ok.all()
    ]
    if failed:
        raise ValueError(
            f"{path}: sequence configuration mismatch: {failed}"
        )

    valid_splits = {
        "train",
        "validation",
        "test",
    }
    unexpected_splits = sorted(
        set(
            frame["split"].unique()
        )
        - valid_splits
    )
    if unexpected_splits:
        raise ValueError(
            f"{path}: unexpected splits {unexpected_splits}"
        )

    if frame.duplicated(
        [
            "symbol",
            "source_id",
            "window_end_datetime",
        ]
    ).any():
        raise ValueError(
            f"{path}: duplicate sequence endpoint keys."
        )

    expected_start = (
        frame[
            "window_end_datetime"
        ]
        - timedelta(
            minutes=(
                address.sequence_length
                - 1
            )
        )
    )
    if (
        frame[
            "window_start_datetime"
        ]
        != expected_start
    ).any():
        raise ValueError(
            f"{path}: sequence window timestamp length mismatch."
        )

    target_values = frame[
        [
            "target_return_bps",
            "target_log_return",
            "target_direction",
        ]
    ].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if (
        len(target_values)
        and not np.isfinite(
            target_values
        ).all()
    ):
        raise ValueError(
            f"{path}: sequence index contains non-finite targets."
        )

    return (
        frame.sort_values(
            [
                "source_id",
                "window_end_datetime",
            ]
        )
        .reset_index(drop=True)
    )


def feature_row_finite_mask(
    frame: pd.DataFrame,
    selected_features: tuple[str, ...],
) -> np.ndarray:
    values = frame[
        list(selected_features)
    ].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    return (
        np.isfinite(values)
        .all(axis=1)
    )


def add_window_finite_flag(
    frame: pd.DataFrame,
    *,
    selected_features: tuple[str, ...],
    sequence_length: int,
) -> pd.DataFrame:
    """
    Mark whether the final sequence_length rows ending at each row contain
    finite values for every selected feature.

    The sequence-index stage remains the source of truth for physical
    one-minute continuity. This rolling finite check deliberately does not
    try to reconstruct continuity; it only adds the selected-feature
    requirement to endpoints already proven structurally valid.
    """
    work = frame.copy()

    row_finite = feature_row_finite_mask(
        work,
        selected_features,
    )

    work[
        "_model_row_finite"
    ] = row_finite

    work[
        "_model_window_finite"
    ] = False

    for (
        source_id,
        group,
    ) in work.groupby(
        "source_id",
        sort=False,
    ):
        idx = group.index.to_numpy()
        flags = (
            work.loc[
                idx,
                "_model_row_finite",
            ]
            .to_numpy(
                dtype=np.int8
            )
        )

        if len(flags) < sequence_length:
            continue

        cumulative = np.concatenate(
            [
                np.array(
                    [0],
                    dtype=np.int64,
                ),
                np.cumsum(
                    flags,
                    dtype=np.int64,
                ),
            ]
        )

        counts = (
            cumulative[
                sequence_length:
            ]
            - cumulative[
                :-sequence_length
            ]
        )

        window_ok = (
            counts
            == sequence_length
        )

        endpoint_positions = idx[
            sequence_length - 1:
        ]

        work.loc[
            endpoint_positions,
            "_model_window_finite",
        ] = window_ok

    return work


def load_feature_month_with_context(
    *,
    current_path: Path,
    previous_path: Path | None,
    symbol: str,
    timezone: str,
    selected_features: tuple[str, ...],
    sequence_length: int,
) -> pd.DataFrame:
    current = read_feature_part(
        current_path,
        symbol=symbol,
        timezone=timezone,
        selected_features=(
            selected_features
        ),
    )
    current[
        "_is_current_month"
    ] = True

    frames = []

    if (
        previous_path is not None
        and sequence_length > 1
    ):
        previous = read_feature_part(
            previous_path,
            symbol=symbol,
            timezone=timezone,
            selected_features=(
                selected_features
            ),
        )

        previous = (
            previous.sort_values(
                [
                    "source_id",
                    "datetime",
                ]
            )
            .groupby(
                "source_id",
                group_keys=False,
                sort=False,
            )
            .tail(
                sequence_length
                - 1
            )
            .reset_index(drop=True)
        )

        previous[
            "_is_current_month"
        ] = False
        frames.append(
            previous
        )

    frames.append(
        current
    )

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


def compute_scaler(
    *,
    feature_parts: dict[tuple[int, int], Path],
    symbol: str,
    timezone: str,
    selected_features: tuple[str, ...],
    allowed_sessions: tuple[str, ...],
    train_last_trading_date,
    scaler: str,
) -> dict:
    width = len(
        selected_features
    )

    moments = (
        RunningMoments.empty(
            width
        )
    )

    eligible_rows = 0

    for key in sorted(
        feature_parts
    ):
        frame = read_feature_part(
            feature_parts[key],
            symbol=symbol,
            timezone=timezone,
            selected_features=(
                selected_features
            ),
        )

        row_finite = (
            feature_row_finite_mask(
                frame,
                selected_features,
            )
        )

        train_mask = (
            frame[
                "is_current_bar_usable"
            ]
            & frame[
                "session"
            ].isin(
                allowed_sessions
            )
            & (
                frame[
                    "trading_date"
                ]
                <= train_last_trading_date
            )
            & row_finite
        )

        values = (
            frame.loc[
                train_mask,
                list(
                    selected_features
                ),
            ]
            .to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
        )

        moments.update(
            values
        )

        eligible_rows += int(
            train_mask.sum()
        )

    if (
        eligible_rows
        != moments.count
    ):
        raise AssertionError(
            "Scaler row-count accounting mismatch."
        )

    if moments.count <= 0:
        raise ValueError(
            "No finite training rows available to fit scaler."
        )

    if scaler == "standard":
        variance = (
            moments.population_variance()
        )

        # Guard against tiny negative values from floating-point roundoff.
        variance = np.maximum(
            variance,
            0.0,
        )

        scale = np.sqrt(
            variance
        )

        scale = np.asarray(
            scale,
            dtype=np.float64,
        )
        constant_mask = (
            ~np.isfinite(scale)
            | (scale == 0.0)
        )
        scale[
            constant_mask
        ] = 1.0

        mean = np.asarray(
            moments.mean,
            dtype=np.float64,
        )

    elif scaler == "none":
        mean = np.zeros(
            width,
            dtype=np.float64,
        )
        scale = np.ones(
            width,
            dtype=np.float64,
        )
        variance = np.ones(
            width,
            dtype=np.float64,
        )
        constant_mask = np.zeros(
            width,
            dtype=bool,
        )
    else:
        raise ValueError(
            f"Unsupported scaler={scaler!r}"
        )

    return {
        "scaler": scaler,
        "fit_policy": (
            "Unique finite usable feature rows whose trading_date is no later "
            "than the final training trading_date and whose session belongs "
            "to the configured sequence session set. Validation/test rows are "
            "never used to fit scaling parameters."
        ),
        "fit_rows": int(
            moments.count
        ),
        "train_last_trading_date": (
            str(
                train_last_trading_date
            )
        ),
        "feature_names": list(
            selected_features
        ),
        "mean": {
            feature: float(value)
            for feature, value
            in zip(
                selected_features,
                mean,
            )
        },
        "scale": {
            feature: float(value)
            for feature, value
            in zip(
                selected_features,
                scale,
            )
        },
        "variance": {
            feature: float(value)
            for feature, value
            in zip(
                selected_features,
                variance,
            )
        },
        "constant_features": [
            feature
            for feature, is_constant
            in zip(
                selected_features,
                constant_mask,
            )
            if bool(is_constant)
        ],
    }


def scaler_arrays(
    scaler_payload: dict,
    selected_features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array(
        [
            scaler_payload[
                "mean"
            ][feature]
            for feature
            in selected_features
        ],
        dtype=np.float64,
    )

    scale = np.array(
        [
            scaler_payload[
                "scale"
            ][feature]
            for feature
            in selected_features
        ],
        dtype=np.float64,
    )

    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "Invalid scaler parameters."
        )

    return (
        mean,
        scale,
    )


def transform_values(
    values: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    output_dtype: str,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    transformed = (
        values
        - mean
    ) / scale

    if not np.isfinite(
        transformed
    ).all():
        raise ValueError(
            "Scaled model matrix contains NaN or infinity."
        )

    return transformed.astype(
        output_dtype,
        copy=False,
    )


def model_parameter_root(
    output_root: Path,
    *,
    address: SequenceAddress,
    feature_set: str,
) -> Path:
    return (
        address.parameter_root(
            output_root
        )
        / f"feature_set={feature_set}"
    )


def process_symbol(
    *,
    features_root: Path,
    sequence_root: Path,
    output_root: Path,
    symbol: str,
    settings: MatrixSettings,
    address: SequenceAddress,
    config,
    reports_root: Path | None,
    configuration_sources: dict[str, str],
) -> dict:
    symbol = normalize_symbol(
        symbol
    )

    feature_parts = (
        discover_parts(
            features_root,
            symbol=symbol,
        )
    )

    sequence_parameter_root = (
        address.parameter_root(
            sequence_root
        )
    )

    sequence_parts = (
        discover_parts(
            sequence_parameter_root,
            symbol=symbol,
        )
    )

    feature_keys = sorted(
        feature_parts
    )
    sequence_keys = sorted(
        sequence_parts
    )

    if feature_keys != sequence_keys:
        missing_sequences = sorted(
            set(feature_keys)
            - set(sequence_keys)
        )
        extra_sequences = sorted(
            set(sequence_keys)
            - set(feature_keys)
        )

        raise ValueError(
            "Feature/sequence monthly partition mismatch. "
            f"missing_sequences={missing_sequences}, "
            f"extra_sequences={extra_sequences}"
        )

    # Schema audit before any output is written.
    reference_feature_schema = None

    for key in feature_keys:
        schema = (
            validate_feature_schema(
                feature_parts[key],
                selected_features=(
                    settings.features
                ),
            )
        )

        feature_only = tuple(
            column
            for column in schema
            if column.startswith(
                "f_"
            )
        )

        if (
            reference_feature_schema
            is None
        ):
            reference_feature_schema = (
                feature_only
            )
        elif (
            feature_only
            != reference_feature_schema
        ):
            raise ValueError(
                f"{key}: feature-store schema changed between months."
            )

    parameter_root = (
        model_parameter_root(
            output_root,
            address=address,
            feature_set=(
                settings.feature_set
            ),
        )
    )

    feature_hash = (
        feature_signature(
            settings.feature_set,
            settings.features,
        )
    )

    sequence_totals = {
        "input_sequence_rows": 0,
        "model_ready_sequence_rows": 0,
        "nonfinite_window_rejected_rows": 0,
        "split_input_rows": {
            "train": 0,
            "validation": 0,
            "test": 0,
        },
        "split_model_ready_rows": {
            "train": 0,
            "validation": 0,
            "test": 0,
        },
        "split_nonfinite_window_rejected_rows": {
            "train": 0,
            "validation": 0,
            "test": 0,
        },
        "direction_counts": {
            "train": {},
            "validation": {},
            "test": {},
        },
    }

    month_summaries = []
    ready_sequence_parts: dict[
        tuple[int, int],
        Path,
    ] = {}

    model_sample_offset = 0
    original_train_dates = []

    previous_key = None

    # ---------------------------------------------------------------------
    # Pass 1: selected-feature window eligibility + ready sequence index
    # ---------------------------------------------------------------------
    for key in feature_keys:
        year, month = key

        previous_feature_path = (
            feature_parts[
                previous_key
            ]
            if (
                previous_key
                is not None
            )
            else None
        )

        feature_context = (
            load_feature_month_with_context(
                current_path=(
                    feature_parts[key]
                ),
                previous_path=(
                    previous_feature_path
                ),
                symbol=symbol,
                timezone=config.timezone,
                selected_features=(
                    settings.features
                ),
                sequence_length=(
                    address.sequence_length
                ),
            )
        )

        feature_context = (
            add_window_finite_flag(
                feature_context,
                selected_features=(
                    settings.features
                ),
                sequence_length=(
                    address.sequence_length
                ),
            )
        )

        endpoints = (
            feature_context[
                [
                    "source_id",
                    "datetime",
                    "_model_window_finite",
                ]
            ]
            .rename(
                columns={
                    "datetime": (
                        "window_end_datetime"
                    )
                }
            )
        )

        sequences = (
            read_sequence_part(
                sequence_parts[key],
                symbol=symbol,
                timezone=config.timezone,
                address=address,
            )
        )

        train_dates = (
            sequences.loc[
                sequences[
                    "split"
                ].eq("train"),
                "trading_date",
            ]
            .dropna()
            .tolist()
        )
        original_train_dates.extend(
            train_dates
        )

        merged = sequences.merge(
            endpoints,
            how="left",
            on=[
                "source_id",
                "window_end_datetime",
            ],
            validate="one_to_one",
        )

        if merged[
            "_model_window_finite"
        ].isna().any():
            examples = (
                merged.loc[
                    merged[
                        "_model_window_finite"
                    ].isna(),
                    [
                        "source_id",
                        "window_end_datetime",
                    ],
                ]
                .head(5)
                .to_dict(
                    orient="records"
                )
            )
            raise ValueError(
                f"{key}: sequence endpoints not found in feature store: "
                f"{examples}"
            )

        model_ready_mask = (
            merged[
                "_model_window_finite"
            ].astype(bool)
        )

        ready = (
            merged.loc[
                model_ready_mask
            ]
            .drop(
                columns=[
                    "_model_window_finite"
                ]
            )
            .copy()
            .reset_index(drop=True)
        )

        ready[
            "model_sample_index"
        ] = np.arange(
            model_sample_offset,
            model_sample_offset
            + len(ready),
            dtype=np.int64,
        )
        model_sample_offset += len(
            ready
        )

        ready[
            "feature_set"
        ] = settings.feature_set

        ready[
            "feature_signature"
        ] = feature_hash

        # Keep the model-local id first while preserving the original
        # sequence sample_index for lineage back to sequences.py.
        ordered_ready = [
            "model_sample_index",
            "sample_index",
            "feature_set",
            "feature_signature",
            *[
                column
                for column
                in SEQUENCE_REQUIRED_COLUMNS
                if column
                != "sample_index"
            ],
        ]

        ready = ready[
            ordered_ready
        ]

        ready_path = (
            parameter_root
            / "sequences"
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )

        atomic_write_parquet(
            ready,
            ready_path,
            compression=(
                config.parquet_compression
            ),
        )

        ready_sequence_parts[
            key
        ] = ready_path

        input_split = (
            sequences[
                "split"
            ]
            .value_counts()
            .to_dict()
        )
        ready_split = (
            ready[
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        rejected = (
            merged.loc[
                ~model_ready_mask,
                "split",
            ]
            .value_counts()
            .to_dict()
        )

        direction_counts = {}

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            split_directions = (
                ready.loc[
                    ready[
                        "split"
                    ].eq(
                        split_name
                    ),
                    "target_direction",
                ]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            normalized_counts = {
                str(
                    int(label)
                    if float(label).is_integer()
                    else label
                ): int(count)
                for label, count
                in split_directions.items()
            }

            direction_counts[
                split_name
            ] = normalized_counts

            aggregate = (
                sequence_totals[
                    "direction_counts"
                ][
                    split_name
                ]
            )

            for label, count in (
                normalized_counts.items()
            ):
                aggregate[
                    label
                ] = (
                    aggregate.get(
                        label,
                        0,
                    )
                    + int(count)
                )

        month_summary = {
            "year": year,
            "month": month,
            "input_sequence_rows": int(
                len(sequences)
            ),
            "model_ready_sequence_rows": int(
                len(ready)
            ),
            "nonfinite_window_rejected_rows": int(
                (
                    ~model_ready_mask
                ).sum()
            ),
            "split_input_rows": {
                split_name: int(
                    input_split.get(
                        split_name,
                        0,
                    )
                )
                for split_name
                in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "split_model_ready_rows": {
                split_name: int(
                    ready_split.get(
                        split_name,
                        0,
                    )
                )
                for split_name
                in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "split_nonfinite_window_rejected_rows": {
                split_name: int(
                    rejected.get(
                        split_name,
                        0,
                    )
                )
                for split_name
                in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "direction_counts": (
                direction_counts
            ),
            "ready_sequence_file": str(
                ready_path
            ),
        }

        month_summaries.append(
            month_summary
        )

        sequence_totals[
            "input_sequence_rows"
        ] += len(sequences)
        sequence_totals[
            "model_ready_sequence_rows"
        ] += len(ready)
        sequence_totals[
            "nonfinite_window_rejected_rows"
        ] += int(
            (
                ~model_ready_mask
            ).sum()
        )

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            sequence_totals[
                "split_input_rows"
            ][
                split_name
            ] += int(
                input_split.get(
                    split_name,
                    0,
                )
            )

            sequence_totals[
                "split_model_ready_rows"
            ][
                split_name
            ] += int(
                ready_split.get(
                    split_name,
                    0,
                )
            )

            sequence_totals[
                "split_nonfinite_window_rejected_rows"
            ][
                split_name
            ] += int(
                rejected.get(
                    split_name,
                    0,
                )
            )

        print(
            f"[{year:04d}-{month:02d}] "
            f"sequences={len(sequences):,}, "
            f"model_ready={len(ready):,}, "
            f"rejected_nonfinite="
            f"{int((~model_ready_mask).sum()):,}"
        )

        previous_key = key

    if not original_train_dates:
        raise ValueError(
            "Sequence index contains no training rows."
        )

    train_last_trading_date = max(
        original_train_dates
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        if (
            sequence_totals[
                "split_model_ready_rows"
            ][split_name]
            <= 0
        ):
            raise ValueError(
                f"No model-ready sequences in {split_name} split."
            )

    # ---------------------------------------------------------------------
    # Pass 2: fit scaler using TRAINING-PERIOD rows only
    # ---------------------------------------------------------------------
    scaler_payload = compute_scaler(
        feature_parts=feature_parts,
        symbol=symbol,
        timezone=config.timezone,
        selected_features=(
            settings.features
        ),
        allowed_sessions=(
            address.sessions
        ),
        train_last_trading_date=(
            train_last_trading_date
        ),
        scaler=settings.scaler,
    )

    scaler_payload.update(
        {
            "model_matrix_version": (
                MODEL_MATRIX_VERSION
            ),
            "feature_set": (
                settings.feature_set
            ),
            "feature_signature": (
                feature_hash
            ),
            "output_dtype": (
                settings.output_dtype
            ),
        }
    )

    scaler_path = (
        parameter_root
        / "scaler.json"
    )

    atomic_write_json(
        scaler_payload,
        scaler_path,
    )

    mean, scale = (
        scaler_arrays(
            scaler_payload,
            settings.features,
        )
    )

    # ---------------------------------------------------------------------
    # Pass 3: write compact scaled row matrix.
    #
    # Only finite, usable rows from configured sessions are written.
    # Every model-ready sequence is guaranteed to find exactly L rows in
    # this store. Non-finite/warm-up rows remain represented in the original
    # broad feature store, not fabricated here.
    # ---------------------------------------------------------------------
    matrix_rows = 0
    month_matrix_rows = {}
    feature_nonfinite_counts = {
        feature: 0
        for feature
        in settings.features
    }

    for key in feature_keys:
        year, month = key

        frame = read_feature_part(
            feature_parts[key],
            symbol=symbol,
            timezone=config.timezone,
            selected_features=(
                settings.features
            ),
        )

        values = frame[
            list(
                settings.features
            )
        ].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        finite_by_feature = (
            np.isfinite(
                values
            )
        )

        for (
            position,
            feature,
        ) in enumerate(
            settings.features
        ):
            feature_nonfinite_counts[
                feature
            ] += int(
                (
                    ~finite_by_feature[
                        :,
                        position,
                    ]
                ).sum()
            )

        row_finite = (
            finite_by_feature.all(
                axis=1
            )
        )

        matrix_mask = (
            frame[
                "is_current_bar_usable"
            ]
            & frame[
                "session"
            ].isin(
                address.sessions
            )
            & row_finite
        )

        matrix = (
            frame.loc[
                matrix_mask,
                [
                    "datetime",
                    "prediction_time",
                    "trading_date",
                    "session",
                    "symbol",
                    "source_id",
                ],
            ]
            .copy()
            .reset_index(drop=True)
        )

        raw_values = (
            frame.loc[
                matrix_mask,
                list(
                    settings.features
                ),
            ]
            .to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
        )

        transformed = (
            transform_values(
                raw_values,
                mean=mean,
                scale=scale,
                output_dtype=(
                    settings.output_dtype
                ),
            )
        )

        for position, feature in enumerate(
            settings.features
        ):
            matrix[
                feature
            ] = transformed[
                :,
                position,
            ]

        matrix[
            "feature_set"
        ] = settings.feature_set
        matrix[
            "feature_signature"
        ] = feature_hash

        ordered_matrix = [
            "datetime",
            "prediction_time",
            "trading_date",
            "session",
            "symbol",
            "source_id",
            "feature_set",
            "feature_signature",
            *settings.features,
        ]

        matrix = matrix[
            ordered_matrix
        ]

        if matrix.duplicated(
            [
                "datetime",
                "source_id",
            ]
        ).any():
            raise ValueError(
                f"{key}: duplicate model matrix row keys."
            )

        matrix_values = matrix[
            list(
                settings.features
            )
        ].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if (
            len(matrix_values)
            and not np.isfinite(
                matrix_values
            ).all()
        ):
            raise AssertionError(
                f"{key}: non-finite values survived model-matrix filtering."
            )

        matrix_path = (
            parameter_root
            / "rows"
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )

        atomic_write_parquet(
            matrix,
            matrix_path,
            compression=(
                config.parquet_compression
            ),
        )

        month_matrix_rows[
            key
        ] = int(
            len(matrix)
        )
        matrix_rows += len(
            matrix
        )

    # Attach matrix-row counts to the already-built month summary list.
    for month_summary in (
        month_summaries
    ):
        key = (
            int(
                month_summary[
                    "year"
                ]
            ),
            int(
                month_summary[
                    "month"
                ]
            ),
        )
        month_summary[
            "matrix_rows"
        ] = int(
            month_matrix_rows[
                key
            ]
        )

    summary = {
        "model_matrix_version": (
            MODEL_MATRIX_VERSION
        ),
        "symbol": symbol,
        "feature_set": (
            settings.feature_set
        ),
        "feature_signature": (
            feature_hash
        ),
        "feature_count": len(
            settings.features
        ),
        "features": list(
            settings.features
        ),
        "scaler": (
            settings.scaler
        ),
        "output_dtype": (
            settings.output_dtype
        ),
        "configuration_precedence": (
            "CLI override > pipeline.yaml > script default"
        ),
        "configuration_sources": (
            configuration_sources
        ),
        "sequence_configuration": {
            "sequence_length": (
                address.sequence_length
            ),
            "target_horizon_minutes": (
                address.target_horizon
            ),
            "stride_minutes": (
                address.stride_minutes
            ),
            "scope": address.scope,
            "sessions": list(
                address.sessions
            ),
        },
        "feature_policy": (
            "The broad causal feature store remains authoritative. "
            "This model matrix selects only the configured feature subset. "
            "A sequence is model-ready only when every selected feature is "
            "finite on every input row of the full sequence window. No rows "
            "are dropped before temporal sequence validation and no missing "
            "feature value is filled here."
        ),
        "scaler_policy": (
            scaler_payload[
                "fit_policy"
            ]
        ),
        "feature_root": str(
            features_root
        ),
        "sequence_root": str(
            sequence_parameter_root
        ),
        "output_root": str(
            parameter_root
        ),
        "scaler_file": str(
            scaler_path
        ),
        "months": len(
            feature_keys
        ),
        "matrix_rows": int(
            matrix_rows
        ),
        "scaler_fit_rows": int(
            scaler_payload[
                "fit_rows"
            ]
        ),
        "scaler_train_last_trading_date": (
            scaler_payload[
                "train_last_trading_date"
            ]
        ),
        "constant_features": (
            scaler_payload[
                "constant_features"
            ]
        ),
        "feature_nonfinite_rows_all_timeline": {
            feature: int(count)
            for feature, count
            in feature_nonfinite_counts.items()
        },
        **sequence_totals,
        "month_summaries": (
            month_summaries
        ),
    }

    report_path = None

    if reports_root is not None:
        report_path = (
            reports_root
            / symbol
            / (
                "model_matrix_summary_"
                f"v{MODEL_MATRIX_VERSION}_"
                f"{settings.feature_set}_"
                f"L{address.sequence_length}_"
                f"H{address.target_horizon}m_"
                f"{address.scope}_"
                f"S{address.stride_minutes}.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only-scaled model feature matrix and "
            "model-ready sequence index from the causal feature store "
            "and sequences.py output."
        )
    )

    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=None,
        help=(
            "Base sequence_index root. The active sequence parameter path "
            "is derived from pipeline.sequences."
        ),
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

    # None defaults preserve CLI > YAML > script-default precedence.
    parser.add_argument(
        "--feature-set",
        default=None,
        help=(
            "CLI override for pipeline.model_matrix.feature_set."
        ),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "CLI override for pipeline.model_matrix.features."
        ),
    )
    parser.add_argument(
        "--scaler",
        choices=sorted(
            VALID_SCALERS
        ),
        default=None,
        help=(
            "CLI override for pipeline.model_matrix.scaler."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=sorted(
            VALID_OUTPUT_DTYPES
        ),
        default=None,
        help=(
            "CLI override for pipeline.model_matrix.output_dtype."
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
            "Resolve model-matrix and sequence addressing configuration "
            "then exit without reading Parquet."
        ),
    )

    return parser.parse_args()


def _require(
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

    raw_yaml = load_raw_yaml(
        args.config
    )

    settings, sources = (
        resolve_matrix_settings(
            args,
            raw_yaml,
        )
    )

    address = (
        resolve_sequence_address(
            raw_yaml
        )
    )

    resolved = {
        "model_matrix_version": (
            MODEL_MATRIX_VERSION
        ),
        "configuration_precedence": (
            "CLI override > pipeline.yaml > script default"
        ),
        "configuration_sources": sources,
        "effective_model_matrix": {
            "feature_set": (
                settings.feature_set
            ),
            "feature_count": len(
                settings.features
            ),
            "features": list(
                settings.features
            ),
            "scaler": settings.scaler,
            "output_dtype": (
                settings.output_dtype
            ),
            "feature_signature": (
                feature_signature(
                    settings.feature_set,
                    settings.features,
                )
            ),
        },
        "active_sequence_address": {
            "sequence_length": (
                address.sequence_length
            ),
            "target_horizon_minutes": (
                address.target_horizon
            ),
            "stride_minutes": (
                address.stride_minutes
            ),
            "scope": address.scope,
            "sessions": list(
                address.sessions
            ),
        },
    }

    if args.show_config:
        print(
            json.dumps(
                resolved,
                indent=2,
                default=str,
            )
        )
        return 0

    features_root = _require(
        args.features_root,
        "--features-root",
    )
    output_root = _require(
        args.output_root,
        "--output-root",
    )
    symbol = _require(
        args.symbol,
        "--symbol",
    )

    sequence_root = _require(
        args.sequence_root,
        "--sequence-root",
    )

    config = load_config(
        args.config
    )

    print(
        "Resolved model-matrix configuration:"
    )
    print(
        json.dumps(
            resolved,
            indent=2,
            default=str,
        )
    )

    summary = process_symbol(
        features_root=features_root,
        sequence_root=sequence_root,
        output_root=output_root,
        symbol=symbol,
        settings=settings,
        address=address,
        config=config,
        reports_root=args.reports,
        configuration_sources=sources,
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
