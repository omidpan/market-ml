from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config


TARGET_VERSION = "1.0.0"
DEFAULT_HORIZONS = (5, 15, 30)

REQUIRED_COLUMNS = [
    "datetime",
    "prediction_time",
    "feature_available_at",
    "calendar_date",
    "trading_date",
    "session",
    "symbol",
    "source_id",
    "close",
    "is_current_bar_usable",
    "contiguous_history_minutes",
]


def normalize_symbol(value: str) -> str:
    return str(value).strip().lower()


def ensure_tz(series: pd.Series, timezone: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(timezone)
    if pd.api.types.is_datetime64_dtype(series.dtype):
        return series.dt.tz_localize(
            timezone,
            ambiguous="raise",
            nonexistent="raise",
        )
    return pd.to_datetime(
        series,
        errors="raise",
        utc=True,
    ).dt.tz_convert(timezone)


def parse_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)

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
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )
    if parsed.isna().any():
        raise ValueError(
            f"Invalid boolean values in {name}: "
            f"{series.loc[parsed.isna()].head(5).tolist()}"
        )
    return parsed.astype(bool)


def normalize_features(
    df: pd.DataFrame,
    symbol: str,
    timezone: str,
) -> pd.DataFrame:
    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Feature input missing columns: {missing}"
        )

    out = df.copy()

    for c in (
        "datetime",
        "prediction_time",
        "feature_available_at",
    ):
        out[c] = ensure_tz(
            out[c],
            timezone,
        )

    for c in (
        "calendar_date",
        "trading_date",
    ):
        out[c] = pd.to_datetime(
            out[c],
            errors="raise",
        ).dt.date

    out["close"] = pd.to_numeric(
        out["close"],
        errors="coerce",
    )
    out["contiguous_history_minutes"] = (
        pd.to_numeric(
            out["contiguous_history_minutes"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int32")
    )
    out["is_current_bar_usable"] = parse_bool(
        out["is_current_bar_usable"],
        "is_current_bar_usable",
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

    expected = normalize_symbol(symbol)
    found = out["symbol"].dropna().unique()
    if len(found) != 1 or found[0] != expected:
        raise ValueError(
            f"Expected symbol={expected!r}; "
            f"found {found.tolist()}"
        )

    if out.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            "Duplicate feature keys found."
        )

    if (
        out["feature_available_at"]
        > out["prediction_time"]
    ).any():
        raise ValueError(
            "Feature leakage detected: "
            "feature_available_at > prediction_time."
        )

    expected_prediction = (
        out["datetime"]
        + timedelta(minutes=1)
    )
    if (
        out["prediction_time"]
        != expected_prediction
    ).any():
        raise ValueError(
            "Expected prediction_time = datetime + 1 minute."
        )

    return (
        out.sort_values(
            ["source_id", "datetime"]
        )
        .reset_index(drop=True)
    )


def discover_parts(
    root: Path,
    symbol: str,
) -> dict[tuple[int, int], Path]:
    base = (
        root
        / f"symbol={normalize_symbol(symbol)}"
    )
    if not base.exists():
        raise FileNotFoundError(base)

    parts = {}
    for path in sorted(
        base.glob(
            "year=*/month=*/part.parquet"
        )
    ):
        year = int(
            path.parent.parent.name
            .split("=", 1)[1]
        )
        month = int(
            path.parent.name
            .split("=", 1)[1]
        )
        key = (year, month)
        if key in parts:
            raise ValueError(
                f"Duplicate partition {key}"
            )
        parts[key] = path

    if not parts:
        raise FileNotFoundError(
            f"No Parquet partitions under {base}"
        )
    return parts


def load_month_with_future_context(
    current_path: Path,
    next_path: Path | None,
    symbol: str,
    timezone: str,
    max_horizon: int,
) -> pd.DataFrame:
    current = normalize_features(
        pd.read_parquet(
            current_path,
            engine="pyarrow",
        ),
        symbol,
        timezone,
    )
    current["_is_current_month"] = True
    frames = [current]

    if next_path is not None:
        nxt = normalize_features(
            pd.read_parquet(
                next_path,
                engine="pyarrow",
            ),
            symbol,
            timezone,
        )
        nxt = (
            nxt.groupby(
                "source_id",
                group_keys=False,
                sort=False,
            )
            .head(max_horizon + 2)
            .reset_index(drop=True)
        )
        nxt["_is_current_month"] = False
        frames.append(nxt)

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


def direction_labels(
    return_bps: pd.Series,
    valid: pd.Series,
    threshold_bps: float,
) -> pd.Series:
    labels = pd.Series(
        pd.NA,
        index=return_bps.index,
        dtype="Int8",
    )
    down = (
        valid
        & return_bps.lt(-threshold_bps)
    )
    up = (
        valid
        & return_bps.gt(threshold_bps)
    )
    neutral = (
        valid
        & ~down
        & ~up
    )

    labels.loc[down] = -1
    labels.loc[neutral] = 0
    labels.loc[up] = 1
    return labels


def add_target(
    df: pd.DataFrame,
    horizon: int,
    threshold_bps: float,
    require_same_session: bool,
) -> None:
    """
    Exact target timing.

    For a feature row:
        datetime=t
        prediction_time=t+1

    an h-minute target uses the close realized exactly at:
        prediction_time + h

    which is the close of the bar:
        datetime=t+h

    The next available row is never substituted.
    """
    h = int(horizon)
    suffix = f"{h}m"

    grouped = df.groupby(
        "source_id",
        sort=False,
        dropna=False,
    )

    future_dt = (
        grouped["datetime"]
        .shift(-h)
    )
    future_close = (
        grouped["close"]
        .shift(-h)
    )
    future_usable = (
        grouped["is_current_bar_usable"]
        .shift(-h)
        .eq(True)
    )
    future_contig = (
        grouped[
            "contiguous_history_minutes"
        ]
        .shift(-h)
    )
    future_session = (
        grouped["session"]
        .shift(-h)
    )
    future_date = (
        grouped["trading_date"]
        .shift(-h)
    )

    exact_future = (
        future_dt.notna()
        & future_dt.eq(
            df["datetime"]
            + timedelta(minutes=h)
        )
    )

    current_close_ok = (
        df["close"].notna()
        & df["close"].gt(0)
    )
    future_close_ok = (
        future_close.notna()
        & future_close.gt(0)
    )

    # h steps into the future means h+1 usable
    # 1-minute rows from current through endpoint.
    continuous = (
        pd.to_numeric(
            future_contig,
            errors="coerce",
        )
        .ge(h + 1)
    )

    same_session = (
        exact_future
        & future_session.eq(
            df["session"]
        )
    )
    same_date = (
        exact_future
        & pd.Series(
            future_date,
            index=df.index,
        ).eq(
            df["trading_date"]
        )
    )

    valid = (
        df["is_current_bar_usable"]
        & exact_future
        & future_usable
        & continuous
        & current_close_ok
        & future_close_ok
    )

    if require_same_session:
        valid = (
            valid
            & same_session
        )

    realized_at = (
        df["prediction_time"]
        + timedelta(minutes=h)
    )

    simple_return = (
        future_close
        / df["close"]
        - 1.0
    ).where(valid)

    log_return = np.log(
        future_close
        / df["close"]
    ).where(valid)

    return_bps = (
        simple_return
        * 10_000.0
    )

    df[
        f"target_valid_{suffix}"
    ] = valid

    df[
        f"target_realized_at_{suffix}"
    ] = realized_at

    df[
        f"target_close_{suffix}"
    ] = future_close.where(valid)

    df[
        f"target_simple_return_{suffix}"
    ] = simple_return

    df[
        f"target_log_return_{suffix}"
    ] = log_return

    df[
        f"target_return_bps_{suffix}"
    ] = return_bps

    df[
        f"target_direction_{suffix}"
    ] = direction_labels(
        return_bps,
        valid,
        threshold_bps,
    )

    df[
        f"target_same_session_{suffix}"
    ] = same_session.astype(bool)

    df[
        f"target_same_trading_date_{suffix}"
    ] = same_date.astype(bool)

    reason = pd.Series(
        "valid",
        index=df.index,
        dtype="object",
    )

    reason.loc[
        ~df["is_current_bar_usable"]
    ] = "current_unusable"

    mask = (
        reason.eq("valid")
        & ~exact_future
    )
    reason.loc[
        mask
    ] = "future_not_exact_horizon"

    mask = (
        reason.eq("valid")
        & ~future_usable
    )
    reason.loc[
        mask
    ] = "future_unusable"

    mask = (
        reason.eq("valid")
        & ~continuous
    )
    reason.loc[
        mask
    ] = "continuity_break"

    mask = (
        reason.eq("valid")
        & ~current_close_ok
    )
    reason.loc[
        mask
    ] = "invalid_current_close"

    mask = (
        reason.eq("valid")
        & ~future_close_ok
    )
    reason.loc[
        mask
    ] = "invalid_future_close"

    if require_same_session:
        mask = (
            reason.eq("valid")
            & ~same_session
        )
        reason.loc[
            mask
        ] = "crosses_session"

    if (
        valid
        != reason.eq("valid")
    ).any():
        raise AssertionError(
            f"{suffix}: validity/reason mismatch"
        )

    df[
        f"target_invalid_reason_{suffix}"
    ] = reason


def build_targets(
    df: pd.DataFrame,
    horizons: list[int],
    threshold_bps: float,
    require_same_session: bool,
) -> pd.DataFrame:
    out = df.copy()
    for horizon in horizons:
        add_target(
            out,
            horizon,
            threshold_bps,
            require_same_session,
        )
    return out


def output_columns(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    columns = [
        "datetime",
        "prediction_time",
        "calendar_date",
        "trading_date",
        "session",
        "symbol",
        "source_id",
        "close",
        "is_current_bar_usable",
        "contiguous_history_minutes",
    ]

    for h in horizons:
        s = f"{h}m"
        columns += [
            f"target_valid_{s}",
            f"target_realized_at_{s}",
            f"target_close_{s}",
            f"target_simple_return_{s}",
            f"target_log_return_{s}",
            f"target_return_bps_{s}",
            f"target_direction_{s}",
            f"target_same_session_{s}",
            f"target_same_trading_date_{s}",
            f"target_invalid_reason_{s}",
        ]

    return df[columns].copy()


def atomic_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        df.to_parquet(
            tmp,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        os.replace(
            tmp,
            path,
        )
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        tmp.write_text(
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
            tmp,
            path,
        )
    finally:
        if tmp.exists():
            tmp.unlink()


def summarize(
    df: pd.DataFrame,
    horizon: int,
) -> dict:
    s = f"{horizon}m"
    valid = df[
        f"target_valid_{s}"
    ]
    direction = df.loc[
        valid,
        f"target_direction_{s}",
    ]
    reasons = (
        df.loc[
            ~valid,
            f"target_invalid_reason_{s}",
        ]
        .value_counts()
        .to_dict()
    )
    returns = df.loc[
        valid,
        f"target_return_bps_{s}",
    ]

    return {
        "valid_rows": int(
            valid.sum()
        ),
        "invalid_rows": int(
            (~valid).sum()
        ),
        "valid_percent": round(
            100.0
            * float(valid.mean()),
            6,
        ),
        "crosses_session_valid_rows": int(
            (
                valid
                & ~df[
                    f"target_same_session_{s}"
                ]
            ).sum()
        ),
        "crosses_trading_date_valid_rows": int(
            (
                valid
                & ~df[
                    f"target_same_trading_date_{s}"
                ]
            ).sum()
        ),
        "direction_counts": {
            "down": int(
                direction.eq(-1).sum()
            ),
            "neutral": int(
                direction.eq(0).sum()
            ),
            "up": int(
                direction.eq(1).sum()
            ),
        },
        "invalid_reason_counts": {
            str(k): int(v)
            for k, v in reasons.items()
        },
        "return_bps_summary": {
            "mean": float(
                returns.mean()
            ) if len(returns) else None,
            "std": float(
                returns.std(ddof=0)
            ) if len(returns) else None,
            "median": float(
                returns.median()
            ) if len(returns) else None,
            "p01": float(
                returns.quantile(0.01)
            ) if len(returns) else None,
            "p99": float(
                returns.quantile(0.99)
            ) if len(returns) else None,
        },
    }


def process_symbol(
    features_root: Path,
    output_root: Path,
    symbol: str,
    horizons: list[int],
    config,
    reports_root: Path | None,
    threshold_bps: float,
    require_same_session: bool,
) -> dict:
    symbol = normalize_symbol(symbol)
    horizons = sorted(
        set(
            int(h)
            for h in horizons
        )
    )

    if (
        not horizons
        or horizons[0] <= 0
    ):
        raise ValueError(
            "Horizons must be positive."
        )

    if threshold_bps < 0:
        raise ValueError(
            "direction threshold must be >= 0"
        )

    parts = discover_parts(
        features_root,
        symbol,
    )
    keys = sorted(parts)

    totals = {
        h: {
            "valid_rows": 0,
            "invalid_rows": 0,
            "crosses_session_valid_rows": 0,
            "crosses_trading_date_valid_rows": 0,
            "direction_counts": {
                "down": 0,
                "neutral": 0,
                "up": 0,
            },
            "invalid_reason_counts": {},
        }
        for h in horizons
    }

    months = []
    total_rows = 0

    for i, key in enumerate(keys):
        year, month = key
        next_path = (
            parts[
                keys[i + 1]
            ]
            if i + 1 < len(keys)
            else None
        )

        data = (
            load_month_with_future_context(
                parts[key],
                next_path,
                symbol,
                config.timezone,
                max(horizons),
            )
        )

        data = build_targets(
            data,
            horizons,
            threshold_bps,
            require_same_session,
        )

        current = (
            data.loc[
                data["_is_current_month"]
            ]
            .copy()
            .reset_index(drop=True)
        )

        out = output_columns(
            current,
            horizons,
        )

        if out.duplicated(
            ["datetime", "source_id"]
        ).any():
            raise ValueError(
                f"{year}-{month:02d}: duplicate keys"
            )

        path = (
            output_root
            / f"symbol={symbol}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / "part.parquet"
        )

        atomic_parquet(
            out,
            path,
            config.parquet_compression,
        )

        month_h = {
            f"{h}m": summarize(
                out,
                h,
            )
            for h in horizons
        }

        for h in horizons:
            m = month_h[
                f"{h}m"
            ]
            t = totals[h]
            for key_name in (
                "valid_rows",
                "invalid_rows",
                "crosses_session_valid_rows",
                "crosses_trading_date_valid_rows",
            ):
                t[key_name] += m[
                    key_name
                ]

            for label in (
                "down",
                "neutral",
                "up",
            ):
                t[
                    "direction_counts"
                ][label] += (
                    m[
                        "direction_counts"
                    ][label]
                )

            for reason, count in (
                m[
                    "invalid_reason_counts"
                ].items()
            ):
                t[
                    "invalid_reason_counts"
                ][reason] = (
                    t[
                        "invalid_reason_counts"
                    ].get(
                        reason,
                        0,
                    )
                    + count
                )

        months.append(
            {
                "year": year,
                "month": month,
                "rows": int(
                    len(out)
                ),
                "output_file": str(
                    path
                ),
                "horizons": month_h,
            }
        )

        total_rows += len(out)

        status = [
            f"[{year:04d}-{month:02d}]",
            f"rows={len(out):,}",
        ]
        for h in horizons:
            status.append(
                f"{h}m_valid="
                f"{month_h[f'{h}m']['valid_rows']:,}"
            )
        print(
            ", ".join(status)
        )

    horizon_summary = {}
    for h in horizons:
        t = totals[h]
        n = (
            t["valid_rows"]
            + t["invalid_rows"]
        )
        horizon_summary[
            f"{h}m"
        ] = {
            **t,
            "valid_percent": round(
                100.0
                * t["valid_rows"]
                / n,
                6,
            ) if n else None,
        }

    summary = {
        "target_version": TARGET_VERSION,
        "symbol": symbol,
        "rows": int(
            total_rows
        ),
        "months": len(
            keys
        ),
        "horizons_minutes": horizons,
        "direction_threshold_bps": (
            threshold_bps
        ),
        "require_same_session": (
            require_same_session
        ),
        "timing_definition": (
            "At prediction_time=P, the h-minute "
            "target uses close(P+h)/close(P)-1. "
            "The endpoint must exist at exactly P+h."
        ),
        "continuity_definition": (
            "Every one-minute row from the "
            "current bar through the target "
            "endpoint must be usable and contiguous."
        ),
        "direction_encoding": {
            "-1": "down",
            "0": "neutral",
            "1": "up",
        },
        "horizons": (
            horizon_summary
        ),
        "output_root": str(
            output_root
        ),
        "month_summaries": months,
    }

    if reports_root is not None:
        report = (
            reports_root
            / symbol
            / (
                "target_summary_"
                f"v{TARGET_VERSION}.json"
            )
        )
        atomic_json(
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
    p = argparse.ArgumentParser(
        description=(
            "Build exact-horizon future-return "
            "targets from the causal 1-minute "
            "feature store."
        )
    )

    p.add_argument(
        "--features-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--symbol",
        required=True,
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_HORIZONS
        ),
    )
    p.add_argument(
        "--direction-threshold-bps",
        type=float,
        default=0.0,
        help=(
            "Neutral band for -1/0/+1 labels. "
            "0 means only exactly-zero future "
            "return is neutral."
        ),
    )
    p.add_argument(
        "--require-same-session",
        action="store_true",
        help=(
            "Reject exact targets that cross "
            "premarket/regular/aftermarket boundaries."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/pipeline.yaml"
        ),
    )
    p.add_argument(
        "--reports",
        type=Path,
        default=None,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(
        args.config
    )

    summary = process_symbol(
        features_root=(
            args.features_root
        ),
        output_root=(
            args.output_root
        ),
        symbol=args.symbol,
        horizons=args.horizons,
        config=config,
        reports_root=(
            args.reports
        ),
        threshold_bps=(
            args.direction_threshold_bps
        ),
        require_same_session=(
            args.require_same_session
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
    raise SystemExit(main())
