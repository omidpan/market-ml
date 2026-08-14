from __future__ import annotations

import argparse
import os
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd

from common import load_config, session_windows_for_trading_date


REQUIRED_COLUMNS = [
    "datetime", "trading_date", "session",
    "open", "high", "low", "close",
    "volume", "wap", "bar_count",
    "symbol", "source_id",
    "is_observed", "is_imputed",
    "quality_status",
]


def normalize_timeframe(value: str) -> tuple[str, timedelta]:
    """
    Parse an integer-minute/hour timeframe without NumPy generic timedelta
    units. This avoids the NumPy 2.5 generic-unit deprecation.
    """
    text = str(value).strip().lower()

    aliases = {
        "60m": "1h",
        "60min": "1h",
        "1hr": "1h",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
    }
    text = aliases.get(text, text)

    match = re.fullmatch(r"(\d+)(m|h)", text)
    if match is None:
        raise ValueError(
            f"Invalid timeframe {value!r}. "
            "Use integer minutes/hours such as 5m, 15m, 30m, 1h."
        )

    amount = int(match.group(1))
    unit = match.group(2)

    if amount <= 0:
        raise ValueError("Timeframe must be greater than zero.")

    minutes = amount if unit == "m" else amount * 60
    delta = timedelta(minutes=minutes)

    label = "1h" if minutes == 60 else f"{minutes}m"
    return label, delta


def load_regularized_parquet(root: Path, symbol: str) -> pd.DataFrame:
    symbol = str(symbol).strip().lower()
    files = sorted((root / f"symbol={symbol}").rglob("part.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found for {symbol} under {root}")

    df = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in files],
        ignore_index=True,
    )
    df.columns = df.columns.str.strip().str.lower()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Regularized input is missing columns: {missing}")

    if not isinstance(df["datetime"].dtype, pd.DatetimeTZDtype):
        raise ValueError("datetime must be timezone-aware in regularized Parquet.")

    df["trading_date"] = pd.to_datetime(
        df["trading_date"], errors="coerce"
    ).dt.date

    if df["trading_date"].isna().any():
        raise ValueError("Invalid trading_date exists.")

    if df.duplicated(["datetime", "source_id"]).any():
        raise ValueError("Duplicate (datetime, source_id) rows exist.")

    impossible = df["is_observed"] & df["is_imputed"]
    if impossible.any():
        raise ValueError("A minute cannot be both observed and imputed.")

    return df.sort_values(["datetime", "source_id"]).reset_index(drop=True)


def first_valid(series: pd.Series):
    s = series.dropna()
    return float("nan") if s.empty else s.iloc[0]


def last_valid(series: pd.Series):
    s = series.dropna()
    return float("nan") if s.empty else s.iloc[-1]


def aggregate_wap(group: pd.DataFrame):
    market = group.loc[group["close"].notna()]
    if market.empty:
        return float("nan")

    positive = market.loc[market["volume"].fillna(0) > 0]
    if not positive.empty:
        denom = float(positive["volume"].sum())
        if denom > 0:
            return float((positive["wap"] * positive["volume"]).sum()) / denom

    return last_valid(market["wap"])


def resample_one_timeframe(
    df: pd.DataFrame,
    *,
    timeframe: str,
    config,
) -> pd.DataFrame:
    label, delta = normalize_timeframe(timeframe)
    one_minute = timedelta(minutes=1)
    rows = []

    grouped = df.groupby(
        ["trading_date", "session", "source_id"],
        sort=True,
        dropna=False,
    )

    for (trading_date, session_name, source_id), session_df in grouped:
        windows, _ = session_windows_for_trading_date(
            trading_date,
            config,
        )
        if session_name not in windows:
            raise ValueError(
                f"No session window for {trading_date} {session_name!r}"
            )

        session_start, session_end = windows[session_name]
        local_times = session_df["datetime"].dt.tz_convert(config.timezone)

        if ((local_times < session_start) | (local_times >= session_end)).any():
            raise ValueError(
                f"Minute exists outside session window: "
                f"{trading_date} {session_name}"
            )

        bin_number = ((local_times - session_start) // delta).astype("int64")

        work = session_df.copy()
        work["_bin"] = bin_number.to_numpy()

        for bin_id, group in work.groupby("_bin", sort=True):
            bar_start = session_start + int(bin_id) * delta
            bar_end = min(bar_start + delta, session_end)

            expected_minutes = int((bar_end - bar_start) / one_minute)
            source_grid_minutes = int(len(group))
            observed_minutes = int(group["is_observed"].sum())
            imputed_minutes = int(group["is_imputed"].sum())
            market_minutes = observed_minutes + imputed_minutes

            missing_minutes = max(expected_minutes - market_minutes, 0)
            is_complete = (
                source_grid_minutes == expected_minutes
                and missing_minutes == 0
            )
            has_imputation = imputed_minutes > 0

            market = group.loc[group["close"].notna()]

            if market.empty:
                o = h = l = c = v = float("nan")
                count = pd.NA
            else:
                o = first_valid(market["open"])
                h = market["high"].max()
                l = market["low"].min()
                c = last_valid(market["close"])
                v = market["volume"].sum(min_count=1)
                count = market["bar_count"].sum(min_count=1)

            if not is_complete:
                quality = "incomplete"
            elif has_imputation:
                quality = "complete_with_imputation"
            else:
                quality = "complete_observed"

            symbols = group["symbol"].dropna().astype(str).str.lower().unique()
            if len(symbols) != 1:
                raise ValueError("Derived bar contains multiple symbols.")

            rows.append(
                {
                    "datetime": bar_start,
                    "bar_end": bar_end,
                    "available_at": bar_end,
                    "trading_date": trading_date,
                    "calendar_date": bar_start.tz_convert(config.timezone).date(),
                    "session": session_name,
                    "timeframe": label,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "wap": aggregate_wap(group),
                    "bar_count": count,
                    "symbol": symbols[0],
                    "source_id": str(source_id).strip().lower(),
                    "expected_minutes": expected_minutes,
                    "source_grid_minutes": source_grid_minutes,
                    "observed_minutes": observed_minutes,
                    "imputed_minutes": imputed_minutes,
                    "missing_minutes": int(missing_minutes),
                    "is_complete": bool(is_complete),
                    "has_imputation": bool(has_imputation),
                    "quality_status": quality,
                }
            )

    result = pd.DataFrame(rows)
    result["bar_count"] = pd.to_numeric(
        result["bar_count"], errors="coerce"
    ).round().astype("Int64")

    if result.duplicated(
        ["datetime", "timeframe", "source_id"]
    ).any():
        raise ValueError("Duplicate derived bar keys exist.")

    if (result["available_at"] <= result["datetime"]).any():
        raise ValueError("available_at must be after bar start.")

    return result.sort_values(
        ["datetime", "source_id"]
    ).reset_index(drop=True)


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        df.to_parquet(
            tmp,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_monthly(
    df: pd.DataFrame,
    *,
    output_root: Path,
    symbol: str,
    timeframe: str,
    timezone: str,
    compression: str,
) -> list[Path]:
    label, _ = normalize_timeframe(timeframe)
    symbol = str(symbol).strip().lower()

    local = df["datetime"].dt.tz_convert(timezone)
    work = df.copy()
    work["_year"] = local.dt.year
    work["_month"] = local.dt.month

    written = []

    for (year, month), group in work.groupby(
        ["_year", "_month"],
        sort=True,
    ):
        path = (
            output_root
            / f"timeframe={label}"
            / f"symbol={symbol}"
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
            / "part.parquet"
        )

        payload = (
            group.drop(columns=["_year", "_month"])
            .sort_values(["datetime", "source_id"])
            .reset_index(drop=True)
        )

        atomic_write_parquet(
            payload,
            path,
            compression=compression,
        )
        written.append(path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive causal higher-timeframe bars from "
            "regularized canonical 1-minute Parquet."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "15m", "30m", "1h"],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    source = load_regularized_parquet(
        args.input_root,
        args.symbol,
    )

    print(
        f"Loaded {len(source):,} regularized 1-minute rows "
        f"for {str(args.symbol).strip().lower()}"
    )

    for timeframe in args.timeframes:
        derived = resample_one_timeframe(
            source,
            timeframe=timeframe,
            config=config,
        )

        files = write_monthly(
            derived,
            output_root=args.output_root,
            symbol=args.symbol,
            timeframe=timeframe,
            timezone=config.timezone,
            compression=config.parquet_compression,
        )

        print(
            f"[{timeframe}] "
            f"bars={len(derived):,}, "
            f"complete={int(derived['is_complete'].sum()):,}, "
            f"incomplete={int((~derived['is_complete']).sum()):,}, "
            f"with_imputation={int(derived['has_imputation'].sum()):,}, "
            f"files={len(files)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())