from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common import (
    CANONICAL_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    OHLCV,
    SOURCE_VALUE_COLUMNS,
    add_market_labels,
    effective_coverage_start,
    exchange_session_dates,
    expected_index_for_trading_date,
    load_config,
    load_instrument,
    load_ohlcv_csv,
    load_source,
    split_consecutive,
)


# IBKR WAP can carry half-cent precision while displayed OHLC values are
# rounded to cents. Preserve the source WAP exactly; only reject values that
# exceed the displayed OHLC range by more than this tolerance.
WAP_OHLC_TOLERANCE = 0.005
FLOAT_EPSILON = 1e-12


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_observed_input(df: pd.DataFrame) -> dict:
    required = [*OHLCV, *MICROSTRUCTURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing corrected source columns: {missing}")

    if df[required].isna().any().any():
        bad = df[required].isna().sum()
        raise ValueError(
            f"Observed source contains missing values: "
            f"{bad[bad > 0].to_dict()}"
        )

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume exists.")
    if (df["bar_count"] < 0).any():
        raise ValueError("Negative bar_count exists.")
    if (~df["bar_count"].eq(df["bar_count"].round())).any():
        raise ValueError("Non-integer bar_count exists.")

    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("Non-positive price exists.")

    if (
        df["high"] < df[["open", "low", "close"]].max(axis=1)
    ).any():
        raise ValueError("Invalid high value exists.")

    if (
        df["low"] > df[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Invalid low value exists.")

    # IBKR can return WAP with half-cent precision while displayed OHLC
    # values are rounded to cents. Preserve WAP exactly and measure only
    # the distance outside the displayed [low, high] interval.
    lower_excess = (df["low"] - df["wap"]).clip(lower=0.0)
    upper_excess = (df["wap"] - df["high"]).clip(lower=0.0)
    wap_excess = pd.concat(
        [lower_excess, upper_excess],
        axis=1,
    ).max(axis=1)

    wap_outside_displayed_range = wap_excess.gt(FLOAT_EPSILON)
    wap_over_tolerance = wap_excess.gt(
        WAP_OHLC_TOLERANCE + FLOAT_EPSILON
    )

    outside_count = int(wap_outside_displayed_range.sum())
    over_tolerance_count = int(wap_over_tolerance.sum())
    max_excess = (
        float(wap_excess.loc[wap_outside_displayed_range].max())
        if outside_count
        else 0.0
    )

    if over_tolerance_count:
        bad = df.loc[
            wap_over_tolerance,
            [
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "wap",
                "volume",
                "bar_count",
            ],
        ].copy()
        bad["wap_excess"] = wap_excess.loc[wap_over_tolerance]

        examples = (
            bad.sort_values("wap_excess", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

        raise ValueError(
            "WAP exceeds displayed OHLC tolerance: "
            f"rows={over_tolerance_count}, "
            f"max_excess={float(wap_excess.loc[wap_over_tolerance].max()):.12f}, "
            f"tolerance={WAP_OHLC_TOLERANCE:.3f}. "
            f"Examples: {examples}"
        )

    return {
        "wap_ohlc_tolerance": WAP_OHLC_TOLERANCE,
        "wap_outside_displayed_ohlc_rows": outside_count,
        "wap_over_tolerance_rows": over_tolerance_count,
        "wap_max_displayed_ohlc_excess": max_excess,
    }


def load_canonical_observed(
    *,
    input_path: Path,
    symbol: str,
    source_id: str,
    config,
) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(input_path, nrows=0)
    header.columns = header.columns.str.strip().str.lower()

    missing = sorted(set(CANONICAL_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(
            f"Input must be canonical observed data. Missing: {missing}"
        )

    df, metrics = load_ohlcv_csv(
        input_path,
        config.timezone,
    )

    if metrics.get("invalid_datetime_rows", 0):
        raise ValueError("Invalid datetime rows exist.")

    duplicate_rows = df.loc[
        df.duplicated("datetime", keep=False)
    ]

    for timestamp, group in duplicate_rows.groupby("datetime", sort=True):
        payload = [c for c in SOURCE_VALUE_COLUMNS if c in group.columns]
        if len(group[payload].drop_duplicates()) > 1:
            raise ValueError(
                f"Conflicting duplicate source timestamp: {timestamp}"
            )

    df = df.drop_duplicates(
        subset=["datetime", *SOURCE_VALUE_COLUMNS],
        keep="first",
    ).copy()

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps remain.")

    validation_metrics = validate_observed_input(df)
    metrics = {
        **metrics,
        **validation_metrics,
    }

    # Recompute labels with the shared calendar logic and verify the canonical
    # input agrees with it.
    stored_calendar = pd.to_datetime(
        df["calendar_date"],
        errors="coerce",
    ).dt.date
    stored_trading = pd.to_datetime(
        df["trading_date"],
        errors="coerce",
    ).dt.date
    stored_session = (
        df["session"].astype(str).str.strip().str.lower()
    )

    labeled = add_market_labels(
        df,
        config,
        source_id=source_id,
        symbol=symbol,
    )

    mismatch = (
        ~stored_calendar.eq(labeled["calendar_date"])
        | ~stored_trading.eq(labeled["trading_date"])
        | ~stored_session.eq(labeled["session"])
    )

    if mismatch.any():
        examples = (
            pd.DataFrame(
                {
                    "datetime": labeled.loc[mismatch, "datetime"],
                    "stored_session": stored_session.loc[mismatch],
                    "derived_session": labeled.loc[mismatch, "session"],
                    "stored_trading_date": stored_trading.loc[mismatch],
                    "derived_trading_date": labeled.loc[
                        mismatch, "trading_date"
                    ],
                }
            )
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Canonical labels disagree with common.py calendar logic. "
            f"Examples: {examples}"
        )

    observed = (
        df["is_observed"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    imputed = (
        df["is_imputed"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    if not observed.all():
        raise ValueError("Input contains non-observed rows.")
    if imputed.any():
        raise ValueError("Input already contains imputed rows.")

    labeled["is_observed"] = True
    labeled["is_imputed"] = False
    labeled["symbol"] = str(symbol).strip().lower()
    labeled["source_id"] = str(source_id).strip().lower()
    labeled["bar_count"] = labeled["bar_count"].round().astype("Int64")

    return labeled, metrics


def regularize_file(
    *,
    input_path: Path,
    output_path: Path,
    symbol: str,
    source_id: str,
    config_path: Path,
    instruments_path: Path,
    sources_path: Path,
    report_root: Path,
) -> dict:
    config = load_config(config_path)
    instrument = load_instrument(instruments_path, symbol)
    source = load_source(sources_path, source_id, symbol)

    coverage_start = effective_coverage_start(
        instrument,
        source,
        config.timezone,
    )
    covered_sessions = [
        str(x).strip().lower()
        for x in source["coverage_sessions"]
    ]

    df, load_metrics = load_canonical_observed(
        input_path=input_path,
        symbol=symbol,
        source_id=source_id,
        config=config,
    )

    before_start = df["datetime"] < coverage_start
    rows_before_start = int(before_start.sum())
    df = df.loc[~before_start].copy()

    outside_claim = ~df["session"].isin(covered_sessions)
    if outside_claim.any():
        examples = (
            df.loc[outside_claim, ["datetime", "session"]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Observed rows fall outside source coverage_sessions. "
            f"Examples: {examples}"
        )

    if df.empty:
        raise ValueError("No observed rows remain after coverage_start.")

    trading_dates = exchange_session_dates(
        config.calendar,
        coverage_start.date(),
        max(df["trading_date"]),
    )

    market_columns = [*OHLCV, *MICROSTRUCTURE_COLUMNS]
    frames = []
    audit_rows = []

    for exchange_date in trading_dates:
        trading_date = pd.Timestamp(exchange_date).date()

        for session_name in covered_sessions:
            expected = expected_index_for_trading_date(
                trading_date,
                config,
                [session_name],
                lower_bound=(
                    coverage_start
                    if trading_date == coverage_start.date()
                    else None
                ),
            )

            if len(expected) == 0:
                continue

            source_rows = df.loc[
                df["trading_date"].eq(trading_date)
                & df["session"].eq(session_name)
            ]

            result = (
                source_rows
                .set_index("datetime")[market_columns]
                .reindex(expected)
            )

            observed_mask = result["close"].notna()
            result["is_observed"] = observed_mask
            result["is_imputed"] = False
            result["quality_status"] = "observed"
            result["imputation_source_datetime"] = pd.NaT

            first = (
                result.index[observed_mask].min()
                if observed_mask.any()
                else pd.NaT
            )
            last = (
                result.index[observed_mask].max()
                if observed_mask.any()
                else pd.NaT
            )

            missing = result.index[~observed_mask]

            for window in split_consecutive(
                missing,
                config.frequency,
            ):
                if len(window) == 0:
                    continue

                if not observed_mask.any():
                    gap_type = "full_session"
                elif window[-1] < first:
                    gap_type = "leading"
                elif window[0] > last:
                    gap_type = "trailing"
                else:
                    gap_type = "internal"

                if gap_type == "internal":
                    should_fill = (
                        config.fill_internal_gaps
                        and len(window)
                        <= config.max_internal_gap_minutes
                    )
                elif gap_type == "leading":
                    should_fill = config.fill_leading_gaps
                elif gap_type == "trailing":
                    should_fill = config.fill_trailing_gaps
                else:
                    should_fill = config.fill_fully_missing_sessions

                source_timestamp = None
                source_close = None

                if should_fill:
                    previous_observed = result.loc[
                        (result.index < window[0])
                        & result["is_observed"]
                    ]

                    if previous_observed.empty:
                        should_fill = False
                    else:
                        source_timestamp = previous_observed.index[-1]
                        source_close = previous_observed.iloc[-1]["close"]

                if should_fill:
                    # Causal synthetic bar from the previous OBSERVED close.
                    result.loc[
                        window,
                        ["open", "high", "low", "close", "wap"],
                    ] = source_close
                    result.loc[window, "volume"] = config.synthetic_volume
                    result.loc[window, "bar_count"] = 0
                    result.loc[window, "is_imputed"] = True
                    result.loc[
                        window, "quality_status"
                    ] = f"imputed_{gap_type}"
                    result.loc[
                        window, "imputation_source_datetime"
                    ] = source_timestamp
                else:
                    if (
                        gap_type == "internal"
                        and len(window)
                        > config.max_internal_gap_minutes
                    ):
                        status = "missing_internal_too_large"
                    else:
                        status = f"missing_{gap_type}"

                    result.loc[window, "quality_status"] = status

                audit_rows.append(
                    {
                        "trading_date": trading_date,
                        "session": session_name,
                        "gap_type": gap_type,
                        "gap_start": window[0],
                        "gap_end": window[-1],
                        "gap_bars": int(len(window)),
                        "filled": bool(should_fill),
                        "source_datetime": source_timestamp,
                        "source_close": source_close,
                    }
                )

            result = result.reset_index().rename(
                columns={"index": "datetime"}
            )
            result["calendar_date"] = (
                result["datetime"]
                .dt.tz_convert(config.timezone)
                .dt.date
            )
            result["trading_date"] = trading_date
            result["session"] = session_name
            result["symbol"] = str(symbol).strip().lower()
            result["source_id"] = str(source_id).strip().lower()
            result["bar_count"] = (
                result["bar_count"].round().astype("Int64")
            )

            ordered = [
                *CANONICAL_COLUMNS,
                "quality_status",
                "imputation_source_datetime",
            ]
            frames.append(result[ordered])

    canonical = (
        pd.concat(frames, ignore_index=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if canonical.duplicated(
        ["datetime", "source_id"]
    ).any():
        raise ValueError(
            "Regularized output contains duplicate "
            "(datetime, source_id) rows."
        )

    observed_rows = int(canonical["is_observed"].sum())
    imputed_rows = int(canonical["is_imputed"].sum())
    unresolved_rows = int(
        (
            ~canonical["is_observed"]
            & ~canonical["is_imputed"]
        ).sum()
    )

    if observed_rows != len(df):
        raise ValueError(
            "Observed row conservation failed: "
            f"input={len(df)}, output_observed={observed_rows}"
        )

    atomic_write_csv(
        canonical,
        output_path,
    )

    run_id = datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    report_dir = (
        report_root
        / str(symbol).strip().lower()
        / str(source_id).strip().lower()
        / run_id
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        report_dir / "regularization_audit.csv",
        index=False,
    )

    (
        canonical.groupby("session")
        .agg(
            rows=("datetime", "size"),
            observed_rows=("is_observed", "sum"),
            imputed_rows=("is_imputed", "sum"),
            unresolved_rows=(
                "quality_status",
                lambda s: int(
                    s.astype(str)
                    .str.startswith("missing_")
                    .sum()
                ),
            ),
        )
        .reset_index()
        .to_csv(
            report_dir / "regularization_by_session.csv",
            index=False,
        )
    )

    summary = {
        "symbol": str(symbol).strip().lower(),
        "source_id": str(source_id).strip().lower(),
        "covered_sessions": covered_sessions,
        "effective_coverage_start": coverage_start.isoformat(),
        "rows_before_effective_coverage_start": rows_before_start,
        "input_observed_rows": int(len(df)),
        "output_grid_rows": int(len(canonical)),
        "observed_rows": observed_rows,
        "imputed_rows": imputed_rows,
        "unresolved_missing_rows": unresolved_rows,
        "quality_status_counts": (
            canonical["quality_status"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "gap_type_counts": (
            audit["gap_type"]
            .value_counts()
            .sort_index()
            .to_dict()
            if not audit.empty
            else {}
        ),
        "output_file": str(output_path.resolve()),
        "report_directory": str(report_dir.resolve()),
        **load_metrics,
    }

    (
        report_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a causal, calendar-aware 1-minute regularized grid "
            "from canonical observed data."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--source-id",
        default="primary_extended",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
    )
    parser.add_argument(
        "--instruments",
        type=Path,
        default=Path("config/instruments.csv"),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("config/sources.csv"),
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports/regularization"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = regularize_file(
        input_path=args.input,
        output_path=args.output,
        symbol=args.symbol,
        source_id=args.source_id,
        config_path=args.config,
        instruments_path=args.instruments,
        sources_path=args.sources,
        report_root=args.reports,
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