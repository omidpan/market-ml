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
    add_market_labels,
    load_config,
    load_ohlcv_csv,
    load_source,
)


def atomic_write_csv(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )

    try:
        frame.to_csv(
            temp_path,
            index=False,
        )
        os.replace(
            temp_path,
            output_path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def canonicalize_observed(
    *,
    input_path: Path,
    output_path: Path,
    symbol: str,
    source_id: str,
    config_path: Path,
    sources_path: Path,
    report_root: Path,
) -> dict:
    """
    Convert validated source CSV rows into canonical observed rows.

    This function does NOT regularize or fill anything.
    Every output row is an actual source row:
        is_observed = True
        is_imputed  = False
    """
    config = load_config(
        config_path
    )
    source = load_source(
        sources_path,
        source_id,
        symbol,
    )

    df, load_metrics = load_ohlcv_csv(
        input_path,
        config.timezone,
    )

    if load_metrics.get(
        "invalid_datetime_rows",
        0,
    ):
        raise ValueError(
            "Invalid datetime rows exist. "
            "Canonicalization stopped."
        )

    # The current canonical schema requires the corrected IBKR
    # microstructure pair.
    missing_microstructure = [
        column
        for column in MICROSTRUCTURE_COLUMNS
        if column not in df.columns
    ]
    if missing_microstructure:
        raise ValueError(
            "Current canonical schema requires wap and bar_count. "
            f"Missing: {missing_microstructure}"
        )

    numeric_columns = [
        *OHLCV,
        *MICROSTRUCTURE_COLUMNS,
    ]

    invalid_numeric = df[
        numeric_columns
    ].isna()

    if invalid_numeric.any().any():
        bad = invalid_numeric.sum()
        bad = bad[
            bad > 0
        ].to_dict()
        raise ValueError(
            "Invalid source numeric values exist: "
            f"{bad}"
        )

    if df["datetime"].duplicated().any():
        examples = (
            df.loc[
                df["datetime"].duplicated(
                    keep=False
                ),
                "datetime",
            ]
            .head(5)
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Duplicate source timestamps exist. "
            f"Examples: {examples}"
        )

    labeled = add_market_labels(
        df,
        config,
        source_id=source_id,
        symbol=symbol,
    )

    covered_sessions = [
        str(value).strip().lower()
        for value in source[
            "coverage_sessions"
        ]
    ]

    unknown = labeled[
        "session"
    ].eq("unknown")

    if unknown.any():
        examples = (
            labeled.loc[
                unknown,
                "datetime",
            ]
            .head(10)
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Observed rows fall outside all configured "
            "calendar-aware sessions. "
            f"Examples: {examples}"
        )

    outside_claim = ~labeled[
        "session"
    ].isin(
        covered_sessions
    )

    if outside_claim.any():
        examples = (
            labeled.loc[
                outside_claim,
                [
                    "datetime",
                    "session",
                ],
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )
        raise ValueError(
            "Observed rows fall outside this source's "
            "claimed sessions. "
            f"Examples: {examples}"
        )

    labeled[
        "is_observed"
    ] = True
    labeled[
        "is_imputed"
    ] = False

    # Canonical storage identity is lower-case. IBKR request identity
    # remains upper-case inside the scraper.
    labeled["symbol"] = (
        str(symbol)
        .strip()
        .lower()
    )
    labeled["source_id"] = (
        str(source_id)
        .strip()
        .lower()
    )
    labeled["session"] = (
        labeled["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # bar_count is semantically integral after validation.
    non_integer_count = ~labeled[
        "bar_count"
    ].eq(
        labeled[
            "bar_count"
        ].round()
    )

    if non_integer_count.any():
        raise ValueError(
            "Non-integer bar_count exists."
        )

    labeled["bar_count"] = (
        labeled["bar_count"]
        .round()
        .astype("int64")
    )

    canonical = (
        labeled[
            CANONICAL_COLUMNS
        ]
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    atomic_write_csv(
        canonical,
        output_path,
    )

    run_id = datetime.now(
        UTC
    ).strftime(
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

    summary = {
        "symbol": (
            str(symbol)
            .strip()
            .lower()
        ),
        "source_id": (
            str(source_id)
            .strip()
            .lower()
        ),
        "input_file": str(
            input_path.resolve()
        ),
        "output_file": str(
            output_path.resolve()
        ),
        "input_rows": int(
            len(df)
        ),
        "output_rows": int(
            len(canonical)
        ),
        "observed_rows": int(
            canonical[
                "is_observed"
            ].sum()
        ),
        "imputed_rows": int(
            canonical[
                "is_imputed"
            ].sum()
        ),
        "first_datetime": (
            canonical[
                "datetime"
            ].min().isoformat()
            if not canonical.empty
            else None
        ),
        "last_datetime": (
            canonical[
                "datetime"
            ].max().isoformat()
            if not canonical.empty
            else None
        ),
        "sessions": (
            canonical[
                "session"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "zero_volume_observed_rows": int(
            (
                canonical[
                    "is_observed"
                ]
                & canonical[
                    "volume"
                ].eq(0)
            ).sum()
        ),
        "report_directory": str(
            report_dir.resolve()
        ),
    }

    (
        report_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create canonical observed rows from "
            "a validated source CSV."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--source-id",
        default="primary_extended",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/pipeline.yaml"
        ),
    )

    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(
            "config/sources.csv"
        ),
    )

    parser.add_argument(
        "--reports",
        type=Path,
        default=Path(
            "reports/canonicalization"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = canonicalize_observed(
        input_path=args.input,
        output_path=args.output,
        symbol=args.symbol,
        source_id=args.source_id,
        config_path=args.config,
        sources_path=args.sources,
        report_root=args.reports,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())