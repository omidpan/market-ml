"""
label_diagnostics.py

Read-only diagnostics for market_ml label-policy research.

Commands
--------
atr-scan
    Compute causal Wilder ATR on features_1m, join it to model-ready sequence
    endpoints, and scan ATR-relative label multipliers on a selected split.

granularity
    Diagnose exact-zero target returns and return/price granularity by year
    and session.

all
    Run both diagnostics.

Safety
------
- Existing Parquet artifacts are READ ONLY.
- This tool never creates or modifies Parquet files.
- Optional CSV/JSON reports are written only under reports/label_thresholds/.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

VERSION = "1.0.0"


@dataclass(frozen=True)
class Paths:
    project_root: Path
    feature_root: Path
    sequence_root: Path
    report_root: Path


@dataclass(frozen=True)
class Config:
    symbol: str
    sequence_length: int
    horizon: int
    scope: str
    stride: int
    sessions: tuple[str, ...]
    feature_set: str
    split: str
    atr_period: int


def parse_csv_floats(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("At least one multiplier is required.")
    return vals


def parse_csv_strings(value: str) -> tuple[str, ...]:
    vals = tuple(x.strip() for x in value.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("At least one session is required.")
    return vals


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        symbol=args.symbol.strip().lower(),
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        scope=args.scope,
        stride=args.stride,
        sessions=tuple(args.sessions),
        feature_set=args.feature_set,
        split=args.split.lower(),
        atr_period=args.atr_period,
    )


def make_paths(args: argparse.Namespace, cfg: Config) -> Paths:
    root = Path(args.project_root).expanduser().resolve()

    feature_root = root / "data" / "parquet" / "features_1m"

    sequence_root = (
        root
        / "data"
        / "parquet"
        / "model_matrix"
        / f"sequence_length={cfg.sequence_length}"
        / f"horizon={cfg.horizon}m"
        / f"scope={cfg.scope}"
        / f"stride={cfg.stride}"
        / f"sessions={'+'.join(cfg.sessions)}"
        / f"feature_set={cfg.feature_set}"
        / "sequences"
    )

    if args.report_root:
        report_root = Path(args.report_root).expanduser()
        if not report_root.is_absolute():
            report_root = root / report_root
        report_root = report_root.resolve()
    else:
        report_root = (
            root
            / "reports"
            / "label_thresholds"
            / f"{cfg.symbol}_atr{cfg.atr_period}_h{cfg.horizon}_{cfg.feature_set}"
        )

    return Paths(root, feature_root, sequence_root, report_root)


def symbol_parquet_files(root: Path, symbol: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    needle = f"symbol={symbol.lower()}"
    files = sorted(
        p for p in root.rglob("*.parquet")
        if needle in p.as_posix().lower()
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files for symbol={symbol!r} under {root}"
        )

    return files


def load_columns(
    files: Sequence[Path],
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    frames = []
    total = len(files)

    for i, path in enumerate(files, 1):
        frames.append(pd.read_parquet(path, columns=list(columns)))
        if i % 20 == 0 or i == total:
            print(f"  {label}: loaded {i:>3}/{total} files")

    return pd.concat(frames, ignore_index=True)


def as_utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise").astype("int64")


def load_features(paths: Paths, cfg: Config) -> pd.DataFrame:
    print("\n[features] Loading features_1m OHLC...")

    cols = [
        "prediction_time",
        "source_id",
        "high",
        "low",
        "close",
        "is_current_bar_usable",
        "session",
        "trading_date",
    ]

    df = load_columns(
        symbol_parquet_files(paths.feature_root, cfg.symbol),
        cols,
        label="features",
    )

    df["_ts"] = as_utc_ns(df["prediction_time"])
    df["source_id"] = df["source_id"].astype(str)

    for c in ["high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_current_bar_usable"] = (
        df["is_current_bar_usable"].fillna(False).astype(bool)
    )

    df = (
        df.sort_values(["source_id", "_ts"], kind="stable")
        .reset_index(drop=True)
    )

    dup = df.duplicated(["source_id", "_ts"])
    if dup.any():
        raise RuntimeError(
            f"Duplicate feature timestamps: {int(dup.sum()):,}"
        )

    print(f"Feature rows: {len(df):,}")
    return df


def load_sequences(paths: Paths, cfg: Config) -> pd.DataFrame:
    print("\n[sequences] Loading model-ready sequence index...")

    cols = [
        "prediction_time",
        "source_id",
        "split",
        "session",
        "trading_date",
        "target_log_return",
        "target_return_bps",
    ]

    df = load_columns(
        symbol_parquet_files(paths.sequence_root, cfg.symbol),
        cols,
        label="sequences",
    )

    df["source_id"] = df["source_id"].astype(str)
    df["_ts"] = as_utc_ns(df["prediction_time"])
    df["target_log_return"] = pd.to_numeric(
        df["target_log_return"], errors="coerce"
    )
    df["target_return_bps"] = pd.to_numeric(
        df["target_return_bps"], errors="coerce"
    )

    df = df.loc[
        df["split"].astype(str).str.lower().eq(cfg.split)
        & np.isfinite(df["target_log_return"])
        & np.isfinite(df["target_return_bps"])
    ].copy()

    print(f"Model-ready {cfg.split.upper()} sequences: {len(df):,}")
    return df


def compute_wilder_atr(
    features: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    """
    Causal Wilder ATR over exact physical 1-minute continuity blocks.

    Continuity rules:
    - same source_id
    - exactly 60 seconds from previous row
    - current and previous rows usable

    At a continuity break, previous close is not imported across the gap.
    The first TR of a block is high-low.
    """
    print(f"\n[atr] Calculating causal Wilder ATR{period}...")

    high = features["high"].to_numpy(np.float64)
    low = features["low"].to_numpy(np.float64)
    close = features["close"].to_numpy(np.float64)
    ts = features["_ts"].to_numpy(np.int64)
    source = features["source_id"].to_numpy()

    usable = (
        features["is_current_bar_usable"].to_numpy()
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (high > 0)
        & (low > 0)
        & (close > 0)
    )

    n = len(features)

    same_source = np.zeros(n, dtype=bool)
    same_source[1:] = source[1:] == source[:-1]

    one_minute = np.zeros(n, dtype=bool)
    one_minute[1:] = (ts[1:] - ts[:-1]) == 60_000_000_000

    prev_usable = np.zeros(n, dtype=bool)
    prev_usable[1:] = usable[:-1]

    continuous = same_source & one_minute & usable & prev_usable
    block_start = ~continuous

    tr = high - low
    idx = np.flatnonzero(continuous)

    if len(idx):
        prev_close = close[idx - 1]
        tr[idx] = np.maximum.reduce(
            [
                high[idx] - low[idx],
                np.abs(high[idx] - prev_close),
                np.abs(low[idx] - prev_close),
            ]
        )

    tr[~usable] = np.nan

    atr = np.full(n, np.nan, dtype=np.float64)

    starts = np.flatnonzero(block_start)
    ends = np.r_[starts[1:], n]
    block_count = 0

    for start, end in zip(starts, ends):
        if not usable[start]:
            continue

        if end - start < period:
            continue

        vals = tr[start:end]

        if not np.all(np.isfinite(vals)):
            raise RuntimeError(
                f"Non-finite TR inside usable block [{start}:{end}]"
            )

        seed_idx = start + period - 1
        atr[seed_idx] = np.mean(vals[:period])

        for pos in range(seed_idx + 1, end):
            atr[pos] = (
                ((period - 1) * atr[pos - 1]) + tr[pos]
            ) / period

        block_count += 1

    atr_relative = np.divide(
        atr,
        close,
        out=np.full_like(atr, np.nan),
        where=np.isfinite(atr) & np.isfinite(close) & (close > 0),
    )

    out = features.copy()
    out[f"atr{period}_1m"] = atr
    out[f"atr{period}_relative"] = atr_relative
    out[f"atr{period}_bps"] = atr_relative * 10_000.0

    print(f"Usable OHLC rows: {int(usable.sum()):,}")
    print(f"Rows with ATR{period}: {int(np.isfinite(atr).sum()):,}")
    print(f"Continuous ATR blocks: {block_count:,}")

    return out


def join_atr(
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    rel = f"atr{period}_relative"
    bps = f"atr{period}_bps"
    val = f"atr{period}_1m"

    print("\n[join] Joining causal ATR to sequence prediction times...")

    atr_map = features.loc[
        np.isfinite(features[rel]),
        ["source_id", "_ts", "close", val, rel, bps],
    ].copy()

    joined = sequences.merge(
        atr_map,
        on=["source_id", "_ts"],
        how="left",
        validate="many_to_one",
    )

    missing = int(joined[rel].isna().sum())
    print(f"Selected rows missing ATR{period}: {missing:,}")

    if missing:
        print(
            joined.loc[
                joined[rel].isna(),
                ["prediction_time", "source_id", "session", "trading_date"],
            ]
            .head(10)
            .to_string(index=False)
        )
        raise RuntimeError(
            "Missing ATR on model-ready rows; investigate before selecting "
            "thresholds."
        )

    joined["year"] = (
        pd.to_datetime(joined["trading_date"], errors="raise").dt.year
    )

    return joined


def scan_atr(
    joined: pd.DataFrame,
    period: int,
    multipliers: Sequence[float],
) -> dict[str, pd.DataFrame]:
    print("\n[scan] Scanning ATR multipliers...")

    r = joined["target_log_return"].to_numpy(np.float64)
    alpha = joined[f"atr{period}_relative"].to_numpy(np.float64)

    agg_rows = []
    session_rows = []
    year_rows = []
    sy_rows = []

    for m in multipliers:
        tau = m * alpha

        label = np.zeros(len(r), dtype=np.int8)
        label[r < -tau] = -1
        label[r > tau] = 1

        tbps = tau * 10_000.0

        down = int(np.sum(label == -1))
        neutral = int(np.sum(label == 0))
        up = int(np.sum(label == 1))
        total = len(label)
        non_neutral = down + up

        agg_rows.append(
            {
                "m": float(m),
                "samples": total,
                "threshold_bps_p25": float(np.quantile(tbps, 0.25)),
                "threshold_bps_p50": float(np.quantile(tbps, 0.50)),
                "threshold_bps_p75": float(np.quantile(tbps, 0.75)),
                "down_count": down,
                "neutral_count": neutral,
                "up_count": up,
                "down_pct": 100.0 * down / total,
                "neutral_pct": 100.0 * neutral / total,
                "up_pct": 100.0 * up / total,
                "up_pct_among_non_neutral": (
                    100.0 * up / non_neutral if non_neutral else np.nan
                ),
            }
        )

        tmp = pd.DataFrame(
            {
                "session": joined["session"].astype(str).to_numpy(),
                "year": joined["year"].to_numpy(),
                "label": label,
            }
        )

        for session, g in tmp.groupby("session", sort=True):
            y = g["label"].to_numpy()
            session_rows.append(
                {
                    "m": float(m),
                    "session": session,
                    "samples": len(y),
                    "down_pct": 100.0 * np.mean(y == -1),
                    "neutral_pct": 100.0 * np.mean(y == 0),
                    "up_pct": 100.0 * np.mean(y == 1),
                }
            )

        for year, g in tmp.groupby("year", sort=True):
            y = g["label"].to_numpy()
            year_rows.append(
                {
                    "m": float(m),
                    "year": int(year),
                    "samples": len(y),
                    "down_pct": 100.0 * np.mean(y == -1),
                    "neutral_pct": 100.0 * np.mean(y == 0),
                    "up_pct": 100.0 * np.mean(y == 1),
                }
            )

        for (year, session), g in tmp.groupby(
            ["year", "session"], sort=True
        ):
            y = g["label"].to_numpy()
            sy_rows.append(
                {
                    "m": float(m),
                    "year": int(year),
                    "session": session,
                    "samples": len(y),
                    "down_pct": 100.0 * np.mean(y == -1),
                    "neutral_pct": 100.0 * np.mean(y == 0),
                    "up_pct": 100.0 * np.mean(y == 1),
                }
            )

    aggregate = pd.DataFrame(agg_rows)
    by_session = pd.DataFrame(session_rows)
    by_year = pd.DataFrame(year_rows)
    by_session_year = pd.DataFrame(sy_rows)

    stability_rows = []

    for m in multipliers:
        s = by_session.loc[by_session["m"].eq(m)]
        y = by_year.loc[by_year["m"].eq(m)]
        sy = by_session_year.loc[
            by_session_year["m"].eq(m)
            & by_session_year["samples"].ge(1000)
        ]

        stability_rows.append(
            {
                "m": float(m),
                "session_neutral_min_pct": float(s["neutral_pct"].min()),
                "session_neutral_max_pct": float(s["neutral_pct"].max()),
                "year_neutral_min_pct": float(y["neutral_pct"].min()),
                "year_neutral_max_pct": float(y["neutral_pct"].max()),
                "session_year_neutral_min_pct": (
                    float(sy["neutral_pct"].min()) if not sy.empty else np.nan
                ),
                "session_year_neutral_max_pct": (
                    float(sy["neutral_pct"].max()) if not sy.empty else np.nan
                ),
                "session_year_min_any_class_pct": (
                    float(
                        sy[["down_pct", "neutral_pct", "up_pct"]]
                        .min(axis=1)
                        .min()
                    )
                    if not sy.empty
                    else np.nan
                ),
            }
        )

    return {
        "aggregate": aggregate,
        "by_session": by_session,
        "by_year": by_year,
        "by_session_year": by_session_year,
        "stability": pd.DataFrame(stability_rows),
    }


def print_atr_results(
    joined: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    period: int,
) -> None:
    exact_zero = int(
        np.sum(
            joined["target_log_return"].to_numpy(np.float64) == 0.0
        )
    )

    print(
        f"\nExact-zero selected-split target_log_return rows: "
        f"{exact_zero:,}"
    )

    atr_bps = joined[f"atr{period}_bps"].to_numpy(np.float64)

    print(f"\nATR{period} relative volatility (before multiplying by m):")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(
            f"  p{int(q*100):02d}: "
            f"{np.quantile(atr_bps, q):8.3f} bps"
        )

    print("\n" + "=" * 60)
    print("ATR MULTIPLIER SCAN")
    print("=" * 60)

    cols = [
        "m",
        "threshold_bps_p25",
        "threshold_bps_p50",
        "threshold_bps_p75",
        "down_pct",
        "neutral_pct",
        "up_pct",
        "up_pct_among_non_neutral",
    ]

    print(
        tables["aggregate"][cols].to_string(
            index=False,
            float_format=lambda x: f"{x:8.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("REGIME-STABILITY SUMMARY")
    print("=" * 60)

    print(
        tables["stability"].to_string(
            index=False,
            float_format=lambda x: f"{x:8.3f}",
        )
    )


def granularity_input(
    features: pd.DataFrame,
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    close_map = features[
        ["source_id", "_ts", "close"]
    ].drop_duplicates(["source_id", "_ts"])

    joined = sequences.merge(
        close_map,
        on=["source_id", "_ts"],
        how="left",
        validate="many_to_one",
    )

    missing = int(joined["close"].isna().sum())
    if missing:
        raise RuntimeError(f"Missing close rows after join: {missing:,}")

    joined["year"] = (
        pd.to_datetime(joined["trading_date"], errors="raise").dt.year
    )

    return joined


def summarize_granularity(group: pd.DataFrame) -> pd.Series:
    r = group["target_return_bps"].to_numpy(np.float64)
    close = group["close"].to_numpy(np.float64)

    zero = r == 0.0
    nz = np.abs(r[~zero])
    median_close = float(np.median(close))

    def q(v: np.ndarray, quantile: float) -> float:
        if len(v) == 0:
            return np.nan
        return float(np.quantile(v, quantile))

    return pd.Series(
        {
            "samples": int(len(group)),
            "median_close": median_close,
            "exact_zero_pct": 100.0 * float(zero.mean()),
            "nonzero_abs_min_bps": (
                float(nz.min()) if len(nz) else np.nan
            ),
            "nonzero_abs_p10_bps": q(nz, 0.10),
            "nonzero_abs_p25_bps": q(nz, 0.25),
            "nonzero_abs_p50_bps": q(nz, 0.50),
            "nonzero_abs_p75_bps": q(nz, 0.75),
            "nonzero_abs_p90_bps": q(nz, 0.90),
            "one_cent_bps_at_median_close": (
                0.01 / median_close * 10_000.0
            ),
        }
    )


def make_granularity_tables(
    joined: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    value_cols = ["target_return_bps", "close"]

    by_year = (
        joined.groupby("year", sort=True)[value_cols]
        .apply(summarize_granularity)
        .reset_index()
    )

    by_year_session = (
        joined.groupby(["year", "session"], sort=True)[value_cols]
        .apply(summarize_granularity)
        .reset_index()
    )

    return {
        "by_year": by_year,
        "by_year_session": by_year_session,
    }


def print_granularity_results(
    tables: dict[str, pd.DataFrame],
) -> None:
    print("\n" + "=" * 60)
    print("RETURN GRANULARITY BY YEAR")
    print("=" * 60)

    print(
        tables["by_year"].to_string(
            index=False,
            float_format=lambda x: f"{x:10.3f}",
        )
    )

    first_years = sorted(
        tables["by_year"]["year"].astype(int).unique()
    )[:3]

    print("\n" + "=" * 60)
    print(f"EARLIEST YEARS BY SESSION: {first_years}")
    print("=" * 60)

    print(
        tables["by_year_session"]
        .loc[tables["by_year_session"]["year"].isin(first_years)]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:10.3f}",
        )
    )


def write_reports(
    paths: Paths,
    cfg: Config,
    joined_atr: pd.DataFrame | None = None,
    atr_tables: dict[str, pd.DataFrame] | None = None,
    multipliers: Sequence[float] | None = None,
    gran_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    paths.report_root.mkdir(parents=True, exist_ok=True)

    if atr_tables is not None and joined_atr is not None:
        atr_tables["aggregate"].to_csv(
            paths.report_root / "atr_multiplier_aggregate.csv",
            index=False,
        )
        atr_tables["by_session"].to_csv(
            paths.report_root / "atr_multiplier_by_session.csv",
            index=False,
        )
        atr_tables["by_year"].to_csv(
            paths.report_root / "atr_multiplier_by_year.csv",
            index=False,
        )
        atr_tables["by_session_year"].to_csv(
            paths.report_root / "atr_multiplier_by_session_year.csv",
            index=False,
        )
        atr_tables["stability"].to_csv(
            paths.report_root / "atr_multiplier_stability.csv",
            index=False,
        )

        metadata = {
            "tool": "label_diagnostics.py",
            "version": VERSION,
            "symbol": cfg.symbol,
            "target_horizon_minutes": cfg.horizon,
            "feature_set": cfg.feature_set,
            "sequence_length": cfg.sequence_length,
            "scope": cfg.scope,
            "stride_minutes": cfg.stride,
            "sessions": list(cfg.sessions),
            "split_used": cfg.split,
            "atr": {
                "timeframe": "1min",
                "period": cfg.atr_period,
                "method": "Wilder",
                "seed": (
                    f"mean of first {cfg.atr_period} TR values "
                    "inside each exact-continuity block"
                ),
                "continuity_rule": (
                    "reset whenever exact physical 1-minute continuity breaks"
                ),
                "first_tr_after_break": (
                    "high-low; previous close is not imported across the gap"
                ),
                "causal": True,
            },
            "label": {
                "return": (
                    f"target_log_return at exact H={cfg.horizon}m"
                ),
                "alpha": f"ATR{cfg.atr_period}_t / close_t",
                "tau": "m * alpha_t",
                "down": "target_log_return < -tau",
                "neutral": "abs(target_log_return) <= tau",
                "up": "target_log_return > tau",
            },
            "candidate_multipliers": (
                [float(x) for x in multipliers]
                if multipliers is not None
                else None
            ),
            "selected_split_samples": int(len(joined_atr)),
            "exact_zero_target_rows": int(
                np.sum(
                    joined_atr["target_log_return"]
                    .to_numpy(np.float64)
                    == 0.0
                )
            ),
            "missing_atr_rows": int(
                joined_atr[
                    f"atr{cfg.atr_period}_relative"
                ].isna().sum()
            ),
            "parquet_modified": False,
        }

        with open(
            paths.report_root / "atr_multiplier_scan_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

    if gran_tables is not None:
        gran_tables["by_year"].to_csv(
            paths.report_root / "return_granularity_by_year.csv",
            index=False,
        )
        gran_tables["by_year_session"].to_csv(
            paths.report_root / "return_granularity_by_year_session.csv",
            index=False,
        )

    print(f"\nReports written to: {paths.report_root}")


def run_atr_scan(
    paths: Paths,
    cfg: Config,
    multipliers: Sequence[float],
    write: bool,
) -> None:
    features = compute_wilder_atr(
        load_features(paths, cfg),
        cfg.atr_period,
    )
    sequences = load_sequences(paths, cfg)
    joined = join_atr(features, sequences, cfg.atr_period)
    tables = scan_atr(joined, cfg.atr_period, multipliers)
    print_atr_results(joined, tables, cfg.atr_period)

    if write:
        write_reports(
            paths,
            cfg,
            joined_atr=joined,
            atr_tables=tables,
            multipliers=multipliers,
        )


def run_granularity(
    paths: Paths,
    cfg: Config,
    write: bool,
) -> None:
    features = load_features(paths, cfg)
    sequences = load_sequences(paths, cfg)
    joined = granularity_input(features, sequences)
    tables = make_granularity_tables(joined)
    print_granularity_results(tables)

    if write:
        write_reports(paths, cfg, gran_tables=tables)


def run_all(
    paths: Paths,
    cfg: Config,
    multipliers: Sequence[float],
    write: bool,
) -> None:
    # Load once.
    features = load_features(paths, cfg)
    sequences = load_sequences(paths, cfg)

    # ATR scan.
    features_atr = compute_wilder_atr(features, cfg.atr_period)
    joined_atr = join_atr(features_atr, sequences, cfg.atr_period)
    atr_tables = scan_atr(
        joined_atr,
        cfg.atr_period,
        multipliers,
    )
    print_atr_results(
        joined_atr,
        atr_tables,
        cfg.atr_period,
    )

    # Granularity.
    joined_gran = granularity_input(features, sequences)
    gran_tables = make_granularity_tables(joined_gran)
    print_granularity_results(gran_tables)

    if write:
        write_reports(
            paths,
            cfg,
            joined_atr=joined_atr,
            atr_tables=atr_tables,
            multipliers=multipliers,
            gran_tables=gran_tables,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read-only label-policy diagnostics for market_ml."
        )
    )

    p.add_argument(
        "command",
        choices=["atr-scan", "granularity", "all"],
    )
    p.add_argument("--project-root", default=".")
    p.add_argument("--symbol", required=True)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--sequence-length", type=int, default=120)
    p.add_argument("--feature-set", default="core_v1")
    p.add_argument("--scope", default="continuous")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--sessions",
        type=parse_csv_strings,
        default=("premarket", "regular", "aftermarket"),
    )
    p.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
    )
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument(
        "--multipliers",
        type=parse_csv_floats,
        default=[
            0.0,
            0.25,
            0.50,
            0.75,
            1.00,
            1.25,
            1.50,
            2.00,
            2.50,
            3.00,
            4.00,
        ],
    )
    p.add_argument("--report-root", default=None)
    p.add_argument(
        "--no-write-reports",
        action="store_true",
        help="Print diagnostics only; do not write CSV/JSON reports.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    return p


def validate_args(args: argparse.Namespace) -> None:
    for name in ["horizon", "sequence_length", "stride", "atr_period"]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")

    if any(
        (not math.isfinite(x)) or x < 0
        for x in args.multipliers
    ):
        raise SystemExit(
            "--multipliers must be finite and non-negative."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    cfg = make_config(args)
    paths = make_paths(args, cfg)
    write = not args.no_write_reports

    print(f"label_diagnostics.py version {VERSION}")
    print(f"Command: {args.command}")
    print(f"Project root: {paths.project_root}")
    print(f"Symbol: {cfg.symbol}")
    print(f"Split: {cfg.split}")
    print(f"Feature root: {paths.feature_root}")
    print(f"Sequence root: {paths.sequence_root}")
    print("Parquet impact: READ-ONLY")
    print(
        "Report impact: "
        + (
            f"WRITE {paths.report_root}"
            if write
            else "NONE"
        )
    )

    if args.command == "atr-scan":
        run_atr_scan(
            paths,
            cfg,
            args.multipliers,
            write,
        )
    elif args.command == "granularity":
        run_granularity(paths, cfg, write)
    else:
        run_all(
            paths,
            cfg,
            args.multipliers,
            write,
        )

    print("\nDone.")
    print("No Parquet files were modified.")


if __name__ == "__main__":
    main()
