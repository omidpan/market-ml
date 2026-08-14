from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="EDA for monthly market_ml feature Parquet partitions."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Feature part.parquet files to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/feature_eda/amat"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Maximum usable rows sampled for correlation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def month_label(df: pd.DataFrame, path: Path) -> str:
    dt = pd.to_datetime(df["datetime"], errors="raise")
    values = dt.dt.strftime("%Y-%m").dropna().unique().tolist()
    if len(values) == 1:
        return values[0]
    return f"{path.stem}_{dt.min()}_{dt.max()}"


def finite_summary(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    records = []
    for col in feature_columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        values = numeric.to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(values)
        finite_values = values[finite]

        record = {
            "feature": col,
            "rows": len(values),
            "nan_count": int(np.isnan(values).sum()),
            "positive_inf_count": int(np.isposinf(values).sum()),
            "negative_inf_count": int(np.isneginf(values).sum()),
            "finite_count": int(finite.sum()),
            "finite_percent": float(finite.mean() * 100.0),
        }

        if finite_values.size:
            record.update(
                {
                    "min": float(np.min(finite_values)),
                    "p01": float(np.quantile(finite_values, 0.01)),
                    "p25": float(np.quantile(finite_values, 0.25)),
                    "median": float(np.quantile(finite_values, 0.50)),
                    "p75": float(np.quantile(finite_values, 0.75)),
                    "p99": float(np.quantile(finite_values, 0.99)),
                    "max": float(np.max(finite_values)),
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                }
            )
        else:
            record.update(
                {
                    "min": np.nan,
                    "p01": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )
        records.append(record)

    return pd.DataFrame(records)


def correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = (
        corr.where(upper)
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return pairs.sort_values(
        ["abs_correlation", "feature_a", "feature_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def heatmap(corr: pd.DataFrame, path: Path, title: str):
    if corr.empty:
        return
    n = len(corr)
    size = max(8, min(22, 0.42 * n))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_recent_day(df: pd.DataFrame, output_dir: Path):
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="raise")
    usable = work[work["is_current_bar_usable"].astype(bool)].copy()
    regular = usable[usable["session"].astype(str).str.lower().eq("regular")].copy()

    if regular.empty:
        return None

    last_day = regular["trading_date"].iloc[-1]
    day = regular[regular["trading_date"].eq(last_day)].sort_values("datetime").copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(day["datetime"], day["close"])
    ax.set_title(f"AMAT regular-session close — {last_day}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    fig.tight_layout()
    price_path = output_dir / "recent_regular_day_close.png"
    fig.savefig(price_path, dpi=180)
    plt.close(fig)

    candidates = [
        "f_1m_true_range_bps",
        "f_1m_session_price_vs_ema20_bps",
        "f_1m_session_relative_volume_20",
        "f_5m_log_return_1",
        "f_15m_log_return_1",
        "f_30m_log_return_1",
        "f_1h_log_return_1",
    ]

    chosen = [c for c in candidates if c in day.columns]
    for col in chosen:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(day["datetime"], pd.to_numeric(day[col], errors="coerce"))
        ax.set_title(f"{col} — {last_day}")
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(output_dir / f"recent_day__{col}.png", dpi=180)
        plt.close(fig)

    return str(last_day)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    per_file = []
    schemas = {}

    for path in args.files:
        path = path.expanduser().resolve()
        df = pd.read_parquet(path, engine="pyarrow")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        label = month_label(df, path)
        df["_eda_month"] = label

        feature_columns = sorted(c for c in df.columns if c.startswith("f_"))
        schemas[label] = feature_columns

        per_file.append(
            {
                "file": str(path),
                "month": label,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "feature_columns": int(len(feature_columns)),
                "first_datetime": str(df["datetime"].min()),
                "last_datetime": str(df["datetime"].max()),
                "usable_rows": int(df["is_current_bar_usable"].astype(bool).sum()),
                "observed_rows": int(df["is_observed"].astype(bool).sum()),
                "imputed_rows": int(df["is_imputed"].astype(bool).sum()),
                "sessions": {
                    str(k): int(v)
                    for k, v in df["session"].value_counts(dropna=False).to_dict().items()
                },
            }
        )
        frames.append(df)

    common_features = sorted(set.intersection(*(set(v) for v in schemas.values())))
    schema_equal = all(v == common_features for v in schemas.values())

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["datetime", "source_id"]).reset_index(drop=True)

    usable = combined[combined["is_current_bar_usable"].astype(bool)].copy()

    if len(usable) > args.sample_size:
        sample = usable.sample(
            n=args.sample_size,
            random_state=args.random_seed,
        ).sort_values("datetime")
    else:
        sample = usable.copy()

    X = sample[common_features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Spearman is useful for monotonic redundancy and less sensitive to extreme
    # market observations than Pearson.
    corr = X.corr(method="spearman", min_periods=max(100, min(5_000, len(X) // 10)))
    pairs = correlation_pairs(corr)

    pairs.to_csv(output_dir / "correlation_pairs_spearman.csv", index=False)
    pairs.head(200).to_csv(output_dir / "top_200_correlations.csv", index=False)

    all_stats = finite_summary(usable, common_features)
    all_stats.to_csv(output_dir / "feature_numeric_summary_usable.csv", index=False)

    missing_by_month = []
    for label, group in combined.groupby("_eda_month", sort=True):
        g = group[group["is_current_bar_usable"].astype(bool)]
        missing = g[common_features].isna().mean().mul(100.0)
        for feature, percent in missing.items():
            missing_by_month.append(
                {"month": label, "feature": feature, "nan_percent_usable": float(percent)}
            )
    pd.DataFrame(missing_by_month).to_csv(
        output_dir / "feature_missingness_by_month.csv",
        index=False,
    )

    # Family heatmaps: avoid one unreadable 133x133 annotated plot.
    family_prefixes = {
        "1m": "f_1m_",
        "5m": "f_5m_",
        "15m": "f_15m_",
        "30m": "f_30m_",
        "1h": "f_1h_",
        "session_time_quality": None,
    }

    for family, prefix in family_prefixes.items():
        if prefix is None:
            columns = [
                c for c in common_features
                if c.startswith(("f_session_", "f_time_", "f_quality_"))
            ]
        else:
            columns = [c for c in common_features if c.startswith(prefix)]

        if len(columns) >= 2:
            heatmap(
                corr.loc[columns, columns],
                output_dir / f"heatmap_spearman_{family}.png",
                f"AMAT Spearman feature correlation — {family}",
            )

    newest = max(frames, key=lambda x: x["datetime"].max())
    recent_day = plot_recent_day(newest, output_dir)

    # Compact visual overview of feature missingness.
    missing = (
        usable[common_features]
        .isna()
        .mean()
        .mul(100.0)
        .sort_values(ascending=False)
    )
    top_missing = missing.head(30)

    fig, ax = plt.subplots(figsize=(11, 9))
    positions = np.arange(len(top_missing))
    ax.barh(positions, top_missing.values)
    ax.set_yticks(positions)
    ax.set_yticklabels(top_missing.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("NaN percent of usable rows")
    ax.set_title("AMAT features with most missing values")
    fig.tight_layout()
    fig.savefig(output_dir / "top_feature_missingness.png", dpi=180)
    plt.close(fig)

    summary = {
        "files": per_file,
        "schema_equal_across_files": bool(schema_equal),
        "common_feature_count": int(len(common_features)),
        "combined_rows": int(len(combined)),
        "combined_usable_rows": int(len(usable)),
        "correlation_sample_rows": int(len(sample)),
        "features_with_positive_or_negative_infinity": int(
            (
                (all_stats["positive_inf_count"] > 0)
                | (all_stats["negative_inf_count"] > 0)
            ).sum()
        ),
        "features_with_any_nan_on_usable_rows": int(
            usable[common_features].isna().any().sum()
        ),
        "top_20_absolute_spearman_correlations": (
            pairs.head(20).to_dict(orient="records")
        ),
        "recent_regular_day_plotted": recent_day,
    }

    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nEDA files written to: {output_dir}")


if __name__ == "__main__":
    main()
