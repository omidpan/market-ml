import os
from pathlib import Path
import pandas as pd
base=Path(__file__).resolve().parent.parent
root = base / "data/parquet/features_1m/symbol=amat"
path = root / "year=2020/month=07/part.parquet"
df = pd.read_parquet(path)

print("Shape:", df.shape)
print()
print(df.dtypes)
print()

feature_columns = [
    c for c in df.columns
    if c.startswith("f_")
]

print("Feature count:", len(feature_columns))

for column in feature_columns:
    print(column)
print(
    df[
        [
            "datetime",
            "session",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "wap",
            "is_current_bar_usable",
            "contiguous_history_minutes",
        ]
    ].tail(30).to_string()
)


files = [
    root / "year=2019/month=12/part.parquet",
    root / "year=2023/month=06/part.parquet",
    root / "year=2026/month=07/part.parquet",
]

frames = [
    pd.read_parquet(path)
    for path in files
]
#### compare old and recent
sample_df = pd.concat(
    frames,
    ignore_index=True,
)
print(sample_df.shape)

print(
    sample_df.groupby(
        [sample_df["datetime"].dt.year, "session"]
    ).size()
)

print("============ missing\n")
feature_columns = [
    c for c in sample_df.columns
    if c.startswith("f_")
]

usable = sample_df[
    sample_df["is_current_bar_usable"]
].copy()

missing = (
    usable[feature_columns]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    missing.head(100).to_string()
)


import numpy as np

X = (
    usable[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
)

# A sample is plenty for exploratory correlation.
X_sample = X.sample(
    n=min(50_000, len(X)),
    random_state=42,
)

corr = X_sample.corr(
    method="spearman",
    min_periods=5_000,
)