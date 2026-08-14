import argparse
import json
import os
import platform
from pathlib import Path

# TensorFlow's optimized graph path can abort inside its C++ runtime on some
# Intel macOS/Conda installations. This must be configured before TensorFlow
# is imported. It has no effect on Apple Silicon, Linux, or Windows.
IS_INTEL_MAC = (
    platform.system() == "Darwin"
    and platform.machine().lower() in {"x86_64", "amd64"}
)
if IS_INTEL_MAC:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

### mlflow dependencies
import requests
from dotenv import load_dotenv
############
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers, Model   
import tensorflow as tf
from utils.utils import set_seeds

############# configs

PROJECT_DIRECTORY = Path(__file__).resolve().parent


def resolve_from_project_directory(path):
    """Resolve relative paths from this Python file, not the terminal."""

    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_DIRECTORY / path).resolve()

load_dotenv(
    PROJECT_DIRECTORY / ".env"
)
from config import (
    DATA_DIR,
    FEATURE_META_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SIZE,
)


# ============================================================
# FIXED INPUT FEATURES FOR THIS EXPERIMENT
# ============================================================
# These are the only columns passed to the LSTM as X.
'''
 Classification — final 8 features
 CLASSIFICATION_FEATURES = [
    "BodyPct",
    "GapPct",
    "UpperShadowPct",
    "LowerShadowPct",
    "EMA_Ratio",
    "PriceEMA20",
    "Volatility5",
    "TrueRangePct",
]
Target class
CLASSIFICATION_TARGETS = [
    "FutureDirection",
]

Regression final features

REGRESSION_FEATURES = [
    "BodyPct",
    "GapPct",
    "UpperShadowPct",
    "LowerShadowPct",
    "EMA_Ratio",
    "PriceEMA20",
    "Volatility5",
    "TrueRangePct",
    "smh_RollingQuantile25",
    "spy_Volatility10",
]
Final Target for regression
REGRESSION_TARGETS = [
    "FutureLogReturn1",
]
'''
LESS_COMPLEX_MATRIX = [
    'BodyPct',
    'GapPct',
     'UpperShadowPct',
     'LowerShadowPct',
       'EMA_Ratio',
       'PriceEMA20',
       'Volatility5',
      'TrueRangePct',
      # 'smh_RollingQuantile25',
      'spy_Volatility10'
      #  "smh_RollingKurtosis",
      #  "RollingStd",
   
    # "EMA_Ratio",
    # "LowerShadowPct",
    # "RollingKurtosis",
    # "RollingSkew",
    # "RollingStd",
    # "UpperShadowPct",
    # Volatility
    # "Volatility5",
    # "VolumeChange",
    # "spy_EMA_Ratio",
    # "spy_LowerShadowPct",
    # "smh_LowerShadowPct",
    # "spy_RollingKurtosis",
    # "smh_RollingKurtosis",
    # "spy_RollingSkew",
    # "smh_RollingSkew",
    # "spy_UpperShadowPct",
    # "smh_UpperShadowPct",
    # "spy_VolumeChange",
    # "smh_VolumeChange",
    # "smh_GapDirection",
    # "spy_GapDirection",
]
# LESS_COMPLEX_MATRIX = [
#     'BodyPct',
#     'GapPct',
#      'UpperShadowPct',
#      'LowerShadowPct',
#        'EMA_Ratio',
#        'PriceEMA20',
#        'Volatility5',
#       'TrueRangePct',
#       # 'smh_RollingQuantile25',
#       'spy_Volatility10'
#       #  "smh_RollingKurtosis",
#       #  "RollingStd",
   
#     # "EMA_Ratio",
#     # "LowerShadowPct",
#     # "RollingKurtosis",
#     # "RollingSkew",
#     # "RollingStd",
#     # "UpperShadowPct",
#     # Volatility
#     # "Volatility5",
#     # "VolumeChange",
#     # "spy_EMA_Ratio",
#     # "spy_LowerShadowPct",
#     # "smh_LowerShadowPct",
#     # "spy_RollingKurtosis",
#     # "smh_RollingKurtosis",
#     # "spy_RollingSkew",
#     # "smh_RollingSkew",
#     # "spy_UpperShadowPct",
#     # "smh_UpperShadowPct",
#     # "spy_VolumeChange",
#     # "smh_VolumeChange",
#     # "smh_GapDirection",
#     # "spy_GapDirection",
# ]

# These target columns already exist in the combined CSV. LogReturn1 and
# Direction must stay here because they are the sources used to create the
# corresponding one-candle-ahead targets.
SOURCE_TARGET_COLUMNS = [
    "LogReturn1",
    "Direction",
    "LogReturn3",
    # "LogReturn5",
    # "LogReturn10",
    "spy_LogReturn1",
    "spy_Direction",
    # "spy_LogReturn3",
    "smh_LogReturn1",
    # "smh_Direction",
    # "smh_LogReturn3",
]

# Derived targets are created in this training script after the combined CSV
# has been loaded and validated. A negative shift means that row t receives the
# value from a later row of the same ticker.
FUTURE_TARGET_SPECS = {
    "FutureLogReturn1": {
        "source_column": "LogReturn1",
        "shift_periods": 1,
    },
    "FutureDirection": {
        "source_column": "Direction",
        "shift_periods": 1,
    },
}

# Every source or derived label that may be selected as y. None of these
# columns is included in LESS_COMPLEX_MATRIX, so none is passed to the LSTM as
# an input feature.
target_classes = [
    *SOURCE_TARGET_COLUMNS,
    *FUTURE_TARGET_SPECS,
]

CLASSIFICATION_TARGETS = [
    "Direction",
    "FutureDirection",
    "spy_Direction",
    "smh_Direction",
]

# Direction columns use three raw labels:
#   -1 = Sell, 0 = Hold, 1 = Buy
# SparseCategoricalCrossentropy requires zero-based class indices, so the
# classification branch encodes them as 0, 1, and 2 respectively.
DIRECTION_RAW_LABELS = (-1, 0, 1)
DIRECTION_LABEL_TO_INDEX = {
    raw_label: class_index
    for class_index, raw_label in enumerate(DIRECTION_RAW_LABELS)
}
DIRECTION_INDEX_TO_LABEL = {
    class_index: raw_label
    for raw_label, class_index in DIRECTION_LABEL_TO_INDEX.items()
}
NUMBER_OF_DIRECTION_CLASSES = len(DIRECTION_RAW_LABELS)

REGRESSION_TARGETS = [
    target_name
    for target_name in target_classes
    if target_name not in CLASSIFICATION_TARGETS
]

# ticker_id identifies the stock only for grouping and sequence boundaries.
# It is deliberately excluded from LESS_COMPLEX_MATRIX and therefore from X.
TICKER_COLUMN = "ticker_id"

MODEL_DATA_COLUMNS = [
    "datetime",
    TICKER_COLUMN,
    *LESS_COMPLEX_MATRIX,
    # Only source targets are required in the input CSV. Future targets are
    # generated later by create_selected_future_targets().
    *SOURCE_TARGET_COLUMNS,
]



parser = argparse.ArgumentParser(
    description="Train one LSTM using all stocks in the combined dataset."
)
parser.add_argument(
    "-s",
    "--symbol",
    type=str,
    required=False,
    default="all",
    help="Deprecated compatibility argument; the combined file contains all tickers.",
)
parser.add_argument(
    "-bs", "--bar_size", type=str, required=False, default="1 day", help="Candle size."
)
parser.add_argument(
    "--data-path",
    type=Path,
    default=None,
    help=(
        "Path to the combined CSV. Relative paths are resolved from this "
        "Python file. If omitted, DATA_DIR/combined_semiconductor_<bar>_feng.csv "
        "is used."
    ),
)
# REGRESSION CHANGE 1: choose the experiment without editing the source code.
parser.add_argument(
    "-t","--task",
    choices=["classification", "regression", "multi_regression"],
    default="regression",
    help="Model type to train (default: regression).",
)
parser.add_argument(
    "--regression-targets",
    default=None,
    help=(
        "Optional comma-separated regression targets from target_classes. "
        "Defaults to LogReturn1 for regression and all available continuous "
        "targets for multi_regression."
    ),
)
parser.add_argument(
    "--classification-target",
    choices=CLASSIFICATION_TARGETS,
    default="Direction",
    help=(
        "Three-class target (-1=Sell, 0=Hold, 1=Buy). Use "
        "FutureDirection to predict the next candle's direction."
    ),
)
parser.add_argument(
    "--window-size",
    type=int,
    default=WINDOW_SIZE,
    help=(
        "Number of consecutive feature rows in each LSTM sequence. "
        "Defaults to WINDOW_SIZE from config.py."
    ),
)
parser.add_argument(
    "--execution-mode",
    choices=["auto", "graph", "eager"],
    default="auto",
    help=(
        "TensorFlow execution mode. 'auto' uses eager mode on Intel macOS "
        "to avoid a native TensorFlow graph-runtime abort and graph mode "
        "elsewhere (default: auto)."
    ),
)
parser.add_argument(
    "-expr","--experiment-name",
    default=None,
    help=(
        "Optional MLflow experiment name. "
        "If omitted, a name is generated automatically."
    ),
)
args = parser.parse_args()

if args.window_size < 1:
    raise ValueError("--window-size must be at least 1.")

runtime_window_size = args.window_size

if args.execution_mode == "auto":
    RUN_EAGERLY = IS_INTEL_MAC
else:
    RUN_EAGERLY = args.execution_mode == "eager"

stock_symbol = args.symbol.lower()
task = args.task

classification_target = args.classification_target

if args.regression_targets:
    regression_targets = [
        name.strip()
        for name in args.regression_targets.split(",")
        if name.strip()
    ]
elif task == "multi_regression":
    regression_targets = list(REGRESSION_TARGETS)
else:
    regression_targets = ["LogReturn1"]

if not regression_targets:
    raise ValueError("At least one regression target must be provided.")

invalid_regression_targets = sorted(
    set(regression_targets) - set(REGRESSION_TARGETS)
)
if invalid_regression_targets:
    raise ValueError(
        "Regression targets must be selected from the continuous "
        f"target_classes columns. Invalid targets: {invalid_regression_targets}"
    )

if task == "classification":
    selected_target_columns = [classification_target]
elif task == "regression":
    selected_target_columns = regression_targets
else:
    selected_target_columns = regression_targets


def infer_target_horizon(target_name):
    """Return the numeric LogReturn horizon; Direction represents one bar."""

    marker = "LogReturn"
    if marker not in target_name:
        return 1

    suffix = target_name.rsplit(marker, maxsplit=1)[-1]
    if not suffix.isdigit():
        raise ValueError(
            f"Could not infer a horizon from target: {target_name}"
        )
    return int(suffix)


selected_target_horizons = [
    infer_target_horizon(target_name)
    for target_name in selected_target_columns
]
primary_horizon = max(selected_target_horizons)

# Keep output names target-specific so experiments do not overwrite one
# another when Direction, SPY, SMH, or different return targets are trained.
if task == "classification":
    target_tag = classification_target.lower()
elif task == "regression":
    target_tag = regression_targets[0].lower()
else:
    target_tag = "all_returns"

# Examples:
# "1 hour" -> "1hour"
# "1 DAY"  -> "1day"
bar_size = (
    str(args.bar_size)
    .strip()
    .replace(" ", "")
    .lower()
)

experiment_name = (
    args.experiment_name
    or f"train_lstm-{task}-{target_tag}-{bar_size}"
)

__MODEL_PATH = (
    str(resolve_from_project_directory(
        f"{MODEL_PATH}-{task}-{target_tag}-{bar_size}.keras"
    ))
)

__SCALER_PATH = (
    str(resolve_from_project_directory(
        f"{SCALER_PATH}-{task}-{target_tag}-{bar_size}.pkl"
    ))
)

__FEATURE_META_PATH = (
    str(resolve_from_project_directory(
        f"{FEATURE_META_PATH}-{task}-{target_tag}-{bar_size}.json"
    ))
)

set_seeds(RANDOM_SEED)
def chronological_split(
    df: pd.DataFrame,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC
):
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("train_frac and val_frac must be positive.")

    if train_frac + val_frac >= 1:
        raise ValueError(
            "train_frac + val_frac must be less than 1."
        )

    df = (
        df.copy()
        .sort_values(["datetime", TICKER_COLUMN])
        .reset_index(drop=True)
    )

    unique_datetimes = (
        df["datetime"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    n_dates = len(unique_datetimes)

    train_end = int(n_dates * train_frac)
    val_end = int(n_dates * (train_frac + val_frac))

    train_dates = unique_datetimes[:train_end]
    validation_dates = unique_datetimes[train_end:val_end]
    test_dates = unique_datetimes[val_end:]

    train_df = df[df["datetime"].isin(train_dates)].copy()
    validation_df = df[
        df["datetime"].isin(validation_dates)
    ].copy()
    test_df = df[df["datetime"].isin(test_dates)].copy()

    return train_df, validation_df, test_df


def create_selected_future_targets(df, selected_targets):
    """Create only the future targets requested for the current run."""

    df = (
        df.copy()
        .sort_values([TICKER_COLUMN, "datetime"])
        .reset_index(drop=True)
    )

    for target_name in selected_targets:
        target_spec = FUTURE_TARGET_SPECS.get(target_name)
        if target_spec is None:
            continue

        source_column = target_spec["source_column"]
        shift_periods = int(target_spec["shift_periods"])

        df[target_name] = (
            df.groupby(TICKER_COLUMN, sort=False)[source_column]
            .shift(-shift_periods)
        )

        print(
            f"Created {target_name}: row t uses "
            f"{source_column} from t+{shift_periods}."
        )

    return df.sort_values(
        ["datetime", TICKER_COLUMN]
    ).reset_index(drop=True)


def purge_future_target_boundary_rows(
    split_df,
    selected_targets,
    split_name,
):
    """Remove split-ending rows whose future labels cross a boundary."""

    purge_periods = max(
        (
            int(FUTURE_TARGET_SPECS[target_name]["shift_periods"])
            for target_name in selected_targets
            if target_name in FUTURE_TARGET_SPECS
        ),
        default=0,
    )

    if purge_periods == 0:
        return split_df.copy()

    split_df = (
        split_df.copy()
        .sort_values([TICKER_COLUMN, "datetime"])
    )

    rows_per_ticker = split_df.groupby(
        TICKER_COLUMN,
        sort=False,
    ).size()
    too_short = rows_per_ticker[rows_per_ticker <= purge_periods]
    if not too_short.empty:
        raise ValueError(
            f"{split_name} does not have enough rows to purge "
            f"{purge_periods} future-target boundary row(s):\n"
            f"{too_short.to_string()}"
        )

    distance_from_ticker_end = split_df.groupby(
        TICKER_COLUMN,
        sort=False,
    ).cumcount(ascending=False)
    purge_mask = distance_from_ticker_end < purge_periods

    purged_df = split_df.loc[~purge_mask].copy()
    number_of_tickers = int(split_df[TICKER_COLUMN].nunique())
    expected_purged_rows = number_of_tickers * purge_periods
    actual_purged_rows = int(purge_mask.sum())

    if actual_purged_rows != expected_purged_rows:
        raise AssertionError(
            f"{split_name} purged {actual_purged_rows} rows; "
            f"expected {expected_purged_rows}."
        )

    print(
        f"{split_name}: purged {actual_purged_rows} ending rows "
        f"({purge_periods} per ticker) for future-target isolation."
    )

    return purged_df.sort_values(
        ["datetime", TICKER_COLUMN]
    ).reset_index(drop=True)


def create_sequences(X, targets, sequence_length):
    """
    Create LSTM sequences for either:

    1. A single NumPy target array
    2. A dictionary containing multiple target arrays
    """
    X_sequences = []

    if isinstance(targets, dict):
        target_sequences = {
            target_name: []
            for target_name in targets
        }
    else:
        target_sequences = []

    for end_index in range(
        sequence_length - 1,
        len(X)
    ):
        start_index = end_index - sequence_length + 1

        X_sequences.append(
            X[start_index:end_index + 1]
        )

        if isinstance(targets, dict):
            for target_name, target_values in targets.items():
                target_sequences[target_name].append(
                    target_values[end_index]
                )
        else:
            target_sequences.append(
                targets[end_index]
            )

    X_sequences = np.asarray(
        X_sequences,
        dtype=np.float32
    )

    if isinstance(targets, dict):
        target_sequences = {
            target_name: np.asarray(
                values,
                dtype=np.float32
            )
            for target_name, values in target_sequences.items()
        }
    else:
        target_sequences = np.asarray(
            target_sequences,
            dtype=np.float32
        )

    return X_sequences, target_sequences
###### important for multiple stock sequeces, 
# we need to create sequences for each stock separately and then concatenate them together. 
# This ensures that the sequences are not mixed across different stocks, 
# which could lead to data leakage and incorrect model training.
def create_ticker_sequences(
    split_df,
    transform_function,
    sequence_length
):
    all_X = []
    all_targets = None

    for ticker, ticker_df in split_df.groupby(TICKER_COLUMN,sort=False):
        ticker_df = (ticker_df.sort_values("datetime").reset_index(drop=True))
        if len(ticker_df) < sequence_length:
            continue

        X, targets = transform_function(ticker_df)

        X_seq, target_seq = create_sequences(X,targets,sequence_length)
        all_X.append(X_seq)

        if isinstance(target_seq, dict):
            if all_targets is None:
                all_targets = {
                    name: []
                    for name in target_seq
                }

            for name, values in target_seq.items():
                all_targets[name].append(values)

        else:
            if all_targets is None:
                all_targets = []

            all_targets.append(target_seq)

    if not all_X:
        raise ValueError(
            "No sequences were created. Check sequence length "
            "and the number of rows per ticker."
        )

    combined_X = np.concatenate(all_X,axis=0)

    if isinstance(all_targets, dict):
        combined_targets = {
            name: np.concatenate(values, axis=0)
            for name, values in all_targets.items()
        }
    else:
        combined_targets = np.concatenate(
            all_targets,
            axis=0
        )

    return combined_X, combined_targets


def validate_combined_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and retain only identifiers, fixed features, and targets."""

    required_columns = set(MODEL_DATA_COLUMNS)
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Combined dataset is missing required columns: {missing_columns}"
        )

    # Drop every unrelated source column before splitting. datetime and
    # ticker_id are retained only for chronological splitting and per-stock
    # sequence creation; neither is passed to the LSTM.
    df = df.loc[:, MODEL_DATA_COLUMNS].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    df[TICKER_COLUMN] = pd.to_numeric(
        df[TICKER_COLUMN],
        errors="raise",
    )

    if df[TICKER_COLUMN].isna().any():
        raise ValueError("ticker_id contains missing values.")

    for column in [*LESS_COMPLEX_MATRIX, *SOURCE_TARGET_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    duplicate_mask = df.duplicated([TICKER_COLUMN, "datetime"], keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, [TICKER_COLUMN, "datetime"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate (ticker_id, datetime) rows were found. "
            f"Examples: {examples}"
        )

    # Every ticker_id must have exactly the same ordered observations.
    ticker_dates = {
        ticker: group["datetime"].sort_values().reset_index(drop=True)
        for ticker, group in df.groupby(TICKER_COLUMN, sort=True)
    }
    reference_ticker = next(iter(ticker_dates))
    reference_dates = ticker_dates[reference_ticker]
    alignment_errors = []
    for ticker, dates in ticker_dates.items():
        if not dates.equals(reference_dates):
            missing = reference_dates[~reference_dates.isin(dates)].head(5)
            extra = dates[~dates.isin(reference_dates)].head(5)
            alignment_errors.append(
                f"{ticker}: rows={len(dates)}, "
                f"missing={missing.astype(str).tolist()}, "
                f"extra={extra.astype(str).tolist()}"
            )

    if alignment_errors:
        raise ValueError(
            "ticker_id datetime observations are not exactly aligned:\n"
            + "\n".join(alignment_errors)
        )

    return df.sort_values(
        ["datetime", TICKER_COLUMN]
    ).reset_index(drop=True)


def validate_finite_features(df, feature_columns, dataset_name):
    values = df[feature_columns].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    invalid_mask = ~np.isfinite(values)
    if not invalid_mask.any():
        return

    invalid_counts = pd.Series(
        invalid_mask.sum(axis=0),
        index=feature_columns,
    )
    invalid_counts = invalid_counts[
        invalid_counts.gt(0)
    ].sort_values(ascending=False)
    raise ValueError(
        f"{dataset_name} contains NaN or infinity in selected features:\n"
        f"{invalid_counts.to_string()}"
    )


def validate_selected_targets(df, selected_targets):
    values = df[selected_targets].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    invalid_mask = ~np.isfinite(values)
    if invalid_mask.any():
        invalid_counts = pd.Series(
            invalid_mask.sum(axis=0),
            index=selected_targets,
        )
        invalid_counts = invalid_counts[
            invalid_counts.gt(0)
        ].sort_values(ascending=False)
        raise ValueError(
            "Selected targets contain NaN or infinity:\n"
            f"{invalid_counts.to_string()}"
        )

    if task == "classification":
        allowed_classes = {
            float(label)
            for label in DIRECTION_RAW_LABELS
        }
        for target_name in selected_targets:
            observed_classes = set(
                df[target_name]
                .astype(np.float64)
                .unique()
                .tolist()
            )
            if not observed_classes.issubset(allowed_classes):
                raise ValueError(
                    f"{target_name} must contain only the three "
                    "direction labels -1, 0, and 1. "
                    f"Observed values: {sorted(observed_classes)}"
                )


def prepare_data(df: pd.DataFrame, target_columns):
    """Use only LESS_COMPLEX_MATRIX for X and keep targets only as labels."""

    feature_columns = list(LESS_COMPLEX_MATRIX)

    # A future target does not exist yet at this stage. Validate its source
    # column now; the derived target itself is validated after split purging.
    pre_split_target_columns = list(dict.fromkeys(
        FUTURE_TARGET_SPECS.get(
            target_name,
            {"source_column": target_name},
        )["source_column"]
        for target_name in target_columns
    ))

    validate_finite_features(df, feature_columns, "Combined dataset")
    validate_selected_targets(df, pre_split_target_columns)

    print("Feature set: LESS_COMPLEX_MATRIX")
    print(f"Selected feature count: {len(feature_columns)}")
    print(f"Selected features: {feature_columns}")
    print(f"Target columns hidden from X: {target_classes}")
    print(f"Targets selected for this run: {target_columns}")

    return df, feature_columns



def build_classification_model(
    sequence_length,
    number_of_features,
    target_name,
):
    inputs = keras.Input(
        shape=(
            sequence_length,
            number_of_features
        )
    )

    x = layers.LSTM(
        64,
        return_sequences=True
    )(inputs)

    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(
        32,
        return_sequences=False
    )(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        16,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        NUMBER_OF_DIRECTION_CLASSES,
        activation="softmax",
        name=target_name
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
        run_eagerly=RUN_EAGERLY,
        jit_compile=False,
    )

    return model


# REGRESSION CHANGE 2: linear output; no sigmoid and no binary cross-entropy.
def build_regression_model(sequence_length, number_of_features, target_name):
    inputs = keras.Input(shape=(sequence_length, number_of_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear", name=target_name)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
        run_eagerly=RUN_EAGERLY,
        jit_compile=False,
    )
    return model


# REGRESSION CHANGE 3: one linear output head for every continuous return target.
def build_multi_regression_model(
    sequence_length,
    number_of_features,
    target_names,
):
    inputs = keras.Input(shape=(sequence_length, number_of_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    shared = layers.Dense(16, activation="relu")(x)

    outputs = {
        name: layers.Dense(1, activation="linear", name=name)(shared)
        for name in target_names
    }
    losses = {name: "mse" for name in target_names}
    metrics = {
        name: [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]
        for name in target_names
    }

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        metrics=metrics,
        run_eagerly=RUN_EAGERLY,
        jit_compile=False,
    )
    return model
def send_run_to_mlflow_tracker(
    metadata_path,
    experiment_name,
    task,
    bar_size,
    horizon,
    parameters=None,
    metrics=None,
):
    """
    Notify the containerized MLflow tracker that training has completed.

    This function sends only JSON data. It does not upload the Keras,
    pickle, or metadata files directly.

    The tracker container locates those files through the shared
    /models Docker volume.
    """

    tracker_base_url = os.getenv(
        "TRACKER_API_BASE_URL"
    )

    if not tracker_base_url:
        raise RuntimeError(
            "TRACKER_API_BASE_URL is not configured. "
            "Add it to the .env file."
        )

    timeout_seconds = int(
        os.getenv(
            "TRACKER_API_TIMEOUT_SECONDS",
            "120",
        )
    )

    metadata_path = os.path.abspath(
        metadata_path
    )

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadata file was not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(metadata_file)

    payload = {
        "task": task,
        "barSize": bar_size,
        "experimentName": experiment_name,
        "horizon": int(horizon),
        "metadata": metadata,
        "parameters": parameters or {},
        "metrics": metrics or {},
    }

    endpoint = (
        f"{tracker_base_url.rstrip('/')}/log-run"
    )

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout_seconds,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        response_details = ""

        if error.response is not None:
            response_details = (
                f"\nTracker response: "
                f"{error.response.text}"
            )

        raise RuntimeError(
            "The model was saved, but the request to the "
            f"MLflow Tracker API failed: {error}"
            f"{response_details}"
        ) from error

    result = response.json()

    print("\nMLflow tracking completed")
    print(f"Experiment: {result['experimentName']}")
    print(f"Run ID: {result['runId']}")
    print(f"Artifact URI: {result['artifactUri']}")

    return result
def main():
    os.makedirs(os.path.dirname(__MODEL_PATH) or ".", exist_ok=True)
    clean_bar_name = bar_size
    if args.data_path is None:
        data_path = resolve_from_project_directory(
            Path(DATA_DIR)
            / f"combined_semiconductor_{clean_bar_name}_feng.csv"
        )
    else:
        data_path = resolve_from_project_directory(args.data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Combined dataset was not found: {data_path}"
        )

    print(f"Loading combined dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = validate_combined_dataset(df)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("Rows per ticker_id before splitting:")
    print(df.groupby(TICKER_COLUMN).size().to_string())

    # Keep every target outside X. Derived future targets are created per
    # ticker here; their split-ending rows are purged after chronological
    # splitting so labels cannot cross train/validation/test boundaries.
    target_columns = list(selected_target_columns)

    df_full, feature_columns = prepare_data(
        df,
        target_columns,
    )

    df_full = create_selected_future_targets(
        df_full,
        target_columns,
    )

    # Split chronologically BEFORE fitting scaler
    train_df, validation_df, test_df = chronological_split(
        df_full,
        TRAIN_FRAC,
        VAL_FRAC,
    )

    # Verify the original chronological partitions before purging their ending
    # future-target rows.
    assert train_df["datetime"].max() < validation_df["datetime"].min()
    assert validation_df["datetime"].max() < test_df["datetime"].min()

    train_df = purge_future_target_boundary_rows(
        train_df,
        target_columns,
        "Train",
    )
    validation_df = purge_future_target_boundary_rows(
        validation_df,
        target_columns,
        "Validation",
    )
    test_df = purge_future_target_boundary_rows(
        test_df,
        target_columns,
        "Test",
    )

    for split_name, split_df in (
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ):
        validate_selected_targets(split_df, target_columns)

    print(
        "Train/Val/Test rows after target-boundary purge: "
        f"{len(train_df)}/{len(validation_df)}/{len(test_df)}"
    )
    print("Train:", train_df["datetime"].min()," to ", train_df["datetime"].max())

    print("Validation:", validation_df["datetime"].min()," to", validation_df["datetime"].max())

    print("Test:", test_df["datetime"].min()," to", test_df["datetime"].max())

    print("Train rows per ticker_id:")
    print(train_df.groupby(TICKER_COLUMN).size().to_string())
    print("Validation rows per ticker_id:")
    print(validation_df.groupby(TICKER_COLUMN).size().to_string())
    print("Test rows per ticker_id:")
    print(test_df.groupby(TICKER_COLUMN).size().to_string())

    # Fit every scaler on TRAINING data only.
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_columns])

    if task == "regression" and len(regression_targets) != 1:
        raise ValueError("regression requires exactly one --regression-targets column")
    if task == "multi_regression" and len(regression_targets) < 2:
        raise ValueError("multi_regression requires at least two target columns")

    # REGRESSION CHANGE 4: scale each continuous target independently.
    # This is especially helpful when targets use different horizons/scales.
    target_scalers = {}
    if task in {"regression", "multi_regression"}:
        for name in regression_targets:
            target_scaler = StandardScaler()
            target_scaler.fit(train_df[[name]])
            target_scalers[name] = target_scaler

    def transform_classification(split_df):
        X = feature_scaler.transform(split_df[feature_columns]).astype("float32")
        y = (
            split_df[classification_target]
            .map(DIRECTION_LABEL_TO_INDEX)
        )
        if y.isna().any():
            invalid_labels = sorted(
                split_df.loc[y.isna(), classification_target]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Could not encode {classification_target} labels: "
                f"{invalid_labels}"
            )
        y = y.to_numpy(dtype="int32")
        return X, y

    def transform_regression(split_df):
        target_name = regression_targets[0]
        X = feature_scaler.transform(split_df[feature_columns]).astype("float32")
        y = target_scalers[target_name].transform(
            split_df[[target_name]]
        ).ravel().astype("float32")
        return X, y

    def transform_multi_regression(split_df):
        X = feature_scaler.transform(split_df[feature_columns]).astype("float32")
        targets = {
            name: target_scalers[name].transform(
                split_df[[name]]
            ).ravel().astype("float32")
            for name in regression_targets
        }
        return X, targets

    scaler_bundle = {
        "feature_scaler": feature_scaler,
        "target_scalers": target_scalers,
        "feature_columns": feature_columns,
        "selected_target_columns": target_columns,
        "classification_label_to_index": (
            DIRECTION_LABEL_TO_INDEX
            if task == "classification"
            else None
        ),
        "classification_index_to_label": (
            DIRECTION_INDEX_TO_LABEL
            if task == "classification"
            else None
        ),
    }
    # Make sure the scaler and metadata directories exist.
    os.makedirs(os.path.dirname(__SCALER_PATH) or ".",exist_ok=True,)
    os.makedirs(os.path.dirname(__FEATURE_META_PATH) or ".",exist_ok=True,)

    # Save the feature scaler and target scalers.
    joblib.dump(scaler_bundle,__SCALER_PATH,)
    print(f"Scaler saved to {__SCALER_PATH}")

    feature_metadata = {
        "data_path": str(data_path),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_set": "LESS_COMPLEX_MATRIX",
        "available_target_classes": target_classes,
        "selected_target_columns": target_columns,
        "selected_future_target_specs": {
            target_name: FUTURE_TARGET_SPECS[target_name]
            for target_name in target_columns
            if target_name in FUTURE_TARGET_SPECS
        },
        "target_columns_in_model_input": False,
        "window_size": runtime_window_size,
        "horizon": primary_horizon,
        "target_horizons": selected_target_horizons,
        "bar_size": bar_size,
        "task": task,
        "regression_targets": (
            regression_targets
            if task in {"regression", "multi_regression"}
            else []
        ),
        "classification_target": (
            classification_target
            if task == "classification"
            else None
        ),
        "classification_raw_labels": (
            list(DIRECTION_RAW_LABELS)
            if task == "classification"
            else None
        ),
        "classification_label_to_index": (
            {
                str(raw_label): class_index
                for raw_label, class_index
                in DIRECTION_LABEL_TO_INDEX.items()
            }
            if task == "classification"
            else None
        ),
        "classification_index_to_label": (
            {
                str(class_index): raw_label
                for class_index, raw_label
                in DIRECTION_INDEX_TO_LABEL.items()
            }
            if task == "classification"
            else None
        ),
        "targets_are_scaled": bool(target_scalers),
    }

    with open(__FEATURE_META_PATH,"w", encoding="utf-8",) as metadata_file:
        json.dump(feature_metadata,metadata_file,indent=2,)

    print(f"Feature metadata saved to "f"{__FEATURE_META_PATH}")

    ############### end of json metadata ##############################
    
    # REGRESSION CHANGE 5: create and train only the selected experiment.
    if task == "classification":transform_function = transform_classification
    elif task == "regression":transform_function = transform_regression
    else:transform_function = transform_multi_regression

    X_train_seq, y_train_seq = create_ticker_sequences(
        train_df,
        transform_function,
        runtime_window_size,
    )
    X_validation_seq, y_validation_seq = create_ticker_sequences(
        validation_df,
        transform_function,
        runtime_window_size,
    )
    X_test_seq, y_test_seq = create_ticker_sequences(
        test_df,
        transform_function,
        runtime_window_size,
    )

    # Give TensorFlow predictable, C-contiguous float32 inputs. This is done
    # after sequence creation so both sequence functions remain unchanged.
    X_train_seq = np.ascontiguousarray(X_train_seq, dtype=np.float32)
    X_validation_seq = np.ascontiguousarray(
        X_validation_seq,
        dtype=np.float32,
    )
    X_test_seq = np.ascontiguousarray(X_test_seq, dtype=np.float32)

    class_weight = None
    training_sample_weight = None
    if task == "classification":
        # create_sequences deliberately returns float32 targets for generic
        # regression support. Sparse categorical classification, however,
        # must receive zero-based integer class indices.
        y_train_seq = np.ascontiguousarray(
            y_train_seq,
            dtype=np.int32,
        )
        y_validation_seq = np.ascontiguousarray(
            y_validation_seq,
            dtype=np.int32,
        )
        y_test_seq = np.ascontiguousarray(
            y_test_seq,
            dtype=np.int32,
        )
        y_train_class_indices = y_train_seq
        expected_class_indices = np.arange(
            NUMBER_OF_DIRECTION_CLASSES,
            dtype=np.int32,
        )
        observed_train_indices = np.unique(
            y_train_class_indices
        )
        missing_train_indices = np.setdiff1d(
            expected_class_indices,
            observed_train_indices,
        )
        if missing_train_indices.size:
            missing_raw_labels = [
                DIRECTION_INDEX_TO_LABEL[int(class_index)]
                for class_index in missing_train_indices
            ]
            raise ValueError(
                "The training split does not contain every direction class. "
                f"Missing raw labels: {missing_raw_labels}"
            )

        balanced_weights = compute_class_weight(
            class_weight="balanced",
            classes=expected_class_indices,
            y=y_train_class_indices,
        )
        class_weight = {
            int(class_index): float(weight)
            for class_index, weight in zip(
                expected_class_indices,
                balanced_weights,
            )
        }

        # Build explicit per-sequence weights instead of asking Keras to
        # convert class_weight inside its graph/data-adapter startup. This is
        # mathematically equivalent for this single-output classifier and
        # avoids the native down_cast abort seen on Intel macOS.
        training_sample_weight = np.ascontiguousarray(
            np.take(
                balanced_weights,
                y_train_class_indices,
            ),
            dtype=np.float32,
        )

        print("Classification label encoding:")
        for raw_label, class_index in DIRECTION_LABEL_TO_INDEX.items():
            class_count = int(
                np.sum(y_train_class_indices == class_index)
            )
            print(
                f"  raw {raw_label:>2} -> class {class_index}: "
                f"{class_count:,} training sequences, "
                f"weight={class_weight[class_index]:.4f}"
            )

        model = build_classification_model(
            X_train_seq.shape[1],
            X_train_seq.shape[2],
            classification_target,
        )
    elif task == "regression":
        model = build_regression_model(X_train_seq.shape[1],
                                       X_train_seq.shape[2],
                                       regression_targets[0],)
    else:
        model = build_multi_regression_model(X_train_seq.shape[1],
                                             X_train_seq.shape[2],
                                             regression_targets,)
        
        
    model.summary()
    # Callbacks
    early_stop = callbacks.EarlyStopping(
         monitor="val_loss",
        patience=12,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(__MODEL_PATH, monitor="val_loss", save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.5,
                                            patience=3,
                                            min_lr=1e-5,
                                            verbose=1)

    print("Number of tickers:", train_df[TICKER_COLUMN].nunique())
    print("Training rows:", len(train_df))
    print("Training sequences:", len(X_train_seq))
    print("Task:", task)
    print("TensorFlow version:", tf.__version__)
    print(
        "TensorFlow execution mode:",
        "eager" if RUN_EAGERLY else "graph",
    )
    print("JIT compilation: disabled")
    if task != "classification":
        print("Regression targets:", regression_targets)

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(
            X_validation_seq,
            y_validation_seq
        ),
        epochs=100,
        batch_size=64,
        shuffle=False,
        callbacks=[
            early_stop,
            checkpoint,
            reduce_lr
        ],
        sample_weight=training_sample_weight,
    )


    test_results = model.evaluate(
        X_test_seq,
        y_test_seq,
        verbose=1,
        return_dict=True,
    )
    print("Test results:", test_results)

    additional_tracking_metrics = {}

    if task == "classification":
        y_test_class_indices = np.asarray(
            y_test_seq,
            dtype=np.int32,
        )
        test_class_counts = np.bincount(
            y_test_class_indices,
            minlength=NUMBER_OF_DIRECTION_CLASSES,
        )
        majority_accuracy = (
            test_class_counts.max()
            / test_class_counts.sum()
        )

        class_probabilities = model.predict(
            X_test_seq,
            verbose=0,
        )
        predicted_class_indices = np.argmax(
            class_probabilities,
            axis=1,
        )
        confusion_matrix = tf.math.confusion_matrix(
            y_test_class_indices,
            predicted_class_indices,
            num_classes=NUMBER_OF_DIRECTION_CLASSES,
        ).numpy()

        class_precision, class_recall, class_f1, class_support = (
            precision_recall_fscore_support(
                y_test_class_indices,
                predicted_class_indices,
                labels=np.arange(NUMBER_OF_DIRECTION_CLASSES),
                zero_division=0,
            )
        )
        test_accuracy = float(
            np.mean(
                predicted_class_indices == y_test_class_indices
            )
        )
        supported_classes = class_support > 0
        balanced_accuracy = float(
            np.mean(class_recall[supported_classes])
        )
        macro_f1 = float(
            np.mean(class_f1[supported_classes])
        )
        accuracy_above_majority = float(
            test_accuracy - majority_accuracy
        )

        print("Test direction class counts:")
        for class_index, class_count in enumerate(test_class_counts):
            raw_label = DIRECTION_INDEX_TO_LABEL[class_index]
            print(
                f"  raw {raw_label:>2} / class {class_index}: "
                f"{int(class_count):,}"
            )
        print(
            "Majority-class baseline accuracy:",
            float(majority_accuracy),
        )
        print("Accuracy above majority baseline:", accuracy_above_majority)
        print("Balanced accuracy:", balanced_accuracy)
        print("Macro F1:", macro_f1)
        print("Per-class benchmark metrics:")
        for class_index, raw_label in DIRECTION_INDEX_TO_LABEL.items():
            print(
                f"  raw {raw_label:>2} / class {class_index}: "
                f"precision={class_precision[class_index]:.4f}, "
                f"recall={class_recall[class_index]:.4f}, "
                f"f1={class_f1[class_index]:.4f}, "
                f"support={int(class_support[class_index]):,}"
            )
        print(
            "Confusion matrix rows=true, columns=predicted "
            "(class order: raw -1, 0, 1):"
        )
        print(confusion_matrix)

        additional_tracking_metrics.update({
            "test_accuracy": test_accuracy,
            "majority_baseline_accuracy": float(majority_accuracy),
            "accuracy_above_majority_baseline": accuracy_above_majority,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        })
        metric_names_by_raw_label = {
            -1: "sell",
            0: "hold",
            1: "buy",
        }
        for class_index, raw_label in DIRECTION_INDEX_TO_LABEL.items():
            metric_label = metric_names_by_raw_label[raw_label]
            additional_tracking_metrics.update({
                f"{metric_label}_precision": float(
                    class_precision[class_index]
                ),
                f"{metric_label}_recall": float(
                    class_recall[class_index]
                ),
                f"{metric_label}_f1": float(class_f1[class_index]),
                f"{metric_label}_support": float(
                    class_support[class_index]
                ),
            })
    else:
        # REGRESSION CHANGE 6: compare against a zero-return baseline and
        # report errors in the ORIGINAL return units, not standardized units.
        predictions = model.predict(X_test_seq, verbose=0)
        if task == "regression":
            predictions = {regression_targets[0]: np.asarray(predictions).ravel()}
            actuals = {regression_targets[0]: y_test_seq}
        else:
            if not isinstance(predictions, dict):
                predictions = dict(zip(model.output_names, predictions))
            actuals = y_test_seq

        for name in regression_targets:
            predicted_original = target_scalers[name].inverse_transform(
                np.asarray(predictions[name]).reshape(-1, 1)
            ).ravel()
            actual_original = target_scalers[name].inverse_transform(
                np.asarray(actuals[name]).reshape(-1, 1)
            ).ravel()
            mae = np.mean(np.abs(actual_original - predicted_original))
            rmse = np.sqrt(np.mean((actual_original - predicted_original) ** 2))
            zero_baseline_mae = np.mean(np.abs(actual_original))
            directional_accuracy = np.mean(
                np.sign(predicted_original) == np.sign(actual_original)
            )
            print(f"{name} original-scale MAE: {mae:.8f}")
            print(f"{name} original-scale RMSE: {rmse:.8f}")
            print(f"{name} zero-return baseline MAE: {zero_baseline_mae:.8f}")
            print(f"{name} directional accuracy: {directional_accuracy:.4f}")
    
    
    ################ save model ###############
    # Save the final model before notifying MLflow.
    model.save(__MODEL_PATH)

    print(f"Model saved to {__MODEL_PATH}")
    print(f"Scaler saved to {__SCALER_PATH}")
    print(
        f"Metadata saved to {__FEATURE_META_PATH}"
    )

    # Convert TensorFlow/NumPy values into regular Python floats
    # so they can be serialized safely as JSON.
    tracking_metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in test_results.items()
    }
    tracking_metrics.update(additional_tracking_metrics)

    tracking_parameters = {
        "ticker_ids": ",".join(
            str(ticker_id)
            for ticker_id in sorted(df_full[TICKER_COLUMN].unique())
        ),
        "number_of_tickers": int(df_full[TICKER_COLUMN].nunique()),
        "feature_set": "LESS_COMPLEX_MATRIX",
        "selected_targets": ",".join(target_columns),
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "window_size": runtime_window_size,
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "number_of_features": len(feature_columns),
        "number_of_training_rows": len(train_df),
        "number_of_validation_rows": len(
            validation_df
        ),
        "number_of_test_rows": len(test_df),
        "number_of_training_sequences": len(
            X_train_seq
        ),
        "classification_raw_labels": (
            ",".join(str(label) for label in DIRECTION_RAW_LABELS)
            if task == "classification"
            else ""
        ),
        "classification_class_weights": (
            json.dumps(class_weight, sort_keys=True)
            if class_weight is not None
            else ""
        ),
    }

    # send_run_to_mlflow_tracker(
    #     metadata_path=__FEATURE_META_PATH,
    #     experiment_name=experiment_name,
    #     task=task,
    #     bar_size=bar_size,
    #     horizon=primary_horizon,
    #     parameters=tracking_parameters,
    #     metrics=tracking_metrics,
    # )


if __name__ == "__main__":
    main()