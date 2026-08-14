#!/usr/bin/env python3
"""
train_tabular_baseline.py — non-sequential baselines for market_ml.

Version 1.0.0

Purpose
-------
Implements the baseline comparison required by the research spec:

  spec section 1  : "Logistic regression or tree-based tabular model —
                     tests whether sequence modeling adds value over
                     lagged/aggregated features."
  spec section 7  : "Baseline comparison — improvement versus majority,
                     persistence, and tabular baselines on the same
                     held-out timestamps."
  spec section 8.3: "Run the non-sequential baselines" (before the LSTM
                     is retained).

This script answers ONE question: does the causal LSTM actually beat a
non-sequential model given the SAME feature matrix, SAME splits, SAME
label policy and SAME held-out timestamps?

What it does NOT do
-------------------
- It does not create or modify any Parquet artifact.
- It does not build new sequences.
- It does not introduce new data sources (no SPY/SMH, no news, no ACD).
- It does not refit the scaler. The already-train-fitted scaled matrix
  from model_matrix.py is used unchanged.
- It does not evaluate TEST.

Feature construction
--------------------
The existing L120 x F window is reduced to a tabular row using only
lags and aggregates of the SAME features, which is precisely what the
spec means by "lagged/aggregated features":

  - lag features : feature values at selected offsets back from the
                   window end (0 = final input bar)
  - aggregates   : per-feature mean and standard deviation across the
                   full window

No indicator is recomputed and no new market data is read.

Leakage controls
----------------
1. Splits are reused exactly as assigned by sequences.py (by trading
   date). They are never recomputed here.
2. An EXPLICIT purge/embargo step drops any TRAIN sample whose
   target_realized_at falls at or after the first VALIDATION
   prediction_time. On regular-session data this normally drops zero
   rows, because the trading-date split already separates them by the
   overnight gap. The check exists to prove the property rather than
   assume it, and it reports the realized embargo gap in minutes.
3. The scaler is not refit.
4. TEST rows are counted but never loaded into a model, scored, or
   printed as a class distribution.

Models
------
  majority     : always predict the most frequent TRAIN class
  persistence  : predict the sign of the most recent observed return
                 (raw scale recovered from scaler.json); 3-class mode
                 maps it to down/up only
  logistic     : multinomial logistic regression
  gbm          : sklearn HistGradientBoostingClassifier
  xgboost      : optional, only if the package is installed

Usage
-----
  python src/train_tabular_baseline.py \
    --project-root . \
    --model-matrix-root data/parquet/model_matrix \
    --label-policy-root data/parquet/label_policy \
    --symbol nvda \
    --config config/pipeline.yaml \
    --prediction-session regular \
    --target-same-session-only \
    --experiment-label binary_sign \
    --reports reports/tabular_baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TABULAR_BASELINE_VERSION = "1.0.0"
SUPPORTED_TRAIN_MODEL_PREFIX = "1.1."

# Offsets back from the final input bar of the window. 0 is the final
# bar (the decision bar), 119 is the oldest bar in an L120 window.
DEFAULT_LAG_OFFSETS = (0, 1, 2, 3, 5, 10, 20, 40, 60, 90, 119)


def import_train_model(project_root: Path):
    path = project_root / "src" / "train_model.py"

    if not path.exists():
        raise FileNotFoundError(f"Missing base trainer: {path}")

    spec = importlib.util.spec_from_file_location(
        "market_ml_base_train_model",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    version = str(getattr(module, "TRAIN_MODEL_VERSION", ""))

    if not version:
        raise RuntimeError("Base train_model.py has no TRAIN_MODEL_VERSION.")

    if not version.startswith(SUPPORTED_TRAIN_MODEL_PREFIX):
        raise RuntimeError(
            "train_tabular_baseline.py expects base train_model.py "
            f"v1.1.x; found {version!r}."
        )

    return module


def to_ns(series: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(series, utc=True, errors="raise")
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False)
    )


def normalize_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    unknown = sorted(set(values.unique()) - set(mapping))

    if unknown:
        raise ValueError(f"Unrecognized boolean values: {unknown}")

    return values.map(mapping).to_numpy(dtype=bool)


def load_sequence_metadata(
    tm,
    matrix_root: Path,
    symbol: str,
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the regime/timing columns that train_model.load_sequences does
    not return, aligned one-to-one onto the model-ready sequence rows.
    """
    sequence_parts = tm.parts(matrix_root / "sequences", symbol)

    required = [
        "source_id",
        "prediction_time",
        "split",
        "session",
        "target_same_session",
        "target_realized_at",
    ]

    first_columns = list(
        pd.read_parquet(sequence_parts[0], engine="pyarrow").columns
    )

    missing = sorted(set(required) - set(first_columns))

    if missing:
        raise ValueError(
            f"Model-ready sequences missing required columns: {missing}"
        )

    frames = []

    for i, path in enumerate(sequence_parts, 1):
        frames.append(
            pd.read_parquet(path, engine="pyarrow", columns=required)
        )

        if i % 20 == 0 or i == len(sequence_parts):
            print(
                f"  sequence metadata: loaded {i:>3}/{len(sequence_parts)} files"
            )

    meta = pd.concat(frames, ignore_index=True)

    meta["source_id"] = meta["source_id"].astype(str).str.strip().str.lower()
    meta["split"] = meta["split"].astype(str).str.strip().str.lower()
    meta["session"] = meta["session"].astype(str).str.strip().str.lower()
    meta["_prediction_ns"] = to_ns(meta["prediction_time"])
    meta["_realized_ns"] = to_ns(meta["target_realized_at"])
    meta["_target_same_session"] = normalize_bool(meta["target_same_session"])

    if meta.duplicated(["source_id", "_prediction_ns"]).any():
        raise ValueError("Duplicate sequence metadata endpoint keys.")

    keys = sequences[["source_id", "_prediction_ns", "split"]].copy()
    keys["_row"] = np.arange(len(keys), dtype=np.int64)

    joined = keys.merge(
        meta[
            [
                "source_id",
                "_prediction_ns",
                "split",
                "session",
                "_target_same_session",
                "_realized_ns",
            ]
        ].rename(columns={"split": "_meta_split"}),
        on=["source_id", "_prediction_ns"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row")

    if joined["session"].isna().any():
        raise ValueError(
            "Missing sequence metadata after endpoint alignment: "
            f"{int(joined['session'].isna().sum()):,}"
        )

    if (joined["split"] != joined["_meta_split"]).any():
        raise ValueError("Sequence metadata split mismatch.")

    return joined.reset_index(drop=True)


def binary_sign_labels(sequences: pd.DataFrame):
    returns = pd.to_numeric(
        sequences["target_return_bps"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(returns).all():
        raise ValueError("Non-finite target_return_bps.")

    keep = returns != 0.0

    labels = np.zeros(len(returns), dtype=np.int64)
    labels[returns > 0.0] = 1

    info = {
        "name": "binary_sign",
        "threshold_bps": 0.0,
        "zero_return_rows_excluded": int((~keep).sum()),
    }

    return labels, keep, ("down", "up"), info


def apply_purge(
    split: np.ndarray,
    realized_ns: np.ndarray,
    prediction_ns: np.ndarray,
):
    """
    Explicit purge/embargo.

    Any TRAIN sample whose target matures at or after the first
    VALIDATION decision time is dropped, because its label encodes
    price information from inside the validation period.

    Returns the keep mask and a diagnostic payload.
    """
    train_mask = split == "train"
    val_mask = split == "validation"

    if not train_mask.any():
        raise ValueError("No train rows available for purge check.")

    if not val_mask.any():
        raise ValueError("No validation rows available for purge check.")

    first_val_ns = int(prediction_ns[val_mask].min())

    offending = train_mask & (realized_ns >= first_val_ns)

    keep = ~offending

    last_train_realized_ns = int(
        realized_ns[train_mask & keep].max()
    )

    gap_minutes = (
        first_val_ns - last_train_realized_ns
    ) / 60_000_000_000.0

    diagnostics = {
        "purged_train_rows": int(offending.sum()),
        "first_validation_prediction_time_utc": str(
            pd.Timestamp(first_val_ns, tz="UTC")
        ),
        "last_retained_train_target_realized_at_utc": str(
            pd.Timestamp(last_train_realized_ns, tz="UTC")
        ),
        "realized_embargo_gap_minutes": float(gap_minutes),
        "policy": (
            "TRAIN samples whose target_realized_at >= the first "
            "VALIDATION prediction_time are removed so no training "
            "label can encode validation-period prices."
        ),
    }

    if gap_minutes <= 0:
        raise AssertionError(
            "Purge failed to produce a positive embargo gap: "
            f"{gap_minutes:.3f} minutes."
        )

    return keep, diagnostics


def build_tabular_features(
    matrix: np.ndarray,
    starts: np.ndarray,
    sequence_length: int,
    feature_names: tuple[str, ...],
    lag_offsets: tuple[int, ...],
    batch_size: int = 20000,
):
    """
    Reduce each L x F window to a single tabular row of lags and
    aggregates. Built in batches so peak memory stays bounded.
    """
    invalid = [
        offset
        for offset in lag_offsets
        if offset < 0 or offset >= sequence_length
    ]

    if invalid:
        raise ValueError(
            f"Lag offsets outside window [0,{sequence_length - 1}]: {invalid}"
        )

    feature_count = len(feature_names)
    sample_count = len(starts)

    column_names = []

    for offset in lag_offsets:
        for name in feature_names:
            column_names.append(f"{name}__lag{offset}")

    for name in feature_names:
        column_names.append(f"{name}__win_mean")

    for name in feature_names:
        column_names.append(f"{name}__win_std")

    width = len(lag_offsets) * feature_count + 2 * feature_count

    out = np.empty((sample_count, width), dtype=np.float32)

    # Position of the final input bar relative to window start.
    end_position = sequence_length - 1

    window_offsets = np.arange(sequence_length, dtype=np.int64)

    for low in range(0, sample_count, batch_size):
        high = min(low + batch_size, sample_count)
        batch_starts = starts[low:high]

        positions = (
            batch_starts[:, None] + window_offsets[None, :]
        )

        window = matrix[positions]

        cursor = 0

        for offset in lag_offsets:
            column = window[:, end_position - offset, :]
            out[low:high, cursor:cursor + feature_count] = column
            cursor += feature_count

        out[low:high, cursor:cursor + feature_count] = window.mean(axis=1)
        cursor += feature_count

        out[low:high, cursor:cursor + feature_count] = window.std(axis=1)
        cursor += feature_count

        if (low // batch_size) % 10 == 0 or high == sample_count:
            print(f"  tabular features: {high:,}/{sample_count:,} rows")

    return out, column_names


def unscale_column(
    scaled_values: np.ndarray,
    feature_name: str,
    scaler_payload: dict,
) -> np.ndarray:
    """
    Recover raw feature units from the standard-scaled matrix so the
    persistence baseline can read a true return sign.
    """
    mean = float(scaler_payload["mean"][feature_name])
    scale = float(scaler_payload["scale"][feature_name])

    return scaled_values * scale + mean


def probability_metrics(truth: np.ndarray, probability_up: np.ndarray):
    """
    Binary ranking/calibration metrics computed without sklearn so the
    baseline report always contains them.
    """
    positive = probability_up[truth == 1]
    negative = probability_up[truth == 0]

    if len(positive) == 0 or len(negative) == 0:
        return {
            "roc_auc": float("nan"),
            "brier_score": float("nan"),
        }

    # Rank-based AUC (equivalent to the Mann-Whitney U statistic).
    order = np.argsort(probability_up, kind="mergesort")
    ranks = np.empty(len(probability_up), dtype=np.float64)
    ranks[order] = np.arange(1, len(probability_up) + 1, dtype=np.float64)

    # Average ranks inside tie groups.
    sorted_values = probability_up[order]
    start = 0

    while start < len(sorted_values):
        stop = start

        while (
            stop + 1 < len(sorted_values)
            and sorted_values[stop + 1] == sorted_values[start]
        ):
            stop += 1

        if stop > start:
            average = (start + stop + 2) / 2.0
            ranks[order[start:stop + 1]] = average

        start = stop + 1

    positive_rank_sum = ranks[truth == 1].sum()
    positive_count = float((truth == 1).sum())
    negative_count = float((truth == 0).sum())

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    brier = float(np.mean((probability_up - truth) ** 2))

    return {
        "roc_auc": float(auc),
        "brier_score": brier,
    }


def precision_at_coverage(
    truth: np.ndarray,
    probability_up: np.ndarray,
    coverages=(0.01, 0.05, 0.10, 0.20, 0.50),
):
    """
    Selective-prediction diagnostic: if the model only acted on its most
    confident predictions, how precise would those calls be?

    Confidence is |p - 0.5|. Reported per coverage level for both
    directions combined.
    """
    confidence = np.abs(probability_up - 0.5)
    order = np.argsort(-confidence, kind="mergesort")

    results = []

    for coverage in coverages:
        count = int(np.floor(len(truth) * coverage))

        if count <= 0:
            continue

        selected = order[:count]
        predicted = (probability_up[selected] >= 0.5).astype(np.int64)
        correct = int((predicted == truth[selected]).sum())

        results.append(
            {
                "coverage": float(coverage),
                "samples": int(count),
                "accuracy": float(correct / count),
                "predicted_up_pct": float(
                    100.0 * predicted.mean()
                ),
            }
        )

    return results


def evaluate(
    tm,
    name: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    class_names: tuple[str, ...],
    probability_up: np.ndarray | None = None,
):
    metrics = tm.validation_metrics(truth, predicted, class_names)

    payload = {
        "model": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": metrics["confusion_matrix_labels"],
        "per_class": metrics["per_class"],
    }

    if probability_up is not None and len(class_names) == 2:
        payload.update(probability_metrics(truth, probability_up))
        payload["precision_at_coverage"] = precision_at_coverage(
            truth,
            probability_up,
        )

    return payload


def print_result(payload: dict):
    line = (
        f"  {payload['model']:<14} "
        f"acc={payload['accuracy']:.4f}  "
        f"bal_acc={payload['balanced_accuracy']:.4f}  "
        f"macro_f1={payload['macro_f1']:.4f}"
    )

    if "roc_auc" in payload:
        line += f"  auc={payload['roc_auc']:.4f}"
        line += f"  brier={payload['brier_score']:.4f}"

    print(line)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Non-sequential baselines (majority, persistence, logistic "
            "regression, gradient boosting) on the existing market_ml "
            "model matrix."
        )
    )

    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--model-matrix-root", type=Path, required=True)
    p.add_argument("--label-policy-root", type=Path)
    p.add_argument("--symbol", required=True)
    p.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    p.add_argument("--reports", type=Path)
    p.add_argument("--run-id")

    p.add_argument(
        "--prediction-session",
        choices=["premarket", "regular", "aftermarket", "all"],
        default="regular",
    )

    p.add_argument(
        "--target-same-session-only",
        action="store_true",
    )

    p.add_argument(
        "--experiment-label",
        choices=["binary_sign", "atr3"],
        default="binary_sign",
    )

    p.add_argument(
        "--models",
        nargs="+",
        default=["majority", "persistence", "logistic", "gbm"],
        help="Subset of: majority persistence logistic gbm xgboost",
    )

    p.add_argument(
        "--lag-offsets",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAG_OFFSETS),
    )

    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Optional chronological subsample cap for TRAIN. Reduces "
            "memory/time. Validation is never subsampled."
        ),
    )

    p.add_argument(
        "--persistence-feature",
        default="f_1m_log_return_15",
        help=(
            "Feature whose raw sign defines the persistence baseline. "
            "Must be present in the configured feature set."
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run every data/leakage check, then exit before fitting.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    tm = import_train_model(project_root)

    raw = tm.load_yaml(args.config)
    address, feature_set, features, cfg = tm.resolve_config(raw)

    if args.experiment_label == "atr3" and args.label_policy_root is None:
        raise ValueError(
            "--label-policy-root is required for --experiment-label atr3."
        )

    symbol = args.symbol.strip().lower()
    expected_signature = tm.signature(feature_set, features)

    matrix_root = (
        address.root(args.model_matrix_root)
        / f"feature_set={feature_set}"
    )

    scaler_path = matrix_root / "scaler.json"
    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))

    if scaler_payload.get("feature_signature") != expected_signature:
        raise ValueError("scaler.json feature signature mismatch.")

    if tuple(scaler_payload.get("feature_names", ())) != features:
        raise ValueError("scaler.json feature order mismatch.")

    if args.persistence_feature not in features:
        raise ValueError(
            f"--persistence-feature {args.persistence_feature!r} is not in "
            f"the configured feature set {feature_set!r}."
        )

    print(
        f"train_tabular_baseline.py={TABULAR_BASELINE_VERSION} "
        f"base_train_model.py={tm.TRAIN_MODEL_VERSION}"
    )
    print(
        f"symbol={symbol} session={args.prediction_session} "
        f"same_target={args.target_same_session_only} "
        f"label={args.experiment_label}"
    )
    print("Parquet impact: READ-ONLY. TEST is never evaluated.")

    print("\n[1/6] Loading scaled model matrix...")
    matrix, time_ns, source_code, code_map = tm.load_matrix(
        matrix_root,
        symbol,
        features,
        expected_signature,
    )

    print("\n[2/6] Loading model-ready sequences...")
    sequences = tm.load_sequences(matrix_root, symbol, expected_signature)

    print("\n[3/6] Loading sequence regime/timing metadata...")
    meta = load_sequence_metadata(tm, matrix_root, symbol, sequences)

    print("\n[4/6] Resolving labels...")

    if args.experiment_label == "atr3":
        labels, label_keep, class_names, label_info = tm.load_policy_labels(
            args.label_policy_root,
            sequences,
            address,
            feature_set,
            cfg,
            symbol,
        )
    else:
        labels, label_keep, class_names, label_info = binary_sign_labels(
            sequences
        )
        print(
            "Binary-sign labels: exact-zero excluded="
            f"{int((~label_keep).sum()):,}"
        )

    # ---------------------------------------------------------------
    # Regime filter
    # ---------------------------------------------------------------
    session_values = meta["session"].to_numpy(dtype=object)

    if args.prediction_session == "all":
        regime_keep = np.ones(len(sequences), dtype=bool)
    else:
        regime_keep = session_values == args.prediction_session

    if args.target_same_session_only:
        regime_keep = regime_keep & meta["_target_same_session"].to_numpy(
            dtype=bool
        )

    print(
        f"Regime filter: {int(regime_keep.sum()):,}/{len(sequences):,} "
        "sequences"
    )

    print("\n[5/6] Mapping windows and applying purge/embargo...")

    starts_all = tm.map_starts(
        time_ns,
        source_code,
        code_map,
        sequences,
        address.length,
    )

    split_all = sequences["split"].to_numpy(dtype=object)
    realized_ns_all = meta["_realized_ns"].to_numpy(dtype=np.int64)
    prediction_ns_all = meta["_prediction_ns"].to_numpy(dtype=np.int64)

    selected = regime_keep & label_keep

    if not selected.any():
        raise ValueError("Regime/label filter selected zero sequences.")

    purge_keep, purge_diagnostics = apply_purge(
        split_all[selected],
        realized_ns_all[selected],
        prediction_ns_all[selected],
    )

    selected_index = np.flatnonzero(selected)
    final_index = selected_index[purge_keep]

    split = split_all[final_index]
    starts = starts_all[final_index]
    y = labels[final_index]

    print(
        f"Purged TRAIN rows: {purge_diagnostics['purged_train_rows']:,}"
    )
    print(
        "Realized embargo gap: "
        f"{purge_diagnostics['realized_embargo_gap_minutes']:.1f} minutes "
        f"(target horizon is {address.horizon} minutes)"
    )
    print(f"Final selected sequences: {len(final_index):,}")

    train_index = np.flatnonzero(split == "train")
    val_index = np.flatnonzero(split == "validation")
    test_count = int((split == "test").sum())

    if not len(train_index):
        raise ValueError("No TRAIN rows after filtering.")

    if not len(val_index):
        raise ValueError("No VALIDATION rows after filtering.")

    if args.max_train_samples and len(train_index) > args.max_train_samples:
        keep_positions = np.linspace(
            0,
            len(train_index) - 1,
            args.max_train_samples,
            dtype=np.int64,
        )
        train_index = train_index[keep_positions]
        print(
            f"TRAIN chronologically subsampled to {len(train_index):,} rows."
        )

    print("\nSPLIT SUMMARY")
    print("-------------")
    print(f"TRAIN samples:      {len(train_index):,}")
    print(f"VALIDATION samples: {len(val_index):,}")
    print(f"TEST samples:       {test_count:,} (sealed, not evaluated)")

    for name, index in (
        ("TRAIN", train_index),
        ("VALIDATION", val_index),
    ):
        counts = np.bincount(y[index], minlength=len(class_names))
        print(f"\n{name} class distribution:")
        for i, class_name in enumerate(class_names):
            pct = 100.0 * counts[i] / max(1, len(index))
            print(f"  {class_name:<8} {counts[i]:>10,} ({pct:7.3f}%)")

    print("\nTEST class distribution: intentionally not shown.")

    if args.preflight_only:
        print("\nPRE-FLIGHT PASS")
        print("No model was fitted and no report was written.")
        return 0

    print("\n[6/6] Building tabular features...")

    lag_offsets = tuple(int(x) for x in args.lag_offsets)

    started = time.time()

    x_train, column_names = build_tabular_features(
        matrix,
        starts[train_index],
        address.length,
        features,
        lag_offsets,
    )

    x_val, _ = build_tabular_features(
        matrix,
        starts[val_index],
        address.length,
        features,
        lag_offsets,
    )

    y_train = y[train_index]
    y_val = y[val_index]

    print(
        f"Tabular design matrix: TRAIN {x_train.shape}, "
        f"VALIDATION {x_val.shape} "
        f"(~{(x_train.nbytes + x_val.nbytes) / 2**20:.1f} MiB)"
    )

    results = []

    print("\nRESULTS (validation only)")
    print("-------------------------")

    # ---------------------------------------------------------------
    # majority
    # ---------------------------------------------------------------
    if "majority" in args.models:
        majority_class = int(
            np.bincount(y_train, minlength=len(class_names)).argmax()
        )
        predicted = np.full(len(y_val), majority_class, dtype=np.int64)

        payload = evaluate(
            tm,
            "majority",
            y_val,
            predicted,
            class_names,
        )
        payload["majority_class"] = class_names[majority_class]
        results.append(payload)
        print_result(payload)

    # ---------------------------------------------------------------
    # persistence
    # ---------------------------------------------------------------
    if "persistence" in args.models:
        feature_position = features.index(args.persistence_feature)
        lag0_position = lag_offsets.index(0) if 0 in lag_offsets else None

        if lag0_position is None:
            raise ValueError(
                "Persistence baseline requires lag offset 0 in --lag-offsets."
            )

        column = lag0_position * len(features) + feature_position

        raw_return = unscale_column(
            x_val[:, column].astype(np.float64),
            args.persistence_feature,
            scaler_payload,
        )

        if len(class_names) == 2:
            predicted = (raw_return > 0).astype(np.int64)
        else:
            # 3-class: persistence has no notion of the neutral band.
            predicted = np.where(
                raw_return > 0,
                class_names.index("up"),
                class_names.index("down"),
            ).astype(np.int64)

        payload = evaluate(
            tm,
            "persistence",
            y_val,
            predicted,
            class_names,
        )
        payload["persistence_feature"] = args.persistence_feature
        results.append(payload)
        print_result(payload)

    # ---------------------------------------------------------------
    # sklearn models
    # ---------------------------------------------------------------
    needs_sklearn = {"logistic", "gbm"} & set(args.models)

    if needs_sklearn:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise SystemExit(
                "scikit-learn is required for the logistic/gbm baselines. "
                "Install it with: pip install scikit-learn"
            ) from exc

    if "logistic" in args.models:
        print("  fitting logistic regression...")

        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=args.seed,
        )
        model.fit(x_train, y_train)

        predicted = model.predict(x_val)
        probability = model.predict_proba(x_val)

        payload = evaluate(
            tm,
            "logistic",
            y_val,
            predicted,
            class_names,
            probability_up=(
                probability[:, 1] if len(class_names) == 2 else None
            ),
        )
        results.append(payload)
        print_result(payload)

    if "gbm" in args.models:
        print("  fitting gradient boosting...")

        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=None,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=args.seed,
        )
        model.fit(x_train, y_train)

        predicted = model.predict(x_val)
        probability = model.predict_proba(x_val)

        payload = evaluate(
            tm,
            "gbm",
            y_val,
            predicted,
            class_names,
            probability_up=(
                probability[:, 1] if len(class_names) == 2 else None
            ),
        )
        payload["iterations_run"] = int(model.n_iter_)
        results.append(payload)
        print_result(payload)

    if "xgboost" in args.models:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("  xgboost not installed; skipping.")
        else:
            print("  fitting xgboost...")

            model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=args.seed,
                n_jobs=-1,
                eval_metric="logloss",
            )
            model.fit(x_train, y_train)

            predicted = model.predict(x_val)
            probability = model.predict_proba(x_val)

            payload = evaluate(
                tm,
                "xgboost",
                y_val,
                predicted,
                class_names,
                probability_up=(
                    probability[:, 1] if len(class_names) == 2 else None
                ),
            )
            results.append(payload)
            print_result(payload)

    elapsed = time.time() - started

    run_id = (
        args.run_id
        or (
            f"{symbol}_{feature_set}_tabular_"
            f"{args.experiment_label}_{args.prediction_session}_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    )

    report = {
        "tool": "train_tabular_baseline.py",
        "version": TABULAR_BASELINE_VERSION,
        "base_train_model_version": tm.TRAIN_MODEL_VERSION,
        "status": "PASS",
        "run_id": run_id,
        "symbol": symbol,
        "feature_set": feature_set,
        "feature_signature": expected_signature,
        "source_feature_count": len(features),
        "sequence_length": address.length,
        "target_horizon_minutes": address.horizon,
        "source_sequence_scope": address.scope,
        "regime_filter": {
            "prediction_session": args.prediction_session,
            "target_same_session_only": args.target_same_session_only,
            "important_semantics": (
                "Prediction endpoint and target are regime-filtered. The "
                "existing continuous L120 window is reused and may causally "
                "include earlier-session bars."
            ),
        },
        "experiment_label": args.experiment_label,
        "label_info": label_info,
        "class_names": list(class_names),
        "tabular_representation": {
            "lag_offsets_from_window_end": list(lag_offsets),
            "aggregates": ["window_mean", "window_std"],
            "design_matrix_columns": len(column_names),
            "policy": (
                "Lags and aggregates of the SAME scaled features. No "
                "indicator is recomputed, no new data source is read, and "
                "the scaler is not refit."
            ),
        },
        "leakage_controls": {
            "split_source": (
                "Reused verbatim from sequences.py trading-date split."
            ),
            "scaler_refit": False,
            "purge": purge_diagnostics,
        },
        "train_samples_used": int(len(train_index)),
        "validation_samples_used": int(len(val_index)),
        "held_out_test_samples_not_evaluated": test_count,
        "train_class_distribution": {
            class_names[i]: int(count)
            for i, count in enumerate(
                np.bincount(y_train, minlength=len(class_names))
            )
        },
        "validation_class_distribution": {
            class_names[i]: int(count)
            for i, count in enumerate(
                np.bincount(y_val, minlength=len(class_names))
            )
        },
        "results": results,
        "total_seconds": float(elapsed),
        "test_policy": (
            "TEST receives the same structural filter but is never "
            "evaluated and its class distribution is never printed."
        ),
    }

    if args.reports:
        report_path = (
            args.reports
            / symbol
            / (
                f"tabular_baseline_v{TABULAR_BASELINE_VERSION}_"
                f"{feature_set}_L{address.length}_H{address.horizon}m_"
                f"{args.experiment_label}_{args.prediction_session}_"
                f"{run_id}.json"
            )
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"\nReport written: {report_path}")

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Held-out TEST not evaluated: {test_count:,} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
