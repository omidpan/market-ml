#!/usr/bin/env python3
"""
regular_binary_probability_diagnostics.py
Version 1.0.0

Validation-only probability / ranking diagnostics for the frozen R2 model:

    R2 = REGULAR prediction session
         target stays inside REGULAR
         binary_sign labels (DOWN / UP)
         existing continuous L120 input history

This script:
- loads only VALIDATION regular/same-target/nonzero-return sequences,
- reconstructs their exact L120 windows,
- loads the saved R2 best_model.keras,
- predicts VALIDATION only,
- computes:
    * ROC AUC from P(UP)
    * Brier score
    * binary log loss
    * probability summaries by actual class
    * a diagnostic threshold sweep from 0.30 to 0.70
    * month-by-month AUC / calibration metrics
    * regular-session time-block AUC / calibration metrics
- writes JSON/CSV reports when --reports is supplied.

Safety:
- TEST is not loaded or evaluated.
- No model training.
- No Parquet writes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "1.0.0"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def softmax_up_probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"Expected binary logits shape (N,2), got {values.shape}."
        )

    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    up = probs[:, 1]

    if not np.isfinite(up).all():
        raise ValueError("Non-finite UP probabilities.")

    if np.any((up < 0.0) | (up > 1.0)):
        raise ValueError("Probability outside [0,1].")

    return up


def roc_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)

    if len(y) != len(s):
        raise ValueError("AUC label/score length mismatch.")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Mann-Whitney U / average-rank AUC, including exact tie handling.
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    sum_pos = float(np.sum(ranks[y == 1]))

    auc = (
        sum_pos
        - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)


def brier_score(y_true: np.ndarray, p_up: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_up, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def binary_log_loss(y_true: np.ndarray, p_up: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_up, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)
    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        )
    )


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    p_up: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    pred = (np.asarray(p_up) >= threshold).astype(np.int64)

    down_mask = y == 0
    up_mask = y == 1

    down_support = int(np.sum(down_mask))
    up_support = int(np.sum(up_mask))

    down_recall = (
        float(np.mean(pred[down_mask] == 0))
        if down_support
        else float("nan")
    )
    up_recall = (
        float(np.mean(pred[up_mask] == 1))
        if up_support
        else float("nan")
    )

    balanced = float(np.mean([down_recall, up_recall]))
    accuracy = float(np.mean(pred == y))

    # Per-class precision/F1 without sklearn.
    per_class = {}
    f1_values = []

    for cls, name in ((0, "down"), (1, "up")):
        tp = int(np.sum((y == cls) & (pred == cls)))
        fp = int(np.sum((y != cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls)))

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )
        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(y == cls)),
        }
        f1_values.append(f1)

    return {
        "threshold": float(threshold),
        "samples": int(len(y)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": float(np.mean(f1_values)),
        "down_recall": down_recall,
        "up_recall": up_recall,
        "pred_down_pct": float(100.0 * np.mean(pred == 0)),
        "pred_up_pct": float(100.0 * np.mean(pred == 1)),
        "per_class": per_class,
    }


def probability_summary(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(
        x,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    names = [
        "p01", "p05", "p10", "p25", "p50",
        "p75", "p90", "p95", "p99",
    ]

    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        **{
            name: float(value)
            for name, value in zip(names, quantiles)
        },
    }


def ranking_bundle(y_true: np.ndarray, p_up: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(p_up, dtype=np.float64)

    return {
        "samples": int(len(y)),
        "actual_down_pct": float(100.0 * np.mean(y == 0)),
        "actual_up_pct": float(100.0 * np.mean(y == 1)),
        "roc_auc": roc_auc_binary(y, p),
        "brier_score": brier_score(y, p),
        "log_loss": binary_log_loss(y, p),
        "p_up_all": probability_summary(p),
        "p_up_given_actual_down": probability_summary(p[y == 0]),
        "p_up_given_actual_up": probability_summary(p[y == 1]),
    }


def print_ranking(title: str, bundle: dict):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    print(
        f"samples={bundle['samples']:,}  "
        f"ROC_AUC={bundle['roc_auc']:.6f}  "
        f"Brier={bundle['brier_score']:.6f}  "
        f"log_loss={bundle['log_loss']:.6f}"
    )
    print(
        f"actual DOWN={bundle['actual_down_pct']:.2f}% "
        f"UP={bundle['actual_up_pct']:.2f}%"
    )

    down = bundle["p_up_given_actual_down"]
    up = bundle["p_up_given_actual_up"]

    print(
        "P(UP) | actual DOWN: "
        f"mean={down['mean']:.4f} "
        f"p25={down['p25']:.4f} "
        f"p50={down['p50']:.4f} "
        f"p75={down['p75']:.4f}"
    )
    print(
        "P(UP) | actual UP:   "
        f"mean={up['mean']:.4f} "
        f"p25={up['p25']:.4f} "
        f"p50={up['p50']:.4f} "
        f"p75={up['p75']:.4f}"
    )


def fixed_probability_calibration(
    y_true: np.ndarray,
    p_up: np.ndarray,
) -> list[dict]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(p_up, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, 21)
    rows = []

    for i in range(len(edges) - 1):
        low = float(edges[i])
        high = float(edges[i + 1])

        if i == len(edges) - 2:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        count = int(np.sum(mask))
        if not count:
            continue

        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "samples": count,
                "mean_predicted_p_up": float(np.mean(p[mask])),
                "actual_up_rate": float(np.mean(y[mask] == 1)),
            }
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-matrix-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--symbol",
        default="nvda",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.005,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()

    diag_path = (
        project_root
        / "src"
        / "regular_binary_diagnostics.py"
    )
    if not diag_path.exists():
        raise FileNotFoundError(
            f"Required diagnostic helper not found: {diag_path}"
        )

    diag = import_module(
        diag_path,
        "regular_binary_diagnostics_helper",
    )

    tm = diag.import_train_model(project_root)

    raw = tm.load_yaml(args.config)
    address, feature_set, features, cfg = tm.resolve_config(raw)

    if address.length != 120:
        raise ValueError(f"Expected L120, found L{address.length}.")
    if address.horizon != 15:
        raise ValueError(f"Expected H15, found H{address.horizon}.")

    symbol = args.symbol.strip().lower()

    expected_signature = tm.signature(
        feature_set,
        features,
    )

    parameter_root = address.root(
        args.model_matrix_root
    )
    matrix_root = (
        parameter_root
        / f"feature_set={feature_set}"
    )

    scaler_path = matrix_root / "scaler.json"
    scaler = json.loads(
        scaler_path.read_text(encoding="utf-8")
    )

    if scaler.get("feature_signature") != expected_signature:
        raise ValueError("scaler.json feature signature mismatch.")
    if tuple(scaler.get("feature_names", ())) != features:
        raise ValueError("scaler.json feature order mismatch.")

    print(
        f"regular_binary_probability_diagnostics.py={VERSION} "
        f"base_train_model.py={tm.TRAIN_MODEL_VERSION}"
    )
    print(
        "R2 = REGULAR | same-target | binary_sign | "
        "continuous L120 input history"
    )
    print("VALIDATION ONLY. TEST will not be loaded or evaluated.")

    print("\n[1/4] Loading R2 VALIDATION sequences...")

    sequences = diag.load_validation_sequences(
        matrix_root
        / "sequences"
        / f"symbol={symbol}",
        expected_signature,
        address.length,
    )

    labels = sequences["_label"].to_numpy(dtype=np.int64)

    if len(sequences) != 92542:
        print(
            "WARNING: validation row count differs from the frozen R2 "
            f"reference 92,542; observed {len(sequences):,}."
        )

    print("\n[2/4] Loading matrix months required by VALIDATION...")

    (
        matrix,
        time_ns,
        source_code,
        code_map,
    ) = diag.load_required_matrix(
        matrix_root
        / "rows"
        / f"symbol={symbol}",
        sequences,
        features,
        expected_signature,
    )

    print("\n[3/4] Mapping exact R2 validation L120 windows...")

    starts = tm.map_starts(
        time_ns,
        source_code,
        code_map,
        sequences,
        address.length,
    )

    print(
        f"Validation window mapping: PASS "
        f"({len(starts):,} sequences)"
    )

    print("\n[4/4] Loading R2 model and predicting VALIDATION...")

    keras, tf = tm.import_ml_stack()

    gpus = tf.config.list_physical_devices("GPU")
    use_mixed = bool(gpus) and cfg.mixed_precision

    keras.mixed_precision.set_global_policy(
        "mixed_float16"
        if use_mixed
        else "float32"
    )

    print(
        f"TensorFlow={tf.__version__} "
        f"Keras={keras.__version__} "
        f"GPU={[x.name for x in gpus]} "
        f"mixed_precision={use_mixed}"
    )

    model = keras.models.load_model(args.model)

    WindowDataset = tm.make_window_dataset_class(keras)

    dataset = WindowDataset(
        matrix,
        starts,
        labels,
        address.length,
        args.batch_size or cfg.batch_size,
        False,
        cfg.seed + 3000,
        1,
    )

    logits = model.predict(dataset, verbose=1)
    p_up = softmax_up_probability(logits)

    if len(p_up) != len(labels):
        raise ValueError("Probability/label length mismatch.")

    default_metrics = binary_metrics_at_threshold(
        labels,
        p_up,
        0.50,
    )

    print("\n" + "=" * 86)
    print("R2 DEFAULT THRESHOLD REPRODUCTION")
    print("=" * 86)
    print(
        f"threshold=0.500  "
        f"accuracy={default_metrics['accuracy']:.4f}  "
        f"balanced_accuracy={default_metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={default_metrics['macro_f1']:.4f}"
    )
    print(
        f"DOWN recall={default_metrics['down_recall']:.4f}  "
        f"UP recall={default_metrics['up_recall']:.4f}  "
        f"pred DOWN={default_metrics['pred_down_pct']:.2f}%  "
        f"pred UP={default_metrics['pred_up_pct']:.2f}%"
    )

    overall = ranking_bundle(labels, p_up)
    print_ranking(
        "R2 VALIDATION — PROBABILITY / RANKING",
        overall,
    )

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max
        + args.threshold_step / 2.0,
        args.threshold_step,
        dtype=np.float64,
    )

    sweep_rows = [
        binary_metrics_at_threshold(
            labels,
            p_up,
            float(threshold),
        )
        for threshold in thresholds
    ]

    sweep = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "per_class"
            }
            for row in sweep_rows
        ]
    )

    max_balanced = float(
        sweep["balanced_accuracy"].max()
    )
    candidate = sweep[
        np.isclose(
            sweep["balanced_accuracy"],
            max_balanced,
            rtol=0.0,
            atol=1e-15,
        )
    ].copy()

    candidate["_distance_from_0_5"] = (
        candidate["threshold"] - 0.5
    ).abs()

    best_threshold_row = candidate.sort_values(
        ["_distance_from_0_5", "threshold"]
    ).iloc[0]

    best_threshold = float(
        best_threshold_row["threshold"]
    )

    best_threshold_metrics = binary_metrics_at_threshold(
        labels,
        p_up,
        best_threshold,
    )

    print("\n" + "=" * 86)
    print("DIAGNOSTIC THRESHOLD SWEEP — VALIDATION ONLY")
    print("=" * 86)
    print(
        f"range={args.threshold_min:.3f}..{args.threshold_max:.3f} "
        f"step={args.threshold_step:.3f}"
    )
    print(
        "This is diagnostic only; the validation-optimal threshold "
        "must NOT be treated as a final production threshold."
    )
    print(
        f"best threshold by balanced accuracy={best_threshold:.3f}"
    )
    print(
        f"balanced_accuracy={best_threshold_metrics['balanced_accuracy']:.4f}  "
        f"accuracy={best_threshold_metrics['accuracy']:.4f}  "
        f"macro_f1={best_threshold_metrics['macro_f1']:.4f}"
    )
    print(
        f"DOWN recall={best_threshold_metrics['down_recall']:.4f}  "
        f"UP recall={best_threshold_metrics['up_recall']:.4f}"
    )
    print(
        f"pred DOWN={best_threshold_metrics['pred_down_pct']:.2f}%  "
        f"pred UP={best_threshold_metrics['pred_up_pct']:.2f}%"
    )

    # Show a compact neighborhood around the best threshold plus 0.50.
    important_thresholds = sorted(
        {
            round(best_threshold + offset, 6)
            for offset in (-0.02, -0.01, 0.0, 0.01, 0.02)
            if (
                args.threshold_min
                <= best_threshold + offset
                <= args.threshold_max
            )
        }
        | {0.50}
    )

    print("\nThreshold neighborhood:")
    print(
        " threshold  accuracy  balanced_acc  macro_f1  "
        "down_recall  up_recall  pred_down%  pred_up%"
    )
    for threshold in important_thresholds:
        row = binary_metrics_at_threshold(
            labels,
            p_up,
            threshold,
        )
        print(
            f"   {threshold:0.3f}     "
            f"{row['accuracy']:0.4f}       "
            f"{row['balanced_accuracy']:0.4f}      "
            f"{row['macro_f1']:0.4f}      "
            f"{row['down_recall']:0.4f}      "
            f"{row['up_recall']:0.4f}      "
            f"{row['pred_down_pct']:7.2f}    "
            f"{row['pred_up_pct']:7.2f}"
        )

    month_values = sequences["_month"].to_numpy(dtype=object)
    monthly_rows = []

    print("\n" + "=" * 86)
    print("MONTHLY RANKING STABILITY")
    print("=" * 86)
    print(
        " month     samples    ROC_AUC    Brier     log_loss  "
        "mean_Pup_down  mean_Pup_up"
    )

    for month in sorted(set(month_values)):
        mask = month_values == month
        bundle = ranking_bundle(
            labels[mask],
            p_up[mask],
        )

        row = {
            "month": month,
            "samples": bundle["samples"],
            "roc_auc": bundle["roc_auc"],
            "brier_score": bundle["brier_score"],
            "log_loss": bundle["log_loss"],
            "mean_p_up_actual_down": (
                bundle["p_up_given_actual_down"]["mean"]
            ),
            "mean_p_up_actual_up": (
                bundle["p_up_given_actual_up"]["mean"]
            ),
        }
        monthly_rows.append(row)

        print(
            f" {month:<8} "
            f"{row['samples']:>8,}   "
            f"{row['roc_auc']:.4f}    "
            f"{row['brier_score']:.4f}    "
            f"{row['log_loss']:.4f}       "
            f"{row['mean_p_up_actual_down']:.4f}         "
            f"{row['mean_p_up_actual_up']:.4f}"
        )

    # Reuse the fixed v1.0.1 time-block helper.
    time_blocks = diag.assign_time_block(
        sequences["_minute_of_day"].to_numpy(dtype=np.int16)
    )

    time_block_order = [
        "09:30-10:30",
        "10:30-11:30",
        "11:30-12:30",
        "12:30-13:30",
        "13:30-14:30",
        "14:30-15:45",
    ]

    block_rows = []

    print("\n" + "=" * 86)
    print("REGULAR TIME-BLOCK RANKING STABILITY")
    print("=" * 86)
    print(
        " block          samples    ROC_AUC    Brier     log_loss  "
        "mean_Pup_down  mean_Pup_up"
    )

    for block in time_block_order:
        mask = time_blocks == block
        if not mask.any():
            continue

        bundle = ranking_bundle(
            labels[mask],
            p_up[mask],
        )

        row = {
            "time_block": block,
            "samples": bundle["samples"],
            "roc_auc": bundle["roc_auc"],
            "brier_score": bundle["brier_score"],
            "log_loss": bundle["log_loss"],
            "mean_p_up_actual_down": (
                bundle["p_up_given_actual_down"]["mean"]
            ),
            "mean_p_up_actual_up": (
                bundle["p_up_given_actual_up"]["mean"]
            ),
        }
        block_rows.append(row)

        print(
            f" {block:<14} "
            f"{row['samples']:>8,}   "
            f"{row['roc_auc']:.4f}    "
            f"{row['brier_score']:.4f}    "
            f"{row['log_loss']:.4f}       "
            f"{row['mean_p_up_actual_down']:.4f}         "
            f"{row['mean_p_up_actual_up']:.4f}"
        )

    calibration = fixed_probability_calibration(
        labels,
        p_up,
    )

    print("\n" + "=" * 86)
    print("FIXED PROBABILITY CALIBRATION BINS")
    print("=" * 86)
    print(
        " bin          samples   mean_pred_Pup   actual_UP_rate"
    )
    for row in calibration:
        print(
            f" [{row['bin_low']:.2f},{row['bin_high']:.2f}) "
            f"{row['samples']:>8,}      "
            f"{row['mean_predicted_p_up']:.4f}          "
            f"{row['actual_up_rate']:.4f}"
        )

    results = {
        "tool": "regular_binary_probability_diagnostics.py",
        "version": VERSION,
        "base_train_model_version": tm.TRAIN_MODEL_VERSION,
        "symbol": symbol,
        "model_file": str(args.model),
        "experiment": {
            "short_name": "R2",
            "prediction_session": "regular",
            "target_same_session_only": True,
            "label": "binary_sign",
            "source_sequence_scope": address.scope,
            "sequence_length": address.length,
            "target_horizon_minutes": address.horizon,
            "feature_set": feature_set,
            "feature_signature": expected_signature,
        },
        "validation_samples": int(len(labels)),
        "test_policy": "TEST is intentionally not loaded or evaluated.",
        "default_threshold_0_5": default_metrics,
        "probability_ranking": overall,
        "threshold_sweep": {
            "min": float(args.threshold_min),
            "max": float(args.threshold_max),
            "step": float(args.threshold_step),
            "validation_only_diagnostic": True,
            "best_threshold_by_balanced_accuracy": best_threshold,
            "best_threshold_metrics": best_threshold_metrics,
        },
        "monthly_ranking": monthly_rows,
        "time_block_ranking": block_rows,
        "calibration_bins": calibration,
    }

    if args.reports is not None:
        report_dir = args.reports / symbol
        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            report_dir
            / "regular_binary_probability_R2_v1.json"
        )
        sweep_path = (
            report_dir
            / "regular_binary_threshold_sweep_R2_v1.csv"
        )
        monthly_path = (
            report_dir
            / "regular_binary_monthly_auc_R2_v1.csv"
        )
        blocks_path = (
            report_dir
            / "regular_binary_timeblock_auc_R2_v1.csv"
        )

        json_path.write_text(
            json.dumps(
                results,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        sweep.to_csv(
            sweep_path,
            index=False,
        )
        pd.DataFrame(monthly_rows).to_csv(
            monthly_path,
            index=False,
        )
        pd.DataFrame(block_rows).to_csv(
            blocks_path,
            index=False,
        )

        print("\nReports written:")
        print(f"  {json_path}")
        print(f"  {sweep_path}")
        print(f"  {monthly_path}")
        print(f"  {blocks_path}")

    print("\nR2 PROBABILITY/AUC DIAGNOSTICS PASS")
    print("TEST was not loaded or evaluated.")


if __name__ == "__main__":
    main()
