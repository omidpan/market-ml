#!/usr/bin/env python3
"""
session_diagnostics_fast.py
Version 1.1.1

Validation-only session diagnostics for an already-trained market_ml model.

Key difference from v1.0:
- Does NOT load the full 81-month model matrix.
- Reads only VALIDATION sequence metadata.
- Reads only VALIDATION ATR-policy labels.
- Loads only matrix month partitions required by those validation windows.
- Does NOT train.
- Does NOT evaluate TEST.

Groups reported:
- overall validation
- prediction session: premarket / regular / aftermarket
- target stays in same session vs crosses session boundary
- session x target-scope combinations
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


VERSION = "1.1.1"


def import_trainer(path: Path):
    spec = importlib.util.spec_from_file_location(
        "market_train_model",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    version = getattr(module, "TRAIN_MODEL_VERSION", None)
    if not version:
        raise RuntimeError("train_model.py missing TRAIN_MODEL_VERSION.")

    print(f"train_model.py={version}")
    return module


def normalize_source(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def to_ns(series: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(
            series,
            utc=True,
            errors="raise",
        )
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False)
    )


def month_keys_from_timestamps(series: pd.Series) -> set[tuple[int, int]]:
    ts = pd.to_datetime(series, errors="raise")

    # Preserve the timestamps' own timezone when present. Matrix partitions
    # follow the source/trading calendar, not a UTC calendar conversion.
    return {
        (int(year), int(month))
        for year, month in zip(
            ts.dt.year.to_numpy(),
            ts.dt.month.to_numpy(),
        )
    }


def monthly_part_map(root: Path) -> dict[tuple[int, int], Path]:
    result = {}

    for path in sorted(
        root.glob("year=*/month=*/part.parquet")
    ):
        try:
            year = int(
                path.parent.parent.name.split("=", 1)[1]
            )
            month = int(
                path.parent.name.split("=", 1)[1]
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid monthly partition path: {path}"
            ) from exc

        key = (year, month)

        if key in result:
            raise ValueError(
                f"Duplicate monthly partition {key}: "
                f"{result[key]} and {path}"
            )

        result[key] = path

    if not result:
        raise FileNotFoundError(
            f"No monthly Parquet parts under {root}"
        )

    return result


def load_validation_sequences(
    sequence_symbol_root: Path,
    expected_signature: str,
) -> pd.DataFrame:
    columns = [
        "feature_signature",
        "source_id",
        "window_start_datetime",
        "window_end_datetime",
        "prediction_time",
        "split",
        "session",
        "target_same_session",
    ]

    print(
        "Reading VALIDATION sequence metadata only "
        "(dataset predicate: split == validation)..."
    )

    frame = pd.read_parquet(
        sequence_symbol_root,
        engine="pyarrow",
        columns=columns,
        filters=[("split", "==", "validation")],
    )

    if frame.empty:
        raise ValueError(
            f"No validation sequences under {sequence_symbol_root}"
        )

    signatures = set(
        frame["feature_signature"].astype(str).unique()
    )

    if signatures != {expected_signature}:
        raise ValueError(
            "Validation sequence feature signature mismatch."
        )

    frame["source_id"] = normalize_source(
        frame["source_id"]
    )
    frame["split"] = (
        frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    frame["session"] = (
        frame["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    frame["_start_ns"] = to_ns(
        frame["window_start_datetime"]
    )
    frame["_end_ns"] = to_ns(
        frame["window_end_datetime"]
    )
    frame["_prediction_ns"] = to_ns(
        frame["prediction_time"]
    )

    expected_prediction_ns = (
        frame["_end_ns"].to_numpy(dtype=np.int64)
        + 60 * 1_000_000_000
    )

    if not np.array_equal(
        frame["_prediction_ns"].to_numpy(dtype=np.int64),
        expected_prediction_ns,
    ):
        raise ValueError(
            "Sequence timing mismatch: expected prediction_time "
            "= window_end_datetime + 1 minute."
        )

    expected_span = (
        119 * 60 * 1_000_000_000
    )

    if (
        frame["_end_ns"].to_numpy(dtype=np.int64)
        - frame["_start_ns"].to_numpy(dtype=np.int64)
        != expected_span
    ).any():
        raise ValueError(
            "Expected L120 sequence timestamp span of 119 minutes."
        )

    if frame.duplicated(
        ["source_id", "_prediction_ns"]
    ).any():
        raise ValueError(
            "Duplicate validation sequence endpoint keys."
        )

    if not frame["split"].eq("validation").all():
        raise ValueError(
            "Non-validation row escaped validation filter."
        )

    print(
        f"Validation sequences: {len(frame):,}"
    )

    return frame.reset_index(drop=True)


def validate_policy_manifest(
    tm,
    root: Path,
    address,
    feature_set: str,
    cfg,
    symbol: str,
) -> tuple[dict, str]:
    manifest_path = root / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Label-policy manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    metadata = manifest.get(
        "policy_metadata",
        {},
    ) or {}

    signature = str(
        manifest.get("policy_signature", "")
    ).strip()

    expected = {
        "policy_name": cfg.label_policy.name,
        "symbol": symbol,
        "target_horizon_minutes": address.horizon,
        "sequence_length": address.length,
        "feature_set": feature_set,
        "scope": address.scope,
        "stride_minutes": address.stride,
        "sessions": list(address.sessions),
    }

    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Label-policy manifest mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )

    atr = metadata.get("atr", {}) or {}
    threshold = metadata.get("threshold", {}) or {}

    if int(
        atr.get("period", -1)
    ) != cfg.label_policy.atr_period:
        raise ValueError(
            "Label-policy ATR period mismatch."
        )

    multiplier = float(
        threshold.get(
            "multiplier",
            float("nan"),
        )
    )

    if not math.isclose(
        multiplier,
        cfg.label_policy.multiplier,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Label-policy multiplier mismatch."
        )

    if not signature:
        raise ValueError(
            "Label-policy manifest missing policy_signature."
        )

    return manifest, signature


def load_validation_labels(
    tm,
    label_policy_base: Path,
    sequences: pd.DataFrame,
    address,
    feature_set: str,
    cfg,
    symbol: str,
):
    root = tm.label_policy_symbol_root(
        label_policy_base,
        address,
        cfg,
        symbol,
    )

    print(f"Label policy root: {root}")

    _, policy_signature = validate_policy_manifest(
        tm,
        root,
        address,
        feature_set,
        cfg,
        symbol,
    )

    columns = [
        "prediction_time",
        "source_id",
        "split",
        "label_direction",
        "policy_name",
        "policy_signature",
    ]

    print(
        "Reading only label-policy month partitions needed by VALIDATION..."
    )

    available = monthly_part_map(root)

    required_months = month_keys_from_timestamps(
        sequences["prediction_time"]
    )

    missing = sorted(
        required_months - set(available)
    )

    if missing:
        raise FileNotFoundError(
            f"Required label-policy months missing: {missing}"
        )

    selected = [
        available[key]
        for key in sorted(required_months)
    ]

    print(
        f"Label-policy months required for validation: {len(selected)}"
    )

    print(
        "  "
        + ", ".join(
            f"{year:04d}-{month:02d}"
            for year, month in sorted(required_months)
        )
    )

    policy_frames = []

    for i, parquet_path in enumerate(selected, 1):
        frame = pd.read_parquet(
            parquet_path,
            engine="pyarrow",
            columns=columns,
            filters=[("split", "==", "validation")],
        )

        if not frame.empty:
            policy_frames.append(frame)

        print(
            f"  label-policy validation months: "
            f"{i:>2}/{len(selected)} "
            f"{parquet_path.parent.parent.name}/"
            f"{parquet_path.parent.name}"
        )

    if not policy_frames:
        raise ValueError(
            "No validation policy rows."
        )

    policy = pd.concat(
        policy_frames,
        ignore_index=True,
    )

    policy["source_id"] = normalize_source(
        policy["source_id"]
    )
    policy["split"] = (
        policy["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if set(
        policy["policy_name"].astype(str).unique()
    ) != {cfg.label_policy.name}:
        raise ValueError(
            "Policy name mismatch in validation policy rows."
        )

    if set(
        policy["policy_signature"].astype(str).unique()
    ) != {policy_signature}:
        raise ValueError(
            "Policy signature mismatch in validation policy rows."
        )

    policy["_prediction_ns"] = to_ns(
        policy["prediction_time"]
    )

    if policy.duplicated(
        ["source_id", "_prediction_ns"]
    ).any():
        raise ValueError(
            "Duplicate validation policy endpoint keys."
        )

    sequence_keys = sequences[
        ["source_id", "_prediction_ns", "split"]
    ].copy()

    sequence_keys["_row"] = np.arange(
        len(sequence_keys),
        dtype=np.int64,
    )

    aligned = sequence_keys.merge(
        policy[
            [
                "source_id",
                "_prediction_ns",
                "split",
                "label_direction",
            ]
        ].rename(
            columns={"split": "_policy_split"}
        ),
        on=["source_id", "_prediction_ns"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row")

    if aligned["label_direction"].isna().any():
        raise ValueError(
            "Missing validation ATR label after endpoint alignment."
        )

    if (
        aligned["split"]
        != aligned["_policy_split"]
    ).any():
        raise ValueError(
            "Validation policy split mismatch."
        )

    raw = pd.to_numeric(
        aligned["label_direction"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if not np.isin(
        raw,
        [-1, 0, 1],
    ).all():
        raise ValueError(
            "Expected label_direction in {-1,0,+1}."
        )

    labels = raw + 1

    print(
        "Validation label alignment: PASS "
        f"({len(labels):,} rows, "
        f"signature={policy_signature[:12]}...)"
    )

    return (
        labels,
        ("down", "neutral", "up"),
        {
            "root": str(root),
            "policy_signature": policy_signature,
            "policy_name": cfg.label_policy.name,
            "atr_period": cfg.label_policy.atr_period,
            "multiplier": cfg.label_policy.multiplier,
            "validation_rows": int(len(labels)),
        },
    )


def load_required_matrix(
    row_symbol_root: Path,
    sequences: pd.DataFrame,
    features: tuple[str, ...],
    expected_signature: str,
):
    available = monthly_part_map(
        row_symbol_root
    )

    required_months = (
        month_keys_from_timestamps(
            sequences["window_start_datetime"]
        )
        | month_keys_from_timestamps(
            sequences["window_end_datetime"]
        )
    )

    missing = sorted(
        required_months - set(available)
    )

    if missing:
        raise FileNotFoundError(
            f"Required matrix months missing: {missing}"
        )

    selected = [
        available[key]
        for key in sorted(required_months)
    ]

    print(
        f"Matrix months required for validation windows: "
        f"{len(selected)}"
    )
    print(
        "  "
        + ", ".join(
            f"{year:04d}-{month:02d}"
            for year, month in sorted(required_months)
        )
    )

    columns = [
        "datetime",
        "source_id",
        "feature_signature",
        *features,
    ]

    xs = []
    times = []
    sources = []

    for i, path in enumerate(
        selected,
        1,
    ):
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=columns,
        )

        signatures = set(
            frame["feature_signature"]
            .astype(str)
            .unique()
        )

        if signatures != {expected_signature}:
            raise ValueError(
                f"Feature signature mismatch: {path}"
            )

        x = frame[
            list(features)
        ].to_numpy(
            dtype=np.float32,
            na_value=np.nan,
        )

        if not np.isfinite(x).all():
            raise ValueError(
                f"NaN/inf in scaled matrix: {path}"
            )

        xs.append(x)
        times.append(
            to_ns(frame["datetime"])
        )
        sources.append(
            normalize_source(
                frame["source_id"]
            ).to_numpy(dtype=object)
        )

        print(
            f"  matrix validation months: "
            f"{i:>2}/{len(selected)} "
            f"{path.parent.parent.name}/"
            f"{path.parent.name}"
        )

    matrix = np.concatenate(xs)
    time_ns = np.concatenate(times)
    source_names = np.concatenate(sources)

    names = sorted(
        set(source_names)
    )

    code_map = {
        name: i
        for i, name in enumerate(names)
    }

    source_code = np.array(
        [
            code_map[value]
            for value in source_names
        ],
        dtype=np.int32,
    )

    order = np.lexsort(
        (
            time_ns,
            source_code,
        )
    )

    matrix = np.ascontiguousarray(
        matrix[order],
        dtype=np.float32,
    )
    time_ns = np.ascontiguousarray(
        time_ns[order],
        dtype=np.int64,
    )
    source_code = np.ascontiguousarray(
        source_code[order],
        dtype=np.int32,
    )

    print(
        f"Validation matrix subset: "
        f"{len(matrix):,} rows x {matrix.shape[1]} features "
        f"(~{matrix.nbytes / 2**20:.1f} MiB)"
    )

    return (
        matrix,
        time_ns,
        source_code,
        code_map,
    )


def group_metrics(
    tm,
    truth: np.ndarray,
    pred: np.ndarray,
    class_names,
):
    metrics = tm.validation_metrics(
        truth,
        pred,
        class_names,
    )

    metrics["samples"] = int(
        len(truth)
    )

    metrics["actual_distribution"] = {}
    metrics["predicted_distribution"] = {}

    for i, name in enumerate(
        class_names
    ):
        actual_count = int(
            np.sum(truth == i)
        )
        predicted_count = int(
            np.sum(pred == i)
        )

        metrics["actual_distribution"][name] = {
            "count": actual_count,
            "pct": (
                100.0
                * actual_count
                / len(truth)
            ),
        }

        metrics["predicted_distribution"][name] = {
            "count": predicted_count,
            "pct": (
                100.0
                * predicted_count
                / len(pred)
            ),
        }

    return metrics


def compact_row(
    group_type: str,
    group_name: str,
    metrics: dict,
    class_names,
):
    row = {
        "group_type": group_type,
        "group_name": group_name,
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }

    for name in class_names:
        row[f"{name}_recall"] = (
            metrics["per_class"][name]["recall"]
        )
        row[f"{name}_precision"] = (
            metrics["per_class"][name]["precision"]
        )
        row[f"{name}_f1"] = (
            metrics["per_class"][name]["f1"]
        )
        row[f"pred_{name}_pct"] = (
            metrics[
                "predicted_distribution"
            ][name]["pct"]
        )

    return row


def print_group(
    title: str,
    metrics: dict,
    class_names,
):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)

    print(
        f"samples={metrics['samples']:,}  "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    print("\nPer-class:")
    for name in class_names:
        p = metrics["per_class"][name]

        print(
            f"  {name:<8} "
            f"precision={p['precision']:.4f} "
            f"recall={p['recall']:.4f} "
            f"f1={p['f1']:.4f} "
            f"support={p['support']:,}"
        )

    print("\nPredicted distribution:")
    for name in class_names:
        d = metrics[
            "predicted_distribution"
        ][name]

        print(
            f"  {name:<8} "
            f"{d['count']:>9,} "
            f"({d['pct']:7.3f}%)"
        )

    cm = np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )

    print(
        "\nConfusion matrix "
        "(rows=actual, cols=predicted):"
    )

    print(
        " " * 12
        + " ".join(
            f"{name:>10}"
            for name in class_names
        )
    )

    for i, name in enumerate(
        class_names
    ):
        print(
            f"{name:<12}"
            + " ".join(
                f"{int(value):>10,}"
                for value in cm[i]
            )
        )


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
        "--label-policy-root",
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
        required=True,
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

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    tm = import_trainer(
        project_root
        / "src"
        / "train_model.py"
    )

    raw = tm.load_yaml(
        args.config
    )

    (
        address,
        feature_set,
        features,
        cfg,
    ) = tm.resolve_config(raw)

    if address.length != 120:
        raise ValueError(
            "This diagnostic currently expects L120."
        )

    if cfg.label_mode != "label_policy_3class":
        raise ValueError(
            "Expected label_mode=label_policy_3class."
        )

    symbol = (
        args.symbol
        .strip()
        .lower()
    )

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

    scaler_path = (
        matrix_root
        / "scaler.json"
    )

    scaler = json.loads(
        scaler_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        scaler.get("feature_signature")
        != expected_signature
    ):
        raise ValueError(
            "scaler.json feature signature mismatch."
        )

    if tuple(
        scaler.get("feature_names", ())
    ) != features:
        raise ValueError(
            "scaler.json feature order mismatch."
        )

    print(
        "\n[1/5] Loading VALIDATION sequences only..."
    )

    sequences = load_validation_sequences(
        matrix_root
        / "sequences"
        / f"symbol={symbol}",
        expected_signature,
    )

    print(
        "\n[2/5] Loading VALIDATION ATR labels only..."
    )

    (
        labels,
        class_names,
        policy_info,
    ) = load_validation_labels(
        tm,
        args.label_policy_root,
        sequences,
        address,
        feature_set,
        cfg,
        symbol,
    )

    print(
        "\n[3/5] Loading only matrix months needed "
        "by VALIDATION..."
    )

    (
        matrix,
        time_ns,
        source_code,
        code_map,
    ) = load_required_matrix(
        matrix_root
        / "rows"
        / f"symbol={symbol}",
        sequences,
        features,
        expected_signature,
    )

    print(
        "\n[4/5] Mapping exact VALIDATION L120 windows..."
    )

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

    print(
        "\n[5/5] Predicting VALIDATION only..."
    )

    keras, tf = tm.import_ml_stack()

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        f"TensorFlow={tf.__version__} "
        f"Keras={keras.__version__} "
        f"GPU={[device.name for device in gpus]}"
    )

    model = keras.models.load_model(
        args.model
    )

    batch_size = (
        args.batch_size
        or cfg.batch_size
    )

    WindowDataset = (
        tm.make_window_dataset_class(
            keras
        )
    )

    dataset = WindowDataset(
        matrix,
        starts,
        labels,
        address.length,
        batch_size,
        False,
        cfg.seed + 1000,
        1,
    )

    logits = model.predict(
        dataset,
        verbose=1,
    )

    predicted = np.argmax(
        logits,
        axis=1,
    ).astype(np.int64)

    if len(predicted) != len(labels):
        raise ValueError(
            "Prediction/label row-count mismatch."
        )

    results = {
        "tool": "session_diagnostics_fast.py",
        "version": VERSION,
        "trainer_version": tm.TRAIN_MODEL_VERSION,
        "symbol": symbol,
        "model_file": str(args.model),
        "feature_set": feature_set,
        "feature_signature": expected_signature,
        "sequence_length": address.length,
        "target_horizon_minutes": address.horizon,
        "label_policy": policy_info,
        "validation_samples": int(
            len(labels)
        ),
        "test_policy": (
            "TEST intentionally not loaded or evaluated."
        ),
        "groups": {},
    }

    summary_rows = []

    overall = group_metrics(
        tm,
        labels,
        predicted,
        class_names,
    )

    results["groups"]["overall"] = (
        overall
    )

    print_group(
        "VALIDATION — OVERALL",
        overall,
        class_names,
    )

    summary_rows.append(
        compact_row(
            "overall",
            "overall",
            overall,
            class_names,
        )
    )

    sessions = (
        sequences["session"]
        .to_numpy(dtype=object)
    )

    same_session = (
        sequences["target_same_session"]
        .astype(bool)
        .to_numpy()
    )

    for session_name in (
        "premarket",
        "regular",
        "aftermarket",
    ):
        mask = (
            sessions == session_name
        )

        if not mask.any():
            continue

        metrics = group_metrics(
            tm,
            labels[mask],
            predicted[mask],
            class_names,
        )

        key = (
            f"session:{session_name}"
        )

        results["groups"][key] = (
            metrics
        )

        print_group(
            "VALIDATION — "
            f"{session_name.upper()}",
            metrics,
            class_names,
        )

        summary_rows.append(
            compact_row(
                "session",
                session_name,
                metrics,
                class_names,
            )
        )

    for value, name in (
        (True, "target_same_session"),
        (False, "target_crosses_session"),
    ):
        mask = (
            same_session == value
        )

        if not mask.any():
            continue

        metrics = group_metrics(
            tm,
            labels[mask],
            predicted[mask],
            class_names,
        )

        results["groups"][name] = (
            metrics
        )

        print_group(
            "VALIDATION — "
            f"{name.upper()}",
            metrics,
            class_names,
        )

        summary_rows.append(
            compact_row(
                "target_scope",
                name,
                metrics,
                class_names,
            )
        )

    for session_name in (
        "premarket",
        "regular",
        "aftermarket",
    ):
        for same_value, suffix in (
            (True, "same"),
            (False, "cross"),
        ):
            mask = (
                (sessions == session_name)
                & (
                    same_session
                    == same_value
                )
            )

            if not mask.any():
                continue

            metrics = group_metrics(
                tm,
                labels[mask],
                predicted[mask],
                class_names,
            )

            key = (
                "session_target:"
                f"{session_name}:"
                f"{suffix}"
            )

            results["groups"][key] = (
                metrics
            )

            print_group(
                "VALIDATION — "
                f"{session_name.upper()} / "
                f"TARGET "
                f"{'SAME SESSION' if same_value else 'CROSSES SESSION'}",
                metrics,
                class_names,
            )

            summary_rows.append(
                compact_row(
                    "session_x_target_scope",
                    f"{session_name}_{suffix}",
                    metrics,
                    class_names,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )

    compact_columns = [
        "group_type",
        "group_name",
        "samples",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "down_recall",
        "neutral_recall",
        "up_recall",
        "pred_down_pct",
        "pred_neutral_pct",
        "pred_up_pct",
    ]

    print(
        "\n" + "=" * 76
    )
    print(
        "COMPACT COMPARISON"
    )
    print(
        "=" * 76
    )

    print(
        summary[
            compact_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    if args.reports is not None:
        report_dir = (
            args.reports
            / symbol
        )

        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_id = (
            args.model
            .parent
            .name
        )

        if run_id.startswith(
            "run_id="
        ):
            run_id = run_id[
                len("run_id="):
            ]

        json_path = (
            report_dir
            / (
                "session_diagnostics_"
                f"{run_id}.json"
            )
        )

        csv_path = (
            report_dir
            / (
                "session_diagnostics_"
                f"{run_id}.csv"
            )
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

        summary.to_csv(
            csv_path,
            index=False,
        )

        print(
            "\nReports written:"
        )
        print(
            f"  {json_path}"
        )
        print(
            f"  {csv_path}"
        )

    print(
        "\nSESSION DIAGNOSTICS PASS"
    )
    print(
        "TEST was not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
