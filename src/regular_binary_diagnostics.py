#!/usr/bin/env python3
"""
regular_binary_diagnostics.py — REGULAR binary validation diagnostics.

Version 1.0.0

Purpose
-------
Evaluate the already-trained REGULAR / same-session-target / binary-sign model
on VALIDATION only, then break the predictions down by:

1) validation month
2) regular-session time block

No training. No TEST evaluation. Existing Parquet/model artifacts are read-only.

Expected source semantics
-------------------------
- Existing continuous L120 model-ready sequences are reused.
- Prediction endpoint must be REGULAR.
- H15 target must remain in the same session.
- Exact-zero forward returns are excluded.
- Binary labels: DOWN < 0, UP > 0.
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


VERSION = "1.0.1"
NY_TZ = "America/New_York"


def import_train_model(project_root: Path):
    path = project_root / "src" / "train_model.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing base trainer: {path}")

    spec = importlib.util.spec_from_file_location(
        "market_ml_base_train_model_diag",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    version = str(getattr(module, "TRAIN_MODEL_VERSION", ""))
    if not version.startswith("1.1."):
        raise RuntimeError(
            "regular_binary_diagnostics.py expects train_model.py v1.1.x; "
            f"found {version!r}."
        )

    return module


def to_ns(series: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(series, utc=True, errors="raise")
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False)
    )


def to_ny(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="raise")
    if ts.dt.tz is None:
        return ts.dt.tz_localize(
            NY_TZ,
            ambiguous="raise",
            nonexistent="raise",
        )
    return ts.dt.tz_convert(NY_TZ)


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
        raise ValueError(
            f"Unrecognized target_same_session values: {unknown}"
        )
    return values.map(mapping).to_numpy(dtype=bool)


def month_keys_from_timestamps(series: pd.Series) -> set[tuple[int, int]]:
    local = to_ny(series)
    return {
        (int(y), int(m))
        for y, m in zip(
            local.dt.year.to_numpy(),
            local.dt.month.to_numpy(),
        )
    }


def monthly_part_map(root: Path) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}

    for path in sorted(root.glob("year=*/month=*/part.parquet")):
        try:
            year = int(path.parent.parent.name.split("=", 1)[1])
            month = int(path.parent.name.split("=", 1)[1])
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
            f"No monthly part.parquet files under {root}"
        )

    return result


def load_validation_sequences(
    sequence_symbol_root: Path,
    expected_signature: str,
    expected_length: int,
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
        "target_return_bps",
    ]

    print(
        "Reading VALIDATION sequence rows only "
        "(split == validation)..."
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

    signatures = set(frame["feature_signature"].astype(str).unique())
    if signatures != {expected_signature}:
        raise ValueError(
            f"Feature signature mismatch in validation sequences: {signatures}"
        )

    frame["source_id"] = (
        frame["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
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

    same = normalize_bool(frame["target_same_session"])
    returns = pd.to_numeric(
        frame["target_return_bps"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(returns).all():
        raise ValueError(
            "Non-finite target_return_bps in validation sequences."
        )

    keep = (
        frame["session"].eq("regular").to_numpy(dtype=bool)
        & same
        & (returns != 0.0)
    )

    frame = frame.loc[keep].copy().reset_index(drop=True)
    returns = returns[keep]

    if frame.empty:
        raise ValueError(
            "REGULAR + same-target + nonzero-return selected zero rows."
        )

    frame["_start_ns"] = to_ns(frame["window_start_datetime"])
    frame["_end_ns"] = to_ns(frame["window_end_datetime"])
    frame["_prediction_ns"] = to_ns(frame["prediction_time"])

    expected_prediction_ns = (
        frame["_end_ns"].to_numpy(dtype=np.int64)
        + 60 * 1_000_000_000
    )
    if not np.array_equal(
        frame["_prediction_ns"].to_numpy(dtype=np.int64),
        expected_prediction_ns,
    ):
        raise ValueError(
            "Expected prediction_time = window_end_datetime + 1 minute."
        )

    expected_span_ns = (
        (expected_length - 1)
        * 60
        * 1_000_000_000
    )
    actual_span_ns = (
        frame["_end_ns"].to_numpy(dtype=np.int64)
        - frame["_start_ns"].to_numpy(dtype=np.int64)
    )
    if not np.all(actual_span_ns == expected_span_ns):
        raise ValueError(
            f"Expected L{expected_length} timestamp span "
            f"of {expected_length - 1} minutes."
        )

    if frame.duplicated(
        ["source_id", "_prediction_ns"]
    ).any():
        raise ValueError(
            "Duplicate validation endpoint keys."
        )

    labels = np.zeros(len(frame), dtype=np.int64)
    labels[returns > 0.0] = 1
    frame["_label"] = labels

    local = to_ny(frame["prediction_time"])
    frame["_month"] = local.dt.strftime("%Y-%m")

    minute_of_day = (
        local.dt.hour.to_numpy(dtype=np.int16) * 60
        + local.dt.minute.to_numpy(dtype=np.int16)
    )
    frame["_minute_of_day"] = minute_of_day

    print(
        "REGULAR same-target binary validation sequences: "
        f"{len(frame):,}"
    )

    return frame


def load_required_matrix(
    row_symbol_root: Path,
    sequences: pd.DataFrame,
    features: tuple[str, ...],
    expected_signature: str,
):
    available = monthly_part_map(row_symbol_root)

    required_months = (
        month_keys_from_timestamps(
            sequences["window_start_datetime"]
        )
        | month_keys_from_timestamps(
            sequences["window_end_datetime"]
        )
    )

    missing = sorted(required_months - set(available))
    if missing:
        raise FileNotFoundError(
            f"Required matrix months missing: {missing}"
        )

    selected = [
        available[key]
        for key in sorted(required_months)
    ]

    print(
        f"Matrix months required for validation: {len(selected)}"
    )
    print(
        "  "
        + ", ".join(
            f"{y:04d}-{m:02d}"
            for y, m in sorted(required_months)
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

    for i, path in enumerate(selected, 1):
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

        x = frame[list(features)].to_numpy(
            dtype=np.float32,
            na_value=np.nan,
        )
        if not np.isfinite(x).all():
            raise ValueError(
                f"NaN/inf in scaled matrix: {path}"
            )

        xs.append(x)
        times.append(to_ns(frame["datetime"]))
        sources.append(
            frame["source_id"]
            .astype(str)
            .str.strip()
            .str.lower()
            .to_numpy(dtype=object)
        )

        print(
            f"  matrix validation months: "
            f"{i:>2}/{len(selected)} "
            f"{path.parent.parent.name}/{path.parent.name}"
        )

    matrix = np.concatenate(xs)
    time_ns = np.concatenate(times)
    source_names = np.concatenate(sources)

    names = sorted(set(source_names))
    code_map = {
        name: i
        for i, name in enumerate(names)
    }
    source_code = np.array(
        [code_map[value] for value in source_names],
        dtype=np.int32,
    )

    order = np.lexsort((time_ns, source_code))

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

    return matrix, time_ns, source_code, code_map


def metric_bundle(
    tm,
    truth: np.ndarray,
    pred: np.ndarray,
):
    class_names = ("down", "up")
    metrics = tm.validation_metrics(
        truth,
        pred,
        class_names,
    )

    metrics["samples"] = int(len(truth))
    metrics["actual_distribution"] = {}
    metrics["predicted_distribution"] = {}

    for i, name in enumerate(class_names):
        actual = int(np.sum(truth == i))
        predicted = int(np.sum(pred == i))

        metrics["actual_distribution"][name] = {
            "count": actual,
            "pct": 100.0 * actual / len(truth),
        }
        metrics["predicted_distribution"][name] = {
            "count": predicted,
            "pct": 100.0 * predicted / len(pred),
        }

    return metrics


def compact_row(
    group_type: str,
    group_name: str,
    metrics: dict,
):
    return {
        "group_type": group_type,
        "group_name": group_name,
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "down_actual_pct": metrics["actual_distribution"]["down"]["pct"],
        "up_actual_pct": metrics["actual_distribution"]["up"]["pct"],
        "down_recall": metrics["per_class"]["down"]["recall"],
        "up_recall": metrics["per_class"]["up"]["recall"],
        "down_precision": metrics["per_class"]["down"]["precision"],
        "up_precision": metrics["per_class"]["up"]["precision"],
        "pred_down_pct": metrics["predicted_distribution"]["down"]["pct"],
        "pred_up_pct": metrics["predicted_distribution"]["up"]["pct"],
    }


def print_group(
    title: str,
    metrics: dict,
):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"samples={metrics['samples']:,}  "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(
        "actual: "
        f"DOWN={metrics['actual_distribution']['down']['pct']:.2f}% "
        f"UP={metrics['actual_distribution']['up']['pct']:.2f}%"
    )
    print(
        "predicted: "
        f"DOWN={metrics['predicted_distribution']['down']['pct']:.2f}% "
        f"UP={metrics['predicted_distribution']['up']['pct']:.2f}%"
    )
    print(
        "recall: "
        f"DOWN={metrics['per_class']['down']['recall']:.4f} "
        f"UP={metrics['per_class']['up']['recall']:.4f}"
    )

    cm = np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )
    print("confusion matrix (rows=actual, cols=predicted):")
    print("                 down         up")
    print(f"down       {cm[0,0]:>10,} {cm[0,1]:>10,}")
    print(f"up         {cm[1,0]:>10,} {cm[1,1]:>10,}")


def assign_time_block(minute_of_day: np.ndarray) -> np.ndarray:
    labels = np.full(
        len(minute_of_day),
        "",
        dtype=object,
    )

    blocks = [
        (570, 630, "09:30-10:30"),
        (630, 690, "10:30-11:30"),
        (690, 750, "11:30-12:30"),
        (750, 810, "12:30-13:30"),
        (810, 870, "13:30-14:30"),
        (870, 946, "14:30-15:45"),
    ]

    for start, end, name in blocks:
        mask = (
            (minute_of_day >= start)
            & (minute_of_day < end)
        )
        labels[mask] = name

    if np.any(labels == ""):
        bad = minute_of_day[labels == ""]
        values, counts = np.unique(bad, return_counts=True)
        sample = ", ".join(
            f"{int(v)//60:02d}:{int(v)%60:02d}({int(c)})"
            for v, c in zip(values[:20], counts[:20])
        )
        raise ValueError(
            "REGULAR same-target binary validation contains prediction "
            f"times outside expected 09:30-15:45 blocks: {sample}"
        )

    return labels


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

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    tm = import_train_model(project_root)

    raw = tm.load_yaml(args.config)
    address, feature_set, features, cfg = tm.resolve_config(raw)

    if address.length != 120:
        raise ValueError(
            f"Expected L120, found L{address.length}."
        )
    if address.horizon != 15:
        raise ValueError(
            f"Expected H15, found H{address.horizon}."
        )

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

    scaler_path = (
        matrix_root
        / "scaler.json"
    )
    scaler = json.loads(
        scaler_path.read_text(
            encoding="utf-8"
        )
    )

    if scaler.get(
        "feature_signature"
    ) != expected_signature:
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
        f"regular_binary_diagnostics.py={VERSION} "
        f"base_train_model.py={tm.TRAIN_MODEL_VERSION}"
    )
    print(
        "VALIDATION ONLY | session=regular | "
        "same_target=True | binary_sign"
    )
    print("TEST will not be loaded or evaluated.")

    print(
        "\n[1/4] Loading REGULAR validation sequences only..."
    )

    sequences = load_validation_sequences(
        matrix_root
        / "sequences"
        / f"symbol={symbol}",
        expected_signature,
        address.length,
    )

    labels = sequences[
        "_label"
    ].to_numpy(dtype=np.int64)

    print(
        "\n[2/4] Loading matrix months needed by VALIDATION..."
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
        "\n[3/4] Mapping exact validation L120 windows..."
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
        "\n[4/4] Loading saved model and predicting VALIDATION..."
    )

    keras, tf = tm.import_ml_stack()

    gpus = tf.config.list_physical_devices(
        "GPU"
    )
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

    model = keras.models.load_model(
        args.model
    )

    WindowDataset = tm.make_window_dataset_class(
        keras
    )

    batch_size = (
        args.batch_size
        or cfg.batch_size
    )

    dataset = WindowDataset(
        matrix,
        starts,
        labels,
        address.length,
        batch_size,
        False,
        cfg.seed + 2000,
        1,
    )

    logits = model.predict(
        dataset,
        verbose=1,
    )
    prediction = np.argmax(
        logits,
        axis=1,
    ).astype(np.int64)

    if len(prediction) != len(labels):
        raise ValueError(
            "Prediction/label length mismatch."
        )

    results = {
        "tool": "regular_binary_diagnostics.py",
        "version": VERSION,
        "base_train_model_version": tm.TRAIN_MODEL_VERSION,
        "symbol": symbol,
        "model_file": str(args.model),
        "feature_set": feature_set,
        "feature_signature": expected_signature,
        "sequence_length": address.length,
        "target_horizon_minutes": address.horizon,
        "regime": {
            "prediction_session": "regular",
            "target_same_session_only": True,
            "label": "binary_sign",
            "source_sequence_scope": address.scope,
        },
        "validation_samples": int(len(labels)),
        "test_policy": "TEST is intentionally not loaded or evaluated.",
        "groups": {},
    }

    summary_rows = []

    overall = metric_bundle(
        tm,
        labels,
        prediction,
    )
    results["groups"]["overall"] = overall
    print_group(
        "VALIDATION — REGULAR BINARY OVERALL",
        overall,
    )
    summary_rows.append(
        compact_row(
            "overall",
            "overall",
            overall,
        )
    )

    print(
        "\n\n### MONTH-BY-MONTH VALIDATION"
    )

    month_values = sequences[
        "_month"
    ].to_numpy(dtype=object)

    for month in sorted(
        set(month_values)
    ):
        mask = (
            month_values == month
        )
        metrics = metric_bundle(
            tm,
            labels[mask],
            prediction[mask],
        )
        results["groups"][
            f"month:{month}"
        ] = metrics

        print_group(
            f"VALIDATION MONTH — {month}",
            metrics,
        )
        summary_rows.append(
            compact_row(
                "month",
                month,
                metrics,
            )
        )

    print(
        "\n\n### REGULAR TIME-BLOCK VALIDATION"
    )

    time_blocks = assign_time_block(
        sequences[
            "_minute_of_day"
        ].to_numpy(dtype=np.int16)
    )

    block_order = [
        "09:30-10:30",
        "10:30-11:30",
        "11:30-12:30",
        "12:30-13:30",
        "13:30-14:30",
        "14:30-15:45",
    ]

    for block in block_order:
        mask = (
            time_blocks == block
        )
        if not mask.any():
            continue

        metrics = metric_bundle(
            tm,
            labels[mask],
            prediction[mask],
        )
        results["groups"][
            f"time_block:{block}"
        ] = metrics

        print_group(
            f"REGULAR TIME BLOCK — {block} ET",
            metrics,
        )
        summary_rows.append(
            compact_row(
                "time_block",
                block,
                metrics,
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
        "down_actual_pct",
        "up_actual_pct",
        "down_recall",
        "up_recall",
        "pred_down_pct",
        "pred_up_pct",
    ]

    print(
        "\n" + "=" * 100
    )
    print(
        "COMPACT COMPARISON"
    )
    print(
        "=" * 100
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

    month_summary = summary[
        summary["group_type"].eq("month")
    ].copy()

    if not month_summary.empty:
        ba = month_summary[
            "balanced_accuracy"
        ].to_numpy(dtype=np.float64)

        stability = {
            "month_count": int(len(ba)),
            "mean_monthly_balanced_accuracy_unweighted": float(
                np.mean(ba)
            ),
            "median_monthly_balanced_accuracy": float(
                np.median(ba)
            ),
            "min_monthly_balanced_accuracy": float(
                np.min(ba)
            ),
            "max_monthly_balanced_accuracy": float(
                np.max(ba)
            ),
            "months_at_or_above_0_50": int(
                np.sum(ba >= 0.50)
            ),
            "months_at_or_above_0_52": int(
                np.sum(ba >= 0.52)
            ),
            "months_at_or_above_0_53": int(
                np.sum(ba >= 0.53)
            ),
        }
        results["monthly_stability_summary"] = stability

        print(
            "\nMONTHLY STABILITY SUMMARY"
        )
        print(
            json.dumps(
                stability,
                indent=2,
                sort_keys=True,
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

        json_path = (
            report_dir
            / "regular_binary_diag_R2_v1.json"
        )
        csv_path = (
            report_dir
            / "regular_binary_diag_R2_v1.csv"
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
        "\nREGULAR BINARY DIAGNOSTICS PASS"
    )
    print(
        "TEST was not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
