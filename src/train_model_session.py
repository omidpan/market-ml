#!/usr/bin/env python3
"""
train_model_session.py — session-specialized TensorFlow/Keras experiments.

Version 1.0.0

Purpose
-------
Train/evaluate a model on prediction endpoints from ONE market session while
reusing the existing continuous L120 sequence artifact.

This intentionally does NOT rebuild sequences. The input window may contain
causal bars from the previous session, but:
- prediction endpoint is restricted to --prediction-session
- with --target-same-session-only, the H15 target must remain in that session
- TRAIN and VALIDATION use the exact same regime filter
- TEST receives the same structural filter but is never evaluated and its
  class distribution is never printed

Supported experiment labels
---------------------------
atr3
    Materialized ATR-relative 3-class policy from data/parquet/label_policy.

binary_sign
    DOWN if target_return_bps < 0, UP if target_return_bps > 0.
    Exact-zero targets are excluded. This is equivalent to a zero-threshold
    binary direction experiment and does not require label-policy Parquet.

Safety
------
- Existing Parquet is READ-ONLY.
- Existing model/scaler/sequence artifacts are READ-ONLY.
- --preflight-only does not import TensorFlow/Keras and creates no output.
- TEST is never scored.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SESSION_TRAINER_VERSION = "1.0.0"
SUPPORTED_TRAIN_MODEL_PREFIX = "1.1."


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
            "train_model_session.py expects base train_model.py v1.1.x; "
            f"found {version!r}."
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

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )
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
            "Unrecognized target_same_session values: "
            f"{unknown}"
        )
    return values.map(mapping).to_numpy(dtype=bool)


def load_regime_metadata(
    tm,
    matrix_root: Path,
    symbol: str,
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    sequence_parts = tm.parts(
        matrix_root / "sequences",
        symbol,
    )

    required = [
        "source_id",
        "prediction_time",
        "split",
        "session",
        "target_same_session",
    ]

    first_cols = list(
        pd.read_parquet(
            sequence_parts[0],
            engine="pyarrow",
        ).columns
    )
    missing = sorted(set(required) - set(first_cols))
    if missing:
        raise ValueError(
            "Model-ready sequences do not contain required regime metadata: "
            f"{missing}"
        )

    frames = []
    for i, path in enumerate(sequence_parts, 1):
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=required,
        )
        frames.append(frame)

        if i % 20 == 0 or i == len(sequence_parts):
            print(
                f"  regime metadata: loaded "
                f"{i:>3}/{len(sequence_parts)} files"
            )

    meta = pd.concat(frames, ignore_index=True)

    meta["source_id"] = (
        meta["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    meta["split"] = (
        meta["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    meta["session"] = (
        meta["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    meta["_prediction_ns"] = to_ns(
        meta["prediction_time"]
    )
    meta["_target_same_session"] = normalize_bool(
        meta["target_same_session"]
    )

    dup = meta.duplicated(
        ["source_id", "_prediction_ns"]
    )
    if dup.any():
        raise ValueError(
            "Duplicate regime metadata endpoint keys: "
            f"{int(dup.sum()):,}"
        )

    seq_keys = sequences[
        ["source_id", "_prediction_ns", "split"]
    ].copy()
    seq_keys["_row"] = np.arange(
        len(seq_keys),
        dtype=np.int64,
    )

    joined = seq_keys.merge(
        meta[
            [
                "source_id",
                "_prediction_ns",
                "split",
                "session",
                "_target_same_session",
            ]
        ].rename(
            columns={"split": "_metadata_split"}
        ),
        on=["source_id", "_prediction_ns"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row")

    if joined["session"].isna().any():
        raise ValueError(
            "Missing regime metadata after endpoint alignment: "
            f"{int(joined['session'].isna().sum()):,}"
        )

    mismatch = (
        joined["split"]
        != joined["_metadata_split"]
    )
    if mismatch.any():
        raise ValueError(
            "Regime metadata split mismatch: "
            f"{int(mismatch.sum()):,}"
        )

    if len(joined) != len(sequences):
        raise ValueError(
            "Regime metadata alignment length mismatch."
        )

    return joined.reset_index(drop=True)


def binary_sign_labels(
    sequences: pd.DataFrame,
):
    returns = pd.to_numeric(
        sequences["target_return_bps"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    finite = np.isfinite(returns)
    if not finite.all():
        raise ValueError(
            "Non-finite target_return_bps in model-ready sequences."
        )

    keep = returns != 0.0
    labels = np.zeros(
        len(returns),
        dtype=np.int64,
    )
    labels[returns > 0.0] = 1

    return (
        labels,
        keep,
        ("down", "up"),
        {
            "name": "binary_sign",
            "threshold_bps": 0.0,
            "zero_return_rows_excluded": int((~keep).sum()),
        },
    )


def class_distribution(
    labels: np.ndarray,
    indices: np.ndarray,
    class_names,
):
    total = len(indices)
    out = {}
    for i, name in enumerate(class_names):
        count = int(
            np.sum(labels[indices] == i)
        )
        out[name] = {
            "count": count,
            "pct": (
                100.0 * count / total
                if total
                else float("nan")
            ),
        }
    return out


def print_distribution(
    title: str,
    labels: np.ndarray,
    indices: np.ndarray,
    class_names,
):
    print(f"\n{title}:")
    dist = class_distribution(
        labels,
        indices,
        class_names,
    )
    for name in class_names:
        info = dist[name]
        print(
            f"  {name:<8} "
            f"{info['count']:>10,} "
            f"({info['pct']:7.3f}%)"
        )


def print_preflight(
    split: np.ndarray,
    labels: np.ndarray,
    class_names,
    prediction_session: str,
    same_session_only: bool,
    experiment_label: str,
):
    train_idx = np.flatnonzero(
        split == "train"
    )
    val_idx = np.flatnonzero(
        split == "validation"
    )
    test_idx = np.flatnonzero(
        split == "test"
    )

    print("\nPRE-FLIGHT SUMMARY")
    print("------------------")
    print(
        f"prediction_session:       {prediction_session}"
    )
    print(
        "target_same_session_only: "
        f"{same_session_only}"
    )
    print(
        f"experiment_label:         {experiment_label}"
    )
    print(f"TRAIN samples:            {len(train_idx):,}")
    print(f"VALIDATION samples:       {len(val_idx):,}")
    print(f"TEST samples:             {len(test_idx):,} (sealed)")

    print_distribution(
        "TRAIN class distribution",
        labels,
        train_idx,
        class_names,
    )
    print_distribution(
        "VALIDATION class distribution",
        labels,
        val_idx,
        class_names,
    )

    print(
        "\nTEST class distribution: intentionally not shown."
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    p.add_argument(
        "--model-matrix-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--label-policy-root",
        type=Path,
    )
    p.add_argument(
        "--output-root",
        type=Path,
    )
    p.add_argument(
        "--symbol",
        required=True,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
    )
    p.add_argument(
        "--reports",
        type=Path,
    )
    p.add_argument(
        "--run-id",
    )

    p.add_argument(
        "--prediction-session",
        choices=[
            "premarket",
            "regular",
            "aftermarket",
        ],
        required=True,
    )

    p.add_argument(
        "--target-same-session-only",
        action="store_true",
        help=(
            "Keep only samples whose exact H target remains "
            "inside the prediction session."
        ),
    )

    p.add_argument(
        "--experiment-label",
        choices=["atr3", "binary_sign"],
        default="atr3",
    )

    p.add_argument(
        "--preflight-only",
        action="store_true",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
    )
    p.add_argument(
        "--show-config",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    tm = import_train_model(
        project_root
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

    if args.smoke_test:
        # Keep every original hyperparameter except epochs.
        cfg_dict = asdict(cfg)
        cfg_dict["label_policy"] = cfg.label_policy
        cfg_dict["epochs"] = 1
        cfg = tm.TrainConfig(**cfg_dict)

    if (
        args.experiment_label == "atr3"
        and args.label_policy_root is None
    ):
        raise ValueError(
            "--label-policy-root is required for --experiment-label atr3."
        )

    if (
        not args.preflight_only
        and not args.show_config
        and args.output_root is None
    ):
        raise ValueError(
            "--output-root is required for training."
        )

    expected_signature = tm.signature(
        feature_set,
        features,
    )

    regime_tag = (
        f"{args.prediction_session}_"
        + (
            "same_target"
            if args.target_same_session_only
            else "all_targets"
        )
    )

    resolved = {
        "session_trainer_version": SESSION_TRAINER_VERSION,
        "base_train_model_version": tm.TRAIN_MODEL_VERSION,
        "backend": "tensorflow (deferred until training)",
        "symbol": args.symbol.strip().lower(),
        "sequence_length": address.length,
        "target_horizon_minutes": address.horizon,
        "scope": address.scope,
        "feature_set": feature_set,
        "feature_count": len(features),
        "prediction_session": args.prediction_session,
        "target_same_session_only": args.target_same_session_only,
        "regime_tag": regime_tag,
        "experiment_label": args.experiment_label,
        "training": {
            **asdict(cfg),
            "label_policy": asdict(
                cfg.label_policy
            ),
        },
        "preflight_only": args.preflight_only,
        "smoke_test": args.smoke_test,
    }

    if args.show_config:
        print(
            json.dumps(
                resolved,
                indent=2,
            )
        )
        return 0

    symbol = (
        args.symbol
        .strip()
        .lower()
    )

    matrix_root = (
        address.root(
            args.model_matrix_root
        )
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
        f"train_model_session.py={SESSION_TRAINER_VERSION} "
        f"base_train_model.py={tm.TRAIN_MODEL_VERSION}"
    )
    print(
        f"session={args.prediction_session} "
        f"same_target={args.target_same_session_only} "
        f"label={args.experiment_label}"
    )

    print("\n[1/5] Loading scaled model matrix...")

    (
        matrix,
        time_ns,
        source_code,
        code_map,
    ) = tm.load_matrix(
        matrix_root,
        symbol,
        features,
        expected_signature,
    )

    print("\n[2/5] Loading model-ready sequences...")

    sequences = tm.load_sequences(
        matrix_root,
        symbol,
        expected_signature,
    )

    print("\n[3/5] Loading/aliging session metadata...")

    regime_meta = load_regime_metadata(
        tm,
        matrix_root,
        symbol,
        sequences,
    )

    structural_keep = (
        regime_meta["session"]
        .to_numpy(dtype=object)
        == args.prediction_session
    )

    if args.target_same_session_only:
        structural_keep &= (
            regime_meta["_target_same_session"]
            .to_numpy(dtype=bool)
        )

    structural_count = int(
        structural_keep.sum()
    )

    if not structural_count:
        raise ValueError(
            "Regime filter selected zero sequences."
        )

    print(
        "Regime structural filter: "
        f"{structural_count:,}/{len(sequences):,} sequences"
    )

    print("\n[4/5] Resolving experiment labels...")

    if args.experiment_label == "atr3":
        (
            labels,
            label_keep,
            class_names,
            label_info,
        ) = tm.load_policy_labels(
            args.label_policy_root,
            sequences,
            address,
            feature_set,
            cfg,
            symbol,
        )
    else:
        (
            labels,
            label_keep,
            class_names,
            label_info,
        ) = binary_sign_labels(
            sequences
        )
        print(
            "Binary-sign labels: "
            f"exact-zero excluded={int((~label_keep).sum()):,}"
        )

    keep = (
        structural_keep
        & label_keep
    )

    print(
        "Final regime + label selection: "
        f"{int(keep.sum()):,}/{len(keep):,} sequences"
    )

    print("\n[5/5] Mapping exact sequence windows...")

    starts_all = tm.map_starts(
        time_ns,
        source_code,
        code_map,
        sequences,
        address.length,
    )

    split = (
        sequences["split"]
        .to_numpy(dtype=object)[keep]
    )
    starts = (
        starts_all[keep]
    )
    labels = (
        labels[keep]
    )

    if not (
        len(split)
        == len(starts)
        == len(labels)
    ):
        raise ValueError(
            "Post-filter alignment length mismatch."
        )

    print(
        f"Window mapping: PASS "
        f"({len(starts):,} selected sequences)"
    )

    print_preflight(
        split,
        labels,
        class_names,
        args.prediction_session,
        args.target_same_session_only,
        args.experiment_label,
    )

    if args.preflight_only:
        print("\nPRE-FLIGHT PASS")
        print("TensorFlow/Keras were not imported.")
        print("No model/report output was created.")
        print("TEST model evaluation was not performed.")
        return 0

    # TensorFlow/Keras starts only after all data/regime/label/window checks.
    keras, tf = tm.import_ml_stack()

    keras.utils.set_random_seed(
        cfg.seed
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    use_mixed = (
        bool(gpus)
        and cfg.mixed_precision
    )

    keras.mixed_precision.set_global_policy(
        "mixed_float16"
        if use_mixed
        else "float32"
    )

    print(
        f"\nTensorFlow={tf.__version__} "
        f"Keras={keras.__version__} "
        f"GPU={[x.name for x in gpus]} "
        f"mixed_precision={use_mixed}"
    )

    train_limit = (
        8192
        if args.smoke_test
        else None
    )
    val_limit = (
        4096
        if args.smoke_test
        else None
    )

    train_idx = tm.limited(
        split,
        "train",
        train_limit,
    )
    val_idx = tm.limited(
        split,
        "validation",
        val_limit,
    )
    test_count = int(
        np.sum(split == "test")
    )

    class_weight = tm.class_weight_from_labels(
        labels,
        train_idx,
        len(class_names),
    )

    print("\nClass weights (balanced, from TRAIN split):")
    for i, name in enumerate(class_names):
        print(f"  {name:<8} weight={class_weight[i]:.4f}")

    WindowDataset = tm.make_window_dataset_class(
        keras
    )

    train_ds = WindowDataset(
        matrix,
        starts[train_idx],
        labels[train_idx],
        address.length,
        cfg.batch_size,
        True,
        cfg.seed,
        cfg.workers,
    )

    val_ds = WindowDataset(
        matrix,
        starts[val_idx],
        labels[val_idx],
        address.length,
        cfg.batch_size,
        False,
        cfg.seed + 1,
        cfg.workers,
    )

    model = tm.build_model(
        keras,
        address.length,
        len(features),
        len(class_names),
        cfg,
    )

    model.summary()

    run_id = (
        args.run_id
        or (
            f"{symbol}_{feature_set}_"
            f"{args.experiment_label}_"
            f"{regime_tag}_"
            + datetime.now(
                timezone.utc
            ).strftime("%Y%m%dT%H%M%SZ")
        )
    )

    run_root = (
        address.root(
            args.output_root
        )
        / f"feature_set={feature_set}"
        / "model=lstm"
        / f"regime={regime_tag}"
        / f"symbol={symbol}"
        / f"run_id={run_id}"
    )
    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    best_path = (
        run_root
        / "best_model.keras"
    )
    last_path = (
        run_root
        / "last_model.keras"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor=cfg.monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=cfg.monitor,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            restore_best_weights=False,
            verbose=1,
        ),
    ]

    started = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(
        last_path
    )

    best_model = keras.models.load_model(
        best_path
    )

    # Validation is permitted. TEST remains untouched.
    val_eval = best_model.evaluate(
        val_ds,
        return_dict=True,
        verbose=1,
    )
    val_logits = best_model.predict(
        val_ds,
        verbose=1,
    )
    val_prediction = np.argmax(
        val_logits,
        axis=1,
    )

    metrics = tm.validation_metrics(
        labels[val_idx],
        val_prediction,
        class_names,
    )
    metrics.update(
        {
            key: float(value)
            for key, value in val_eval.items()
        }
    )

    history_dict = {
        key: [
            float(x)
            for x in values
        ]
        for key, values in history.history.items()
    }

    if cfg.monitor not in history_dict:
        raise ValueError(
            f"Monitor {cfg.monitor!r} missing from training history."
        )

    best_epoch = (
        int(
            np.argmin(
                history_dict[cfg.monitor]
            )
        )
        + 1
    )

    train_distribution = class_distribution(
        labels,
        train_idx,
        class_names,
    )
    validation_distribution = class_distribution(
        labels,
        val_idx,
        class_names,
    )

    report = {
        "session_trainer_version": SESSION_TRAINER_VERSION,
        "base_train_model_version": tm.TRAIN_MODEL_VERSION,
        "status": "PASS",
        "run_id": run_id,
        "symbol": symbol,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "gpu_devices": [
            x.name for x in gpus
        ],
        "mixed_precision_effective": use_mixed,
        "feature_set": feature_set,
        "feature_count": len(features),
        "feature_signature": expected_signature,
        "sequence_length": address.length,
        "target_horizon_minutes": address.horizon,
        "source_sequence_scope": address.scope,
        "regime_filter": {
            "prediction_session": args.prediction_session,
            "target_same_session_only": (
                args.target_same_session_only
            ),
            "regime_tag": regime_tag,
            "important_semantics": (
                "Prediction endpoint and target are regime-filtered. "
                "The existing continuous L120 input window is reused and "
                "may causally include earlier-session bars."
            ),
        },
        "experiment_label": args.experiment_label,
        "label_info": label_info,
        "class_names": list(class_names),
        "train_samples_used": int(
            len(train_idx)
        ),
        "validation_samples_used": int(
            len(val_idx)
        ),
        "held_out_test_samples_not_evaluated": test_count,
        "train_label_distribution": train_distribution,
        "validation_label_distribution": validation_distribution,
        "class_weight": class_weight,
        "training_configuration": {
            **asdict(cfg),
            "label_policy": asdict(
                cfg.label_policy
            ),
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(
            history.epoch
        ),
        "history": history_dict,
        "best_validation_metrics": metrics,
        "best_model_file": str(best_path),
        "last_model_file": str(last_path),
        "scaler_file": str(scaler_path),
        "run_root": str(run_root),
        "total_training_seconds": float(
            time.time() - started
        ),
        "smoke_test": args.smoke_test,
        "test_policy": (
            "TEST receives the same structural regime filter but is "
            "intentionally not evaluated and its class distribution is "
            "not printed."
        ),
    }

    report_path = (
        run_root
        / "training_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.reports:
        external = (
            args.reports
            / symbol
            / (
                f"training_summary_session_"
                f"v{SESSION_TRAINER_VERSION}_"
                f"{feature_set}_"
                f"L{address.length}_"
                f"H{address.horizon}m_"
                f"{args.experiment_label}_"
                f"{regime_tag}_"
                f"{run_id}.json"
            )
        )
        external.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        external.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Training report: {external}"
        )

    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        "Validation: "
        f"loss={metrics['loss']:.6f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(
        f"Held-out TEST not evaluated: "
        f"{test_count:,} samples"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
