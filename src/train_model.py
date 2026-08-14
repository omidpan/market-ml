#!/usr/bin/env python3
"""
train_model.py — TensorFlow/Keras trainer with local preflight support.

v1.1 additions
--------------
- Supports materialized ATR-relative label-policy Parquet:
    label_mode: label_policy_3class
- Validates policy manifest/signature/config against the training config.
- Exact one-to-one alignment of policy labels with model-ready sequences.
- Adds --preflight-only. This performs ALL data/label/window alignment checks
  without importing TensorFlow or Keras, so it can run locally.
- TensorFlow/Keras imports are deferred until actual training.
- TEST remains sealed: no TEST model evaluation and no TEST class distribution.

Existing modes remain supported:
- canonical_3class
- threshold_3class
- binary_threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


TRAIN_MODEL_VERSION = "1.1.1-tf"


@dataclass(frozen=True)
class SequenceAddress:
    length: int
    horizon: int
    stride: int
    scope: str
    sessions: tuple[str, ...]

    @property
    def sessions_tag(self) -> str:
        return "+".join(self.sessions)

    def root(self, base: Path) -> Path:
        return (
            base
            / f"sequence_length={self.length}"
            / f"horizon={self.horizon}m"
            / f"scope={self.scope}"
            / f"stride={self.stride}"
            / f"sessions={self.sessions_tag}"
        )


@dataclass(frozen=True)
class LabelPolicyConfig:
    name: str = "atr_relative_3class_v1"
    atr_period: int = 14
    multiplier: float = 0.75


@dataclass(frozen=True)
class TrainConfig:
    label_mode: str = "canonical_3class"
    neutral_threshold_bps: float = 0.0
    label_policy: LabelPolicyConfig = LabelPolicyConfig()
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.20
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    monitor: str = "val_loss"
    patience: int = 3
    min_delta: float = 0.0001
    seed: int = 42
    mixed_precision: bool = True
    workers: int = 2


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def resolve_config(raw: dict):
    pipeline = raw.get("pipeline", {})

    seq = pipeline.get("sequences", {}) or {}
    sessions = tuple(
        str(x).strip().lower()
        for x in seq.get(
            "sessions",
            ["premarket", "regular", "aftermarket"],
        )
    )

    address = SequenceAddress(
        length=int(seq.get("sequence_length", 120)),
        horizon=int(seq.get("target_horizon_minutes", 15)),
        stride=int(seq.get("stride_minutes", 1)),
        scope=str(seq.get("scope", "continuous")).strip().lower(),
        sessions=sessions,
    )

    matrix = pipeline.get("model_matrix", {}) or {}
    feature_set = str(
        matrix.get("feature_set", "core_v1")
    ).strip().lower()

    features = tuple(
        str(x).strip()
        for x in matrix.get("features", [])
    )

    if not features:
        raise ValueError("pipeline.model_matrix.features is required.")

    training = pipeline.get("training", {}) or {}
    policy_raw = training.get("label_policy", {}) or {}

    policy = LabelPolicyConfig(
        name=str(
            policy_raw.get("name", "atr_relative_3class_v1")
        ).strip(),
        atr_period=int(policy_raw.get("atr_period", 14)),
        multiplier=float(policy_raw.get("multiplier", 0.75)),
    )

    cfg = TrainConfig(
        label_mode=str(
            training.get("label_mode", "canonical_3class")
        ).strip().lower(),
        neutral_threshold_bps=float(
            training.get("neutral_threshold_bps", 0.0)
        ),
        label_policy=policy,
        hidden_size=int(training.get("hidden_size", 64)),
        num_layers=int(training.get("num_layers", 2)),
        dropout=float(training.get("dropout", 0.20)),
        batch_size=int(training.get("batch_size", 512)),
        epochs=int(training.get("epochs", 10)),
        learning_rate=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
        gradient_clip_norm=float(
            training.get("gradient_clip_norm", 1.0)
        ),
        monitor=str(training.get("monitor", "val_loss")).strip(),
        patience=int(training.get("early_stopping_patience", 3)),
        min_delta=float(training.get("min_delta", 0.0001)),
        seed=int(training.get("seed", 42)),
        mixed_precision=parse_bool(
            training.get("mixed_precision"),
            True,
        ),
        workers=int(training.get("workers", 2)),
    )

    if cfg.label_mode == "label_policy_3class":
        if not cfg.label_policy.name:
            raise ValueError("training.label_policy.name is required.")
        if cfg.label_policy.atr_period <= 0:
            raise ValueError(
                "training.label_policy.atr_period must be positive."
            )
        if (
            not math.isfinite(cfg.label_policy.multiplier)
            or cfg.label_policy.multiplier < 0
        ):
            raise ValueError(
                "training.label_policy.multiplier must be finite and >= 0."
            )

    return address, feature_set, features, cfg


def signature(
    feature_set: str,
    features: tuple[str, ...],
) -> str:
    payload = {
        "feature_set": feature_set,
        "features": list(features),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def parts(root: Path, symbol: str):
    found = sorted(
        (root / f"symbol={symbol}").glob(
            "year=*/month=*/part.parquet"
        )
    )
    if not found:
        raise FileNotFoundError(
            f"No parquet files under {root / f'symbol={symbol}'}"
        )
    return found


def to_ns(series: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(series, utc=True, errors="raise")
        .astype("int64")
        .to_numpy(dtype=np.int64, copy=False)
    )


def load_matrix(
    root: Path,
    symbol: str,
    features: tuple[str, ...],
    expected_signature: str,
):
    columns = [
        "datetime",
        "source_id",
        "feature_signature",
        *features,
    ]

    xs, times, sources = [], [], []

    matrix_parts = parts(root / "rows", symbol)

    for i, path in enumerate(matrix_parts, 1):
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=columns,
        )

        if set(
            frame["feature_signature"].astype(str).unique()
        ) != {expected_signature}:
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

        if i % 20 == 0 or i == len(matrix_parts):
            print(
                f"  matrix rows: loaded {i:>3}/{len(matrix_parts)} files"
            )

    x = np.concatenate(xs)
    time_ns = np.concatenate(times)
    source_name = np.concatenate(sources)

    names = sorted(set(source_name))
    code_map = {
        name: i
        for i, name in enumerate(names)
    }

    source_code = np.array(
        [code_map[value] for value in source_name],
        dtype=np.int32,
    )

    order = np.lexsort((time_ns, source_code))

    x = np.ascontiguousarray(
        x[order],
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
        f"Matrix: {len(x):,} rows x {x.shape[1]} features "
        f"(~{x.nbytes / 2**20:.1f} MiB)"
    )

    return x, time_ns, source_code, code_map


def load_sequences(
    root: Path,
    symbol: str,
    expected_signature: str,
):
    sequence_parts = parts(root / "sequences", symbol)

    first_columns = list(
        pd.read_parquet(
            sequence_parts[0],
            engine="pyarrow",
        ).columns
    )

    required = [
        "feature_signature",
        "source_id",
        "window_start_datetime",
        "window_end_datetime",
        "prediction_time",
        "split",
        "target_return_bps",
        "target_direction",
    ]

    missing = sorted(set(required) - set(first_columns))
    if missing:
        raise ValueError(
            f"Model-ready sequences missing columns: {missing}"
        )

    optional = [
        "model_sample_index",
        "sample_index",
        "target_log_return",
        "trading_date",
        "session",
    ]

    columns = required + [
        c for c in optional
        if c in first_columns
    ]

    frames = []

    for i, path in enumerate(sequence_parts, 1):
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=columns,
        )

        if set(
            frame["feature_signature"].astype(str).unique()
        ) != {expected_signature}:
            raise ValueError(
                f"Feature signature mismatch: {path}"
            )

        frames.append(frame)

        if i % 20 == 0 or i == len(sequence_parts):
            print(
                f"  sequences: loaded {i:>3}/{len(sequence_parts)} files"
            )

    frame = pd.concat(
        frames,
        ignore_index=True,
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

    frame["_start_ns"] = to_ns(
        frame["window_start_datetime"]
    )
    frame["_end_ns"] = to_ns(
        frame["window_end_datetime"]
    )
    frame["_prediction_ns"] = to_ns(
        frame["prediction_time"]
    )

    # 1-minute feature bars are bar-start labeled. The final input bar at
    # window_end_datetime covers [t, t+1m) and becomes available at t+1m.
    # Therefore prediction_time must be exactly one minute AFTER the final
    # input bar timestamp, matching sequences.py semantics.
    expected_prediction_ns = (
        frame["_end_ns"].to_numpy(dtype=np.int64)
        + 60_000_000_000
    )

    actual_prediction_ns = frame[
        "_prediction_ns"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        actual_prediction_ns,
        expected_prediction_ns,
    ):
        delta_minutes = (
            actual_prediction_ns
            - frame["_end_ns"].to_numpy(dtype=np.int64)
        ) / 60_000_000_000

        unique_delta, counts = np.unique(
            delta_minutes,
            return_counts=True,
        )

        examples = list(
            zip(
                unique_delta[:10].tolist(),
                counts[:10].tolist(),
            )
        )

        raise ValueError(
            "Sequence timing mismatch: expected "
            "prediction_time = window_end_datetime + 1 minute "
            "for every model-ready sequence. "
            f"Observed delta-minute counts (sample): {examples}"
        )

    dup = frame.duplicated(
        ["source_id", "_prediction_ns"]
    )

    if dup.any():
        raise ValueError(
            "Duplicate model-ready (source_id,prediction_time) rows: "
            f"{int(dup.sum()):,}"
        )

    print(f"Model-ready sequences: {len(frame):,}")

    return frame


def legacy_labels_for(
    sequences: pd.DataFrame,
    cfg: TrainConfig,
):
    returns = pd.to_numeric(
        sequences["target_return_bps"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if cfg.label_mode == "canonical_3class":
        raw = pd.to_numeric(
            sequences["target_direction"],
            errors="raise",
        ).to_numpy(dtype=np.int64)

        if not np.isin(raw, [-1, 0, 1]).all():
            raise ValueError(
                "Expected target_direction in {-1,0,+1}."
            )

        return (
            raw + 1,
            np.ones(len(raw), dtype=bool),
            ("down", "neutral", "up"),
            None,
        )

    if cfg.label_mode == "threshold_3class":
        t = cfg.neutral_threshold_bps
        y = np.full(len(returns), 1, dtype=np.int64)
        y[returns < -t] = 0
        y[returns > t] = 2

        return (
            y,
            np.ones(len(y), dtype=bool),
            ("down", "neutral", "up"),
            None,
        )

    if cfg.label_mode == "binary_threshold":
        t = cfg.neutral_threshold_bps
        keep = (
            (returns < -t)
            | (returns > t)
        )
        y = np.zeros(len(returns), dtype=np.int64)
        y[returns > t] = 1

        return (
            y,
            keep,
            ("down", "up"),
            None,
        )

    raise ValueError(
        f"Unsupported label_mode={cfg.label_mode!r}"
    )


def path_number(value: float) -> str:
    text = f"{value:.12g}"
    return (
        text.rstrip("0").rstrip(".")
        if "." in text
        else text
    )


def label_policy_symbol_root(
    base: Path,
    address: SequenceAddress,
    cfg: TrainConfig,
    symbol: str,
) -> Path:
    p = cfg.label_policy

    return (
        base
        / f"policy={p.name}"
        / f"atr_period={p.atr_period}"
        / f"horizon={address.horizon}m"
        / f"multiplier={path_number(p.multiplier)}"
        / f"symbol={symbol}"
    )


def load_policy_labels(
    base: Path,
    sequences: pd.DataFrame,
    address: SequenceAddress,
    feature_set: str,
    cfg: TrainConfig,
    symbol: str,
):
    root = label_policy_symbol_root(
        base,
        address,
        cfg,
        symbol,
    )

    print(f"Label policy root: {root}")

    manifest_path = root / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Label-policy manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    metadata = manifest.get("policy_metadata", {}) or {}
    manifest_signature = str(
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
                "Label-policy manifest mismatch for "
                f"{key}: expected {expected_value!r}, got {actual!r}"
            )

    atr = metadata.get("atr", {}) or {}
    threshold = metadata.get("threshold", {}) or {}

    if int(atr.get("period", -1)) != cfg.label_policy.atr_period:
        raise ValueError(
            "Label-policy ATR period mismatch."
        )

    actual_multiplier = float(
        threshold.get("multiplier", float("nan"))
    )

    if not math.isclose(
        actual_multiplier,
        cfg.label_policy.multiplier,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Label-policy multiplier mismatch."
        )

    if not manifest_signature:
        raise ValueError(
            "Label-policy manifest missing policy_signature."
        )

    policy_parts = sorted(
        root.glob("year=*/month=*/part.parquet")
    )

    if not policy_parts:
        raise FileNotFoundError(
            f"No label-policy Parquet parts under {root}"
        )

    required = [
        "prediction_time",
        "source_id",
        "split",
        "label_direction",
        "policy_name",
        "policy_signature",
    ]

    first_columns = list(
        pd.read_parquet(
            policy_parts[0],
            engine="pyarrow",
        ).columns
    )

    missing = sorted(set(required) - set(first_columns))
    if missing:
        raise ValueError(
            f"Label-policy Parquet missing columns: {missing}"
        )

    optional = [
        "model_sample_index",
        "sample_index",
        "threshold_bps",
        "target_return_bps",
        "target_log_return",
    ]

    columns = required + [
        c for c in optional
        if c in first_columns
    ]

    frames = []

    for i, path in enumerate(policy_parts, 1):
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=columns,
        )

        signatures = set(
            frame["policy_signature"]
            .astype(str)
            .unique()
        )

        if signatures != {manifest_signature}:
            raise ValueError(
                f"Policy signature mismatch: {path}"
            )

        names = set(
            frame["policy_name"]
            .astype(str)
            .unique()
        )

        if names != {cfg.label_policy.name}:
            raise ValueError(
                f"Policy name mismatch: {path}"
            )

        frames.append(frame)

        if i % 20 == 0 or i == len(policy_parts):
            print(
                f"  policy labels: loaded {i:>3}/{len(policy_parts)} files"
            )

    policy = pd.concat(
        frames,
        ignore_index=True,
    )

    policy["source_id"] = (
        policy["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    policy["split"] = (
        policy["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    policy["_prediction_ns"] = to_ns(
        policy["prediction_time"]
    )

    dup = policy.duplicated(
        ["source_id", "_prediction_ns"]
    )

    if dup.any():
        raise ValueError(
            "Duplicate label-policy (source_id,prediction_time) rows: "
            f"{int(dup.sum()):,}"
        )

    if len(policy) != len(sequences):
        raise ValueError(
            "Label-policy/model-ready row-count mismatch: "
            f"{len(policy):,} vs {len(sequences):,}"
        )

    # Join by immutable endpoint identity. Keep sequence order.
    sequence_keys = sequences[
        ["source_id", "_prediction_ns", "split"]
    ].copy()

    sequence_keys["_row"] = np.arange(
        len(sequence_keys),
        dtype=np.int64,
    )

    policy_for_join = policy[
        [
            "source_id",
            "_prediction_ns",
            "split",
            "label_direction",
        ]
    ].rename(
        columns={"split": "_policy_split"}
    )

    aligned = sequence_keys.merge(
        policy_for_join,
        on=["source_id", "_prediction_ns"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row")

    if aligned["label_direction"].isna().any():
        missing_count = int(
            aligned["label_direction"].isna().sum()
        )
        raise ValueError(
            f"Missing aligned policy labels: {missing_count:,}"
        )

    split_mismatch = (
        aligned["split"]
        != aligned["_policy_split"]
    )

    if split_mismatch.any():
        raise ValueError(
            "Label-policy split mismatch on "
            f"{int(split_mismatch.sum()):,} rows."
        )

    raw = pd.to_numeric(
        aligned["label_direction"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if not np.isin(raw, [-1, 0, 1]).all():
        raise ValueError(
            "Expected label_direction in {-1,0,+1}."
        )

    labels = raw + 1
    keep = np.ones(len(labels), dtype=bool)

    policy_info = {
        "root": str(root),
        "manifest_file": str(manifest_path),
        "policy_name": cfg.label_policy.name,
        "policy_signature": manifest_signature,
        "atr_period": cfg.label_policy.atr_period,
        "multiplier": cfg.label_policy.multiplier,
        "rows": int(len(policy)),
    }

    print(
        "Label-policy alignment: PASS "
        f"({len(labels):,} rows, signature={manifest_signature[:12]}...)"
    )

    return (
        labels,
        keep,
        ("down", "neutral", "up"),
        policy_info,
    )


def map_starts(
    time_ns: np.ndarray,
    source_code: np.ndarray,
    code_map: dict[str, int],
    sequences: pd.DataFrame,
    sequence_length: int,
):
    source_names = sequences[
        "source_id"
    ].to_numpy(dtype=object)

    try:
        sequence_codes = np.array(
            [code_map[str(x)] for x in source_names],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError(
            f"Sequence source missing from matrix: {exc}"
        ) from exc

    starts = np.full(
        len(sequences),
        -1,
        dtype=np.int64,
    )

    start_ns = sequences[
        "_start_ns"
    ].to_numpy(dtype=np.int64)

    end_ns = sequences[
        "_end_ns"
    ].to_numpy(dtype=np.int64)

    expected_span = (
        sequence_length - 1
    ) * 60 * 1_000_000_000

    if (
        end_ns - start_ns
        != expected_span
    ).any():
        raise ValueError(
            "Sequence timestamp span mismatch."
        )

    for name, code in code_map.items():
        matrix_pos = np.flatnonzero(
            source_code == code
        )
        seq_pos = np.flatnonzero(
            sequence_codes == code
        )

        if not len(seq_pos):
            continue

        matrix_times = time_ns[
            matrix_pos
        ]

        local_end = np.searchsorted(
            matrix_times,
            end_ns[seq_pos],
        )

        if (
            local_end >= len(matrix_times)
        ).any():
            raise ValueError(
                f"Sequence endpoint missing for source={name}"
            )

        global_end = matrix_pos[
            local_end
        ]

        if not np.array_equal(
            time_ns[global_end],
            end_ns[seq_pos],
        ):
            raise ValueError(
                f"Sequence endpoint mismatch for source={name}"
            )

        global_start = (
            global_end
            - (sequence_length - 1)
        )

        if (global_start < 0).any():
            raise ValueError(
                f"Negative sequence start for source={name}"
            )

        if not np.array_equal(
            time_ns[global_start],
            start_ns[seq_pos],
        ):
            raise ValueError(
                f"Sequence start mismatch for source={name}"
            )

        if not np.array_equal(
            source_code[global_start],
            sequence_codes[seq_pos],
        ):
            raise ValueError(
                f"Sequence start source mismatch for source={name}"
            )

        starts[seq_pos] = global_start

    if (starts < 0).any():
        raise ValueError(
            "Some sequences could not be mapped to matrix rows."
        )

    return starts


def limited(
    split: np.ndarray,
    name: str,
    limit: int | None,
):
    idx = np.flatnonzero(
        split == name
    )

    if not len(idx):
        raise ValueError(
            f"No {name} rows."
        )

    if limit is None or len(idx) <= limit:
        return idx

    return idx[
        np.linspace(
            0,
            len(idx) - 1,
            limit,
            dtype=np.int64,
        )
    ]


def class_counts_for_indices(
    labels: np.ndarray,
    indices: np.ndarray,
    class_names: tuple[str, ...],
):
    values = labels[indices]
    return {
        class_names[i]: int(
            np.sum(values == i)
        )
        for i in range(len(class_names))
    }


def class_weight_from_labels(
    labels: np.ndarray,
    indices: np.ndarray,
    num_classes: int,
):
    counts = np.bincount(
        labels[indices],
        minlength=num_classes,
    ).astype(np.float64)

    if (counts <= 0).any():
        raise ValueError(
            "Cannot compute class weights: empty class in "
            f"training split (counts={counts.tolist()})."
        )

    total = counts.sum()
    weights = total / (num_classes * counts)

    return {
        int(i): float(weights[i])
        for i in range(num_classes)
    }


def print_preflight_summary(
    split: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
):
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")

    print("\nPRE-FLIGHT SUMMARY")
    print("------------------")
    print(f"TRAIN samples:      {len(train_idx):,}")
    print(f"VALIDATION samples: {len(val_idx):,}")
    print(f"TEST samples:       {len(test_idx):,} (sealed)")

    print("\nTRAIN class distribution:")
    counts = class_counts_for_indices(
        labels,
        train_idx,
        class_names,
    )

    for name in class_names:
        count = counts[name]
        pct = (
            100.0 * count / len(train_idx)
            if len(train_idx)
            else float("nan")
        )
        print(
            f"  {name:<8} {count:>10,} ({pct:7.3f}%)"
        )

    # Validation labels are allowed for model selection, but TEST remains sealed.
    print("\nVALIDATION class distribution:")
    counts = class_counts_for_indices(
        labels,
        val_idx,
        class_names,
    )

    for name in class_names:
        count = counts[name]
        pct = (
            100.0 * count / len(val_idx)
            if len(val_idx)
            else float("nan")
        )
        print(
            f"  {name:<8} {count:>10,} ({pct:7.3f}%)"
        )

    print("\nTEST class distribution: intentionally not shown.")


def validation_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    class_names,
):
    n = len(class_names)

    matrix = np.zeros(
        (n, n),
        dtype=np.int64,
    )

    np.add.at(
        matrix,
        (truth, predicted),
        1,
    )

    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    tp = np.diag(matrix)

    precision = np.divide(
        tp,
        predicted_count,
        out=np.zeros(n, dtype=float),
        where=predicted_count != 0,
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros(n, dtype=float),
        where=support != 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(n, dtype=float),
        where=(precision + recall) != 0,
    )

    return {
        "accuracy": float(
            tp.sum() / matrix.sum()
        ),
        "balanced_accuracy": float(
            recall.mean()
        ),
        "macro_f1": float(
            f1.mean()
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(
            class_names
        ),
        "per_class": {
            name: {
                "precision": float(
                    precision[i]
                ),
                "recall": float(
                    recall[i]
                ),
                "f1": float(
                    f1[i]
                ),
                "support": int(
                    support[i]
                ),
            }
            for i, name in enumerate(
                class_names
            )
        },
    }


def import_ml_stack():
    # Select backend before importing Keras.
    os.environ.setdefault(
        "KERAS_BACKEND",
        "tensorflow",
    )

    import keras
    import tensorflow as tf

    return keras, tf


def make_window_dataset_class(keras):
    class WindowDataset(keras.utils.PyDataset):
        """Materialize only current batch: B x L x F."""

        def __init__(
            self,
            matrix,
            starts,
            labels,
            length,
            batch_size,
            shuffle,
            seed,
            workers,
        ):
            super().__init__(
                workers=workers,
                use_multiprocessing=False,
                max_queue_size=8,
            )

            self.matrix = matrix
            self.starts = np.asarray(
                starts,
                dtype=np.int64,
            )
            self.labels = np.asarray(
                labels,
                dtype=np.int64,
            )
            self.length = int(length)
            self.batch_size = int(batch_size)
            self.shuffle = bool(shuffle)
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(
                len(self.labels),
                dtype=np.int64,
            )
            self.offsets = np.arange(
                self.length,
                dtype=np.int64,
            )

            if self.shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return math.ceil(
                len(self.labels)
                / self.batch_size
            )

        def __getitem__(self, batch_index):
            low = (
                batch_index
                * self.batch_size
            )

            high = min(
                low + self.batch_size,
                len(self.labels),
            )

            ids = self.order[low:high]
            starts_local = self.starts[ids]

            positions = (
                starts_local[:, None]
                + self.offsets[None, :]
            )

            x = self.matrix[positions]
            y = self.labels[ids]

            return (
                np.asarray(x, dtype=np.float32),
                np.asarray(y, dtype=np.int64),
            )

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)

    return WindowDataset


def build_model(
    keras,
    sequence_length: int,
    feature_count: int,
    class_count: int,
    cfg: TrainConfig,
):
    inputs = keras.Input(
        shape=(sequence_length, feature_count),
        dtype="float32",
        name="features",
    )

    x = inputs

    for i in range(cfg.num_layers):
        x = keras.layers.LSTM(
            cfg.hidden_size,
            return_sequences=(
                i < cfg.num_layers - 1
            ),
            dropout=0.0,
            recurrent_dropout=0.0,
            use_cudnn="auto",
            name=f"lstm_{i + 1}",
        )(x)

        if cfg.dropout:
            x = keras.layers.Dropout(
                cfg.dropout,
                name=f"dropout_{i + 1}",
            )(x)

    outputs = keras.layers.Dense(
        class_count,
        dtype="float32",
        name="logits",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="market_lstm",
    )

    optimizer_kwargs = dict(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if cfg.gradient_clip_norm > 0:
        optimizer_kwargs["clipnorm"] = (
            cfg.gradient_clip_norm
        )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            **optimizer_kwargs
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model-matrix-root",
        type=Path,
    )

    p.add_argument(
        "--label-policy-root",
        type=Path,
        help=(
            "Base data/parquet/label_policy directory. "
            "Required when label_mode=label_policy_3class."
        ),
    )

    p.add_argument(
        "--output-root",
        type=Path,
    )

    p.add_argument(
        "--symbol",
    )

    p.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/pipeline.yaml"
        ),
    )

    p.add_argument(
        "--reports",
        type=Path,
    )

    p.add_argument(
        "--run-id",
    )

    p.add_argument(
        "--smoke-test",
        action="store_true",
    )

    p.add_argument(
        "--show-config",
        action="store_true",
    )

    p.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Load/validate matrix, sequences, labels and exact windows, "
            "then exit before importing TensorFlow/Keras."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()
    raw = load_yaml(args.config)

    (
        address,
        feature_set,
        features,
        cfg,
    ) = resolve_config(raw)

    if args.smoke_test:
        cfg = TrainConfig(
            **{
                **asdict(cfg),
                "label_policy": cfg.label_policy,
                "epochs": 1,
            }
        )

    expected_signature = signature(
        feature_set,
        features,
    )

    resolved = {
        "version": TRAIN_MODEL_VERSION,
        "backend": "tensorflow (deferred until training)",
        "sequence_length": address.length,
        "target_horizon": address.horizon,
        "feature_set": feature_set,
        "feature_count": len(features),
        "training": {
            **asdict(cfg),
            "label_policy": asdict(
                cfg.label_policy
            ),
        },
        "smoke_test": args.smoke_test,
        "preflight_only": args.preflight_only,
    }

    if args.show_config:
        print(
            json.dumps(
                resolved,
                indent=2,
            )
        )
        return 0

    if (
        args.model_matrix_root is None
        or not args.symbol
    ):
        raise ValueError(
            "--model-matrix-root and --symbol are required."
        )

    if (
        not args.preflight_only
        and args.output_root is None
    ):
        raise ValueError(
            "--output-root is required for training."
        )

    if (
        cfg.label_mode == "label_policy_3class"
        and args.label_policy_root is None
    ):
        raise ValueError(
            "--label-policy-root is required when "
            "label_mode=label_policy_3class."
        )

    symbol = args.symbol.strip().lower()

    matrix_root = (
        address.root(
            args.model_matrix_root
        )
        / f"feature_set={feature_set}"
    )

    scaler_path = (
        matrix_root / "scaler.json"
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
        f"train_model.py={TRAIN_MODEL_VERSION} "
        f"label_mode={cfg.label_mode}"
    )

    print("\n[1/4] Loading scaled model matrix...")

    (
        matrix,
        time_ns,
        source_code,
        code_map,
    ) = load_matrix(
        matrix_root,
        symbol,
        features,
        expected_signature,
    )

    print("\n[2/4] Loading model-ready sequences...")

    sequences = load_sequences(
        matrix_root,
        symbol,
        expected_signature,
    )

    print("\n[3/4] Resolving labels...")

    if cfg.label_mode == "label_policy_3class":
        (
            labels,
            keep,
            class_names,
            policy_info,
        ) = load_policy_labels(
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
            keep,
            class_names,
            policy_info,
        ) = legacy_labels_for(
            sequences,
            cfg,
        )

    print("\n[4/4] Mapping exact sequence windows...")

    starts = map_starts(
        time_ns,
        source_code,
        code_map,
        sequences,
        address.length,
    )

    split = sequences[
        "split"
    ].to_numpy(dtype=object)[keep]

    starts = starts[keep]
    labels = labels[keep]

    if not (
        len(split)
        == len(starts)
        == len(labels)
    ):
        raise ValueError(
            "Post-label alignment length mismatch."
        )

    print(
        f"Window mapping: PASS ({len(starts):,} labeled sequences)"
    )

    print_preflight_summary(
        split,
        labels,
        class_names,
    )

    if args.preflight_only:
        print("\nPRE-FLIGHT PASS")
        print("TensorFlow/Keras were not imported.")
        print("No model/report output was created.")
        print("TEST model evaluation was not performed.")
        return 0

    # Actual ML stack starts here, after every input/alignment check passes.
    keras, tf = import_ml_stack()

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

    train_idx = limited(
        split,
        "train",
        train_limit,
    )

    val_idx = limited(
        split,
        "validation",
        val_limit,
    )

    test_count = int(
        (split == "test").sum()
    )

    class_weight = class_weight_from_labels(
        labels,
        train_idx,
        len(class_names),
    )

    print("\nClass weights (balanced, from TRAIN split):")
    for i, name in enumerate(class_names):
        print(f"  {name:<8} weight={class_weight[i]:.4f}")

    WindowDataset = make_window_dataset_class(
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

    model = build_model(
        keras,
        address.length,
        len(features),
        len(class_names),
        cfg,
    )

    model.summary()

    run_id = (
        args.run_id
        or datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    run_root = (
        address.root(
            args.output_root
        )
        / f"feature_set={feature_set}"
        / "model=lstm"
        / f"symbol={symbol}"
        / f"run_id={run_id}"
    )

    if run_root.exists():
        raise FileExistsError(
            f"Run exists: {run_root}"
        )

    run_root.mkdir(
        parents=True
    )

    best_path = (
        run_root / "best_model.keras"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor=cfg.monitor,
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=cfg.monitor,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(
            str(
                run_root / "history.csv"
            )
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
        run_root / "last_model.keras"
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

    metrics = validation_metrics(
        labels[val_idx],
        val_prediction,
        class_names,
    )

    metrics.update(
        {
            key: float(value)
            for key, value
            in val_eval.items()
        }
    )

    history_dict = {
        key: [
            float(x)
            for x in values
        ]
        for key, values
        in history.history.items()
    }

    best_epoch = (
        int(
            np.argmin(
                history_dict[
                    cfg.monitor
                ]
            )
        )
        + 1
    )

    report = {
        "train_model_version": (
            TRAIN_MODEL_VERSION
        ),
        "status": "PASS",
        "run_id": run_id,
        "symbol": symbol,
        "tensorflow_version": (
            tf.__version__
        ),
        "keras_version": (
            keras.__version__
        ),
        "keras_backend": (
            keras.backend.backend()
        ),
        "gpu_devices": [
            x.name
            for x in gpus
        ],
        "mixed_precision_effective": (
            use_mixed
        ),
        "feature_set": feature_set,
        "feature_count": len(features),
        "feature_signature": (
            expected_signature
        ),
        "sequence_length": (
            address.length
        ),
        "target_horizon_minutes": (
            address.horizon
        ),
        "label_mode": cfg.label_mode,
        "label_policy": policy_info,
        "class_names": list(
            class_names
        ),
        "train_samples_used": int(
            len(train_idx)
        ),
        "validation_samples_used": int(
            len(val_idx)
        ),
        "held_out_test_samples_not_evaluated": (
            test_count
        ),
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
        "class_weight": class_weight,
        "best_validation_metrics": (
            metrics
        ),
        "best_model_file": str(
            best_path
        ),
        "scaler_file": str(
            scaler_path
        ),
        "run_root": str(
            run_root
        ),
        "total_training_seconds": float(
            time.time() - started
        ),
        "smoke_test": (
            args.smoke_test
        ),
        "test_policy": (
            "TEST is intentionally not evaluated by train_model.py."
        ),
    }

    report_path = (
        run_root / "training_report.json"
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
                f"training_summary_"
                f"{TRAIN_MODEL_VERSION}_"
                f"{feature_set}_"
                f"L{address.length}_"
                f"H{address.horizon}m_"
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
