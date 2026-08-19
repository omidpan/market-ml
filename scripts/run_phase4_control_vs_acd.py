#!/usr/bin/env python3
"""
Milestone 1 Phase 4 — controlled CONTROL-vs-ACD LSTM experiment orchestrator.

Attaches the existing frozen ATR14x0.75 label policy to the Phase-3 common
sample universe, then prepares (and, only after explicit approval, trains)
two paired LSTM runs that differ ONLY in feature set:

    CONTROL  feature_set=core_v1          (23 features)
    ACD      feature_set=core_v1_acd_v1   (150 features)

Orchestrates existing approved modules (src/build_label_policy.py,
src/train_model.py, scripts/run_phase3_control_vs_acd.py) rather than
duplicating their logic. Does not touch data/raw. Does not touch production
data/parquet/label_policy or data/models. TEST is never opened.

    python scripts/run_phase4_control_vs_acd.py --mode labels
    python scripts/run_phase4_control_vs_acd.py --mode preflight
    python scripts/run_phase4_control_vs_acd.py --mode validate
    python scripts/run_phase4_control_vs_acd.py --mode all
    python scripts/run_phase4_control_vs_acd.py --mode train-control --confirm-training
    python scripts/run_phase4_control_vs_acd.py --mode train-acd --confirm-training
    python scripts/run_phase4_control_vs_acd.py --mode compare

All root paths default to the locations used throughout Milestone 1 Phase 4
(unchanged local-run behavior). They can be overridden via CLI flags to
point at Concourse-mounted input/output directories instead of assumed-
present local paths — see ci/concourse/phase04.yml.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from state_machine_features import COMMON_MODELING_START  # noqa: E402
from build_label_policy import path_number  # noqa: E402
import run_phase3_control_vs_acd as phase3  # noqa: E402

SYMBOL = "nvda"
BASE_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"

# Default roots — unchanged local-run behavior.
DEFAULT_MODEL_MATRIX_ROOT = phase3.DEFAULT_MODEL_MATRIX_ROOT
DEFAULT_ENCODED_FEATURES_ROOT = phase3.DEFAULT_FEATURES_1M_ACD_V1_ENCODED
DEFAULT_LABEL_POLICY_ROOT = REPO_ROOT / "data/parquet/label_policy_phase4_common_v1"
DEFAULT_LABEL_POLICY_REPORTS_ROOT = REPO_ROOT / "reports/label_policy_phase4"
DEFAULT_MODELS_ROOT = REPO_ROOT / "data/models_phase4_common_v1"
DEFAULT_TRAINING_REPORTS_ROOT = REPO_ROOT / "reports/training_phase4"
DEFAULT_GENERATED_CONFIG_DIR = REPO_ROOT / "config" / "generated"

SEQUENCE_SHAPE = phase3.SEQUENCE_SHAPE

# The shared label artifact is built once against core_v1's sequence
# endpoints (the canonical shared subtree — see build_labels()), so its
# manifest.json always records feature_set=core_v1 regardless of which
# variant is training. train_model.py's --label-policy-feature-set override
# tells it to validate against that declared value instead of assuming the
# manifest must match its own --config feature_set (see the train_model.py
# docstring/flag help for why this is safe: Phase 3 proved core_v1 and
# core_v1_acd_v1 share a row-identical sequence/label universe).
LABEL_POLICY_FEATURE_SET = "core_v1"

POLICY_NAME = "atr_relative_3class_v1"
ATR_PERIOD = 14
MULTIPLIER = 0.75
HORIZON = 15
SEQUENCE_LENGTH = 120

CONTROL_RUN_ID = "nvda_core_v1_phase4_v1"
ACD_RUN_ID = "nvda_core_v1_acd_v1_phase4_v1"

FROZEN_TRAINING_KEYS = [
    "architecture", "label_mode", "label_policy", "neutral_threshold_bps",
    "hidden_size", "num_layers", "dropout", "batch_size", "epochs",
    "learning_rate", "weight_decay", "gradient_clip_norm", "monitor",
    "early_stopping_patience", "min_delta", "seed", "mixed_precision",
    "workers",
]


def _run(cmd: list) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ {printable[:300]}{'...' if len(printable) > 300 else ''}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# Config generation — CONTROL/ACD differ only in model_matrix.feature_set/features
# ---------------------------------------------------------------------------

def generate_variant_configs(
    encoded_features_root: Path,
    generated_config_dir: Path,
) -> tuple[Path, Path]:
    """Writes two config clones of config/pipeline.yaml. The `training:` and
    `sequences:` blocks are copied byte-for-byte from the single base file
    for both variants — this is the mechanical guarantee that CONTROL and
    ACD are never tuned independently. Only pipeline.model_matrix.feature_set
    and .features differ."""

    with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    control_cfg = copy.deepcopy(base)
    control_cfg["pipeline"]["model_matrix"]["feature_set"] = "core_v1"
    # features left as-is: the existing 23 core_v1 entries.

    acd_cfg = copy.deepcopy(base)
    manifest = phase3._load_manifest(encoded_features_root)
    acd_features = phase3.derive_acd_features(manifest)
    acd_cfg["pipeline"]["model_matrix"]["feature_set"] = "core_v1_acd_v1"
    acd_cfg["pipeline"]["model_matrix"]["features"] = acd_features

    base_training = base["pipeline"]["training"]
    control_training = control_cfg["pipeline"]["training"]
    acd_training = acd_cfg["pipeline"]["training"]
    for key in FROZEN_TRAINING_KEYS:
        assert control_training[key] == base_training[key] == acd_training[key], (
            f"training.{key} diverged during config generation"
        )

    generated_config_dir.mkdir(parents=True, exist_ok=True)
    control_config_path = generated_config_dir / "pipeline_phase4_control.yaml"
    acd_config_path = generated_config_dir / "pipeline_phase4_acd.yaml"

    header = (
        "# AUTO-GENERATED by scripts/run_phase4_control_vs_acd.py\n"
        "# Do not hand-edit. Regenerate via --mode labels/preflight/validate/all.\n"
    )

    with open(control_config_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(control_cfg, f, sort_keys=False)

    with open(acd_config_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(acd_cfg, f, sort_keys=False)

    print(f"wrote {control_config_path}")
    print(f"wrote {acd_config_path} ({len(acd_features)} features)")

    return control_config_path, acd_config_path


# ---------------------------------------------------------------------------
# Shared label policy
# ---------------------------------------------------------------------------

def build_labels(model_matrix_root: Path, label_policy_root: Path, reports_root: Path) -> None:
    _run([
        sys.executable, SRC / "build_label_policy.py",
        "--project-root", REPO_ROOT,
        "--symbol", SYMBOL,
        "--horizon", HORIZON,
        "--sequence-length", SEQUENCE_LENGTH,
        "--feature-set", "core_v1",
        "--scope", "continuous",
        "--stride", "1",
        "--sessions", "premarket,regular,aftermarket",
        "--atr-period", ATR_PERIOD,
        "--multiplier", MULTIPLIER,
        "--policy-name", POLICY_NAME,
        "--model-matrix-root", model_matrix_root,
        "--label-policy-root", label_policy_root,
        "--reports-root", reports_root,
        "--write",
    ])


def label_policy_symbol_root(label_policy_root: Path) -> Path:
    return (
        label_policy_root
        / f"policy={POLICY_NAME}"
        / f"atr_period={ATR_PERIOD}"
        / f"horizon={HORIZON}m"
        / f"multiplier={path_number(MULTIPLIER)}"
        / f"symbol={SYMBOL}"
    )


# ---------------------------------------------------------------------------
# Preflight (no TF/Keras import, no training)
# ---------------------------------------------------------------------------

def _train_model_common_args(
    model_matrix_root: Path, label_policy_root: Path, config_path: Path,
) -> list:
    return [
        sys.executable, SRC / "train_model.py",
        "--model-matrix-root", model_matrix_root,
        "--label-policy-root", label_policy_root,
        "--label-policy-feature-set", LABEL_POLICY_FEATURE_SET,
        "--symbol", SYMBOL,
        "--config", config_path,
    ]


def run_preflight(
    model_matrix_root: Path, label_policy_root: Path, control_cfg: Path, acd_cfg: Path,
) -> None:
    print("\n[preflight] CONTROL")
    _run(_train_model_common_args(model_matrix_root, label_policy_root, control_cfg)
         + ["--preflight-only"])

    print("\n[preflight] ACD")
    _run(_train_model_common_args(model_matrix_root, label_policy_root, acd_cfg)
         + ["--preflight-only"])


# ---------------------------------------------------------------------------
# Training (gated behind --confirm-training; not run by --mode all)
# ---------------------------------------------------------------------------

def run_train(
    variant: str,
    confirm_training: bool,
    control_cfg: Path,
    acd_cfg: Path,
    model_matrix_root: Path,
    label_policy_root: Path,
    models_root: Path,
    training_reports_root: Path,
) -> None:
    if not confirm_training:
        raise SystemExit(
            f"Refusing to train ({variant}): pass --confirm-training explicitly. "
            "Training requires separate, explicit user approval per the Phase-4 "
            "Step-3 gate — this flag is a second, script-level enforcement of "
            "that gate, independent of the chat-level approval phrase."
        )

    config_path = control_cfg if variant == "control" else acd_cfg
    run_id = CONTROL_RUN_ID if variant == "control" else ACD_RUN_ID

    _run(_train_model_common_args(model_matrix_root, label_policy_root, config_path) + [
        "--output-root", models_root,
        "--reports", training_reports_root,
        "--run-id", run_id,
    ])


# ---------------------------------------------------------------------------
# Validation (pre-training pairing/label assertions; extends Phase-3 checks)
# ---------------------------------------------------------------------------

def _load_sequence_frame(root: Path, columns: list) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "sequences" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No sequences/ partitions found under {root}.")
    frames = [pq.ParquetFile(f).read(columns=columns).to_pandas() for f in files]
    return pd.concat(frames, ignore_index=True)


def _load_label_policy_frame(label_policy_root: Path) -> pd.DataFrame:
    root = label_policy_symbol_root(label_policy_root)
    files = sorted(glob.glob(str(root / "year=*/month=*/part.parquet")))
    if not files:
        raise FileNotFoundError(f"No label-policy Parquet parts under {root}.")
    columns = [
        "prediction_time", "source_id", "split", "trading_date",
        "label_direction", "policy_name", "policy_signature",
    ]
    frames = [pq.ParquetFile(f).read(columns=columns).to_pandas() for f in files]
    return pd.concat(frames, ignore_index=True)


def run_validation(model_matrix_root: Path, label_policy_root: Path) -> bool:
    checks: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))

    print("\nPhase-4 pre-training validation:")

    control_root = phase3.control_matrix_root(model_matrix_root)
    acd_root = phase3.acd_matrix_root(model_matrix_root)

    key_cols = [
        "source_id", "window_end_datetime", "prediction_time", "trading_date",
        "split", "target_return_bps", "target_direction",
    ]
    control_seq = _load_sequence_frame(control_root, key_cols)
    acd_seq = _load_sequence_frame(acd_root, key_cols)

    check(
        "1. CONTROL/ACD sample counts match Phase-3 frozen total",
        len(control_seq) == len(acd_seq) == 1_259_396,
        f"control={len(control_seq)} acd={len(acd_seq)}",
    )

    labels = _load_label_policy_frame(label_policy_root)
    check(
        "2. label rows align to the shared sequence universe",
        len(labels) == len(control_seq),
        f"labels={len(labels)} sequences={len(control_seq)}",
    )
    check(
        "3. policy name/signature is single-valued and frozen",
        labels["policy_name"].nunique() == 1
        and labels["policy_name"].iloc[0] == POLICY_NAME
        and labels["policy_signature"].nunique() == 1,
    )
    check(
        "4. no pre-2020-01-03 samples in labels",
        pd.to_datetime(labels["trading_date"]).min().date() >= COMMON_MODELING_START,
        f"min_trading_date={pd.to_datetime(labels['trading_date']).min().date()}",
    )

    label_keys = labels[["source_id", "prediction_time", "split", "label_direction"]].copy()
    label_keys["source_id"] = label_keys["source_id"].astype(str).str.strip().str.lower()

    control_keys = control_seq[["source_id", "prediction_time", "split"]].copy()
    control_keys["source_id"] = control_keys["source_id"].astype(str).str.strip().str.lower()
    aligned = control_keys.merge(
        label_keys, on=["source_id", "prediction_time"], how="left",
        suffixes=("_seq", "_label"), validate="one_to_one",
    )
    check(
        "5. every CONTROL sequence has a matching label row",
        aligned["label_direction"].notna().all(),
        f"unmatched={int(aligned['label_direction'].isna().sum())}",
    )
    check(
        "6. label split assignment matches sequence split assignment",
        (aligned["split_seq"] == aligned["split_label"]).all(),
    )

    acd_keys = acd_seq[["source_id", "prediction_time"]].copy()
    acd_keys["source_id"] = acd_keys["source_id"].astype(str).str.strip().str.lower()
    acd_aligned = acd_keys.merge(
        label_keys[["source_id", "prediction_time", "label_direction"]],
        on=["source_id", "prediction_time"], how="left", validate="one_to_one",
    ).rename(columns={"label_direction": "acd_label"})
    both = aligned.merge(
        acd_aligned, on=["source_id", "prediction_time"], validate="one_to_one",
    )
    check(
        "7. CONTROL and ACD resolve to identical labels per shared key",
        (both["label_direction"] == both["acd_label"]).all(),
    )

    test_count = int((control_seq["split"] == "test").sum())
    check(
        "8. TEST row count present but not opened for values",
        test_count > 0,
        f"test_count={test_count}",
    )

    passed = all(ok for _, ok, _ in checks)
    print(f"\nOverall: {'PASS' if passed else 'FAIL'} ({sum(ok for _, ok, _ in checks)}/{len(checks)} checks)")
    return passed


# ---------------------------------------------------------------------------
# Compare (validation-only metrics; TEST never opened)
# ---------------------------------------------------------------------------

def _run_root(models_root: Path, feature_set: str, run_id: str) -> Path:
    return (
        models_root
        / SEQUENCE_SHAPE
        / f"feature_set={feature_set}"
        / "model=lstm"
        / f"symbol={SYMBOL}"
        / f"run_id={run_id}"
    )


def run_compare(models_root: Path, training_reports_root: Path) -> dict:
    control_report_path = _run_root(models_root, "core_v1", CONTROL_RUN_ID) / "training_report.json"
    acd_report_path = _run_root(models_root, "core_v1_acd_v1", ACD_RUN_ID) / "training_report.json"

    with open(control_report_path, encoding="utf-8") as f:
        control_report = json.load(f)
    with open(acd_report_path, encoding="utf-8") as f:
        acd_report = json.load(f)

    for key in (
        "train_samples_used", "validation_samples_used",
        "held_out_test_samples_not_evaluated",
    ):
        if control_report[key] != acd_report[key]:
            raise ValueError(
                f"Sample-identity check failed on {key}: "
                f"control={control_report[key]} acd={acd_report[key]}"
            )

    control_metrics = control_report["best_validation_metrics"]
    acd_metrics = acd_report["best_validation_metrics"]

    deltas = {
        "accuracy": acd_metrics["accuracy"] - control_metrics["accuracy"],
        "balanced_accuracy": acd_metrics["balanced_accuracy"] - control_metrics["balanced_accuracy"],
        "macro_f1": acd_metrics["macro_f1"] - control_metrics["macro_f1"],
        "per_class": {
            name: {
                metric: acd_metrics["per_class"][name][metric]
                - control_metrics["per_class"][name][metric]
                for metric in ("precision", "recall", "f1")
            }
            for name in control_metrics["per_class"]
        },
    }

    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "control_run_id": CONTROL_RUN_ID,
        "acd_run_id": ACD_RUN_ID,
        "control_report": str(control_report_path),
        "acd_report": str(acd_report_path),
        "sample_counts": {
            "train_samples_used": control_report["train_samples_used"],
            "validation_samples_used": control_report["validation_samples_used"],
            "held_out_test_samples_not_evaluated": control_report["held_out_test_samples_not_evaluated"],
        },
        "control_best_epoch": control_report["best_epoch"],
        "acd_best_epoch": acd_report["best_epoch"],
        "control_validation_metrics": control_metrics,
        "acd_validation_metrics": acd_metrics,
        "acd_minus_control_deltas": deltas,
        "test_evaluated": False,
    }

    training_reports_root.mkdir(parents=True, exist_ok=True)
    out_path = (
        training_reports_root / SYMBOL
        / f"phase4_comparison_{CONTROL_RUN_ID}_vs_{ACD_RUN_ID}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison written: {out_path}")
    print(f"ACD - CONTROL: accuracy={deltas['accuracy']:+.4f} "
          f"balanced_accuracy={deltas['balanced_accuracy']:+.4f} "
          f"macro_f1={deltas['macro_f1']:+.4f}")
    print(f"TEST evaluated: {comparison['test_evaluated']}")

    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "labels", "preflight", "train-control", "train-acd",
            "compare", "validate", "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required in addition to --mode train-control/train-acd.",
    )
    parser.add_argument("--model-matrix-root", type=Path, default=DEFAULT_MODEL_MATRIX_ROOT,
                         help="Phase-3 output (read-only input to Phase 4).")
    parser.add_argument("--encoded-features-root", type=Path, default=DEFAULT_ENCODED_FEATURES_ROOT,
                         help="Phase-3 output (read-only input to Phase 4).")
    parser.add_argument("--label-policy-root", type=Path, default=DEFAULT_LABEL_POLICY_ROOT)
    parser.add_argument("--label-policy-reports-root", type=Path, default=DEFAULT_LABEL_POLICY_REPORTS_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--training-reports-root", type=Path, default=DEFAULT_TRAINING_REPORTS_ROOT)
    parser.add_argument("--generated-config-dir", type=Path, default=DEFAULT_GENERATED_CONFIG_DIR)
    args = parser.parse_args()

    control_cfg = args.generated_config_dir / "pipeline_phase4_control.yaml"
    acd_cfg = args.generated_config_dir / "pipeline_phase4_acd.yaml"

    if args.mode in ("labels", "all"):
        control_cfg, acd_cfg = generate_variant_configs(
            args.encoded_features_root, args.generated_config_dir
        )
        build_labels(args.model_matrix_root, args.label_policy_root, args.label_policy_reports_root)

    if args.mode in ("preflight", "all"):
        if not (control_cfg.exists() and acd_cfg.exists()):
            control_cfg, acd_cfg = generate_variant_configs(
                args.encoded_features_root, args.generated_config_dir
            )
        run_preflight(args.model_matrix_root, args.label_policy_root, control_cfg, acd_cfg)

    if args.mode == "train-control":
        run_train("control", args.confirm_training, control_cfg, acd_cfg,
                   args.model_matrix_root, args.label_policy_root,
                   args.models_root, args.training_reports_root)

    if args.mode == "train-acd":
        run_train("acd", args.confirm_training, control_cfg, acd_cfg,
                   args.model_matrix_root, args.label_policy_root,
                   args.models_root, args.training_reports_root)

    if args.mode == "compare":
        run_compare(args.models_root, args.training_reports_root)

    passed = True
    if args.mode in ("validate", "all"):
        passed = run_validation(args.model_matrix_root, args.label_policy_root)

    print("\nOutput paths:")
    print(f"  shared labels:     {label_policy_symbol_root(args.label_policy_root)}")
    print(f"  CONTROL config:    {control_cfg}")
    print(f"  ACD config:        {acd_cfg}")
    print(f"  models root:       {args.models_root}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
