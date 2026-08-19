#!/usr/bin/env python3
"""
Milestone 1 Phase 1 — static contract regression check.

Verifies config/pipeline.yaml's `pipeline.state_machine_features` block still
matches the approved Phase-1 integration identity (config_id, regime_config_id,
regime policy, feature-set identity, common modeling start). This is a
read-only static check — no Parquet is read, no training runs. It exists so
Concourse Phase 01 can call a real, versioned Python entry point instead of
embedding assertions in YAML.

Exit code 0 = PASS, non-zero = FAIL (drift from the approved contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"

# Approved Milestone 1 Phase 1 integration identity. Frozen by the
# design/verify skills; changing any of these values is a scope change that
# requires re-running Phase 1 design + verification, not a silent CI edit.
APPROVED = {
    "config_id": "nvda__or5__atr14__a010__c020__rules-v2",
    "regime_config_id": (
        "nvda__or5__atr14__a010__c020__rules-v2__env-e1__thr-quantile-frozen-6ee45849cb"
        "__regime-moderate_or_anchored-r1__score-r1"
    ),
    "regime_policy_name": "moderate_or_anchored",
    "regime_policy_version": "r1",
    "feature_set_identity": "core_v1_acd_v1",
    "common_modeling_start": "2020-01-03",
}


def main() -> int:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = raw.get("pipeline", {}).get("state_machine_features", {})

    failures = []
    for key, expected in APPROVED.items():
        actual = str(cfg.get(key, "")).strip()
        if actual != expected:
            failures.append(f"  {key}: expected {expected!r}, got {actual!r}")

    if failures:
        print("Phase 1 static contract check: FAIL")
        print("\n".join(failures))
        return 1

    print("Phase 1 static contract check: PASS")
    for key, expected in APPROVED.items():
        print(f"  {key} = {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
