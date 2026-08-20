"""Focused tests for the Phase 5 v2 gates (D/E/F) and overall-status rollup
in scripts/run_phase5_integrity_audit.py. Uses synthetic fixtures only —
never reads real repository Parquet/TEST data."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_phase5_integrity_audit as audit  # noqa: E402


def test_v2_wiring_audit_passes_against_real_code():
    # Gate D is purely static/code-level (no Parquet reads), so it can run
    # against the actual committed scripts/config in any environment.
    result = audit.run_v2_wiring_audit()
    assert result["gate"] == "v2_wiring"
    assert result["status"] == "pass", result["checks"]
    assert result["passed"] is True
    assert result["control_feature_count"] == 23
    assert result["acd_feature_count"] == 150


def test_v1_provenance_audit_is_path_exists_only(monkeypatch):
    # Every root reported "present" must correspond to Path.exists() truth,
    # and the function must never read file contents — patch every
    # content-reading primitive to explode if ever invoked.
    def _explode(*args, **kwargs):
        raise AssertionError("v1provenance must never read file contents")

    monkeypatch.setattr(Path, "read_bytes", _explode)
    monkeypatch.setattr(Path, "read_text", _explode)
    monkeypatch.setattr(Path, "open", _explode)
    monkeypatch.setattr("builtins.open", _explode)

    result = audit.run_v1_provenance_audit()
    assert result["gate"] == "v1_provenance_untouched"
    for rel_path, exists in result["checks"].items():
        assert exists == (REPO_ROOT / rel_path).exists()
    # For the current repository, all three frozen v1 provenance roots are
    # expected to exist, so the gate status must consistently reflect that.
    assert result["status"] == "pass"
    assert result["passed"] is True


def _patch_missing_v2_roots(monkeypatch, phase3, phase4, tmp_path):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(phase3, "DEFAULT_SEQUENCE_INDEX_STRUCTURAL_ROOT_V2", missing / "a")
    monkeypatch.setattr(phase3, "DEFAULT_SEQUENCE_INDEX_TRAINVAL_ROOT_V2", missing / "b")
    monkeypatch.setattr(phase3, "DEFAULT_MODEL_MATRIX_ROOT_V2", missing / "c")
    monkeypatch.setattr(phase4, "DEFAULT_LABEL_POLICY_ROOT_V2", missing / "d")


def _patch_present_v2_roots(monkeypatch, phase3, phase4, tmp_path):
    """Creates real (empty) directories for every v2 root the existence gate
    checks, so Gate F proceeds past the existence check into calling the
    (monkeypatched) validators — without ever touching real Parquet data."""
    roots = {
        "DEFAULT_SEQUENCE_INDEX_STRUCTURAL_ROOT_V2": tmp_path / "structural",
        "DEFAULT_SEQUENCE_INDEX_TRAINVAL_ROOT_V2": tmp_path / "trainval",
        "DEFAULT_MODEL_MATRIX_ROOT_V2": tmp_path / "model_matrix",
    }
    for attr, path in roots.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(phase3, attr, path)
    label_policy_root = tmp_path / "label_policy"
    label_policy_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(phase4, "DEFAULT_LABEL_POLICY_ROOT_V2", label_policy_root)
    return roots["DEFAULT_MODEL_MATRIX_ROOT_V2"], label_policy_root


def test_v2_data_integrity_reports_blocked_when_artifacts_missing(tmp_path, monkeypatch):
    phase3, phase4 = audit._import_phase_modules()
    _patch_missing_v2_roots(monkeypatch, phase3, phase4, tmp_path)

    result = audit.run_v2_data_integrity_audit()
    assert result["gate"] == "v2_data_integrity"
    assert result["status"] == "blocked"
    assert result["passed"] is False
    assert "blocked_reason" in result


def test_v2_data_integrity_passes_when_both_validators_pass(tmp_path, monkeypatch):
    phase3, phase4 = audit._import_phase_modules()
    _patch_present_v2_roots(monkeypatch, phase3, phase4, tmp_path)
    monkeypatch.setattr(phase3, "run_validation_v2", lambda *a, **k: True)
    monkeypatch.setattr(phase4, "run_validation_v2", lambda *a, **k: True)

    result = audit.run_v2_data_integrity_audit()
    assert result["status"] == "pass"
    assert result["passed"] is True
    assert result["phase3_validation_v2_passed"] is True
    assert result["phase4_validation_v2_passed"] is True


def test_v2_data_integrity_fails_when_either_validator_fails(tmp_path, monkeypatch):
    phase3, phase4 = audit._import_phase_modules()

    _patch_present_v2_roots(monkeypatch, phase3, phase4, tmp_path)
    monkeypatch.setattr(phase3, "run_validation_v2", lambda *a, **k: False)
    monkeypatch.setattr(phase4, "run_validation_v2", lambda *a, **k: True)
    result = audit.run_v2_data_integrity_audit()
    assert result["status"] == "fail"
    assert result["passed"] is False

    _patch_present_v2_roots(monkeypatch, phase3, phase4, tmp_path)
    monkeypatch.setattr(phase3, "run_validation_v2", lambda *a, **k: True)
    monkeypatch.setattr(phase4, "run_validation_v2", lambda *a, **k: False)
    result = audit.run_v2_data_integrity_audit()
    assert result["status"] == "fail"
    assert result["passed"] is False


def test_v2_data_integrity_reports_error_when_validator_raises(tmp_path, monkeypatch):
    phase3, phase4 = audit._import_phase_modules()
    _patch_present_v2_roots(monkeypatch, phase3, phase4, tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("synthetic validator failure")

    # Isolate the synthetic Phase 3 exception: Phase 4 must PASS so the
    # error can only be attributed to the patched Phase 3 call.
    monkeypatch.setattr(phase4, "run_validation_v2", lambda *a, **k: True)
    monkeypatch.setattr(phase3, "run_validation_v2", _boom)

    result = audit.run_v2_data_integrity_audit()
    assert result["status"] == "error"
    assert result["passed"] is False
    assert "synthetic validator failure" in result["error"]


def test_rollup_precedence_error_beats_blocked_beats_fail_beats_pass():
    def gate(status: str) -> dict:
        return {"passed": status == "pass", "status": status}

    assert audit._rollup_gate_status([gate("pass"), gate("pass")]) == "pass"
    assert audit._rollup_gate_status([gate("pass"), gate("fail")]) == "fail"
    assert audit._rollup_gate_status([gate("pass"), gate("blocked")]) == "blocked"
    assert audit._rollup_gate_status([gate("fail"), gate("blocked")]) == "blocked"
    assert audit._rollup_gate_status([gate("error"), gate("blocked")]) == "error"


def test_run_compare_known_issue_is_informational_only():
    result = audit.check_run_compare_v2_known_issue()
    assert result["phase5_blocker"] is False
    assert "known_issue" in result
