#!/usr/bin/env python3
"""
Milestone 1 Skill 05 — Phase-4 configuration-integrity and dependency audit.

Read-only with respect to Parquet/model artifacts. Does not train, does not
run Concourse, does not inspect TEST values.

    python scripts/run_phase5_integrity_audit.py --mode config
    python scripts/run_phase5_integrity_audit.py --mode dependencies
    python scripts/run_phase5_integrity_audit.py --mode concourse
    python scripts/run_phase5_integrity_audit.py --mode all

Config paths default to the normal local layout (config/pipeline.yaml,
config/generated/pipeline_phase4_control.yaml,
config/generated/pipeline_phase4_acd.yaml) but are fully overridable via
--canonical-config/--control-config/--acd-config so this can also validate
Concourse-persisted copies at a different mounted path (see
ci/concourse/phase05.yml, which passes the Phase-4 generated-config
resource's actual mount point explicitly rather than assuming the local
config/generated layout exists in the container).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANONICAL_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"
DEFAULT_CONTROL_CONFIG = REPO_ROOT / "config" / "generated" / "pipeline_phase4_control.yaml"
DEFAULT_ACD_CONFIG = REPO_ROOT / "config" / "generated" / "pipeline_phase4_acd.yaml"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "phase5_integrity"
DEFAULT_CONCOURSE_PIPELINE = REPO_ROOT / "ci" / "concourse" / "pipeline.yml"

# ---------------------------------------------------------------------------
# Gate A — configuration-integrity
# ---------------------------------------------------------------------------

# Paths (dot-notation) allowed to differ between CONTROL and ACD. Anything
# else that differs is an unapproved diff -> FAIL.
ACD_WHITELISTED_DIFF_PATHS = {
    "pipeline.model_matrix.feature_set",
    "pipeline.model_matrix.features",
}

EXPECTED_CONTROL_FEATURE_COUNT = 23
EXPECTED_ACD_FEATURE_COUNT = 150
EXPECTED_CONTROL_FEATURE_SET = "core_v1"
EXPECTED_ACD_FEATURE_SET = "core_v1_acd_v1"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten(value, prefix: str = "") -> dict:
    """Recursively flattens a nested dict/list into {dot.path: value}."""

    flat = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            flat.update(_flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        flat[prefix] = value  # compare lists as whole values, not element-wise
    else:
        flat[prefix] = value
    return flat


def _semantic_diff(a: dict, b: dict) -> dict:
    """{path: {"a": ..., "b": ...}} for every path where a and b differ,
    including paths present in only one side (value reported as MISSING)."""

    flat_a = _flatten(a)
    flat_b = _flatten(b)
    paths = set(flat_a) | set(flat_b)
    diff = {}
    for path in paths:
        va = flat_a.get(path, "<<MISSING>>")
        vb = flat_b.get(path, "<<MISSING>>")
        if va != vb:
            diff[path] = {"a": va, "b": vb}
    return diff


def run_config_audit(canonical_path: Path, control_path: Path, acd_path: Path) -> dict:
    canonical = _load_yaml(canonical_path)
    control = _load_yaml(control_path)
    acd = _load_yaml(acd_path)

    canonical_to_control = _semantic_diff(canonical, control)
    control_to_acd = _semantic_diff(control, acd)

    unapproved_canonical_control = dict(canonical_to_control)  # zero paths whitelisted
    unapproved_acd = {
        path: d for path, d in control_to_acd.items()
        if path not in ACD_WHITELISTED_DIFF_PATHS
    }

    control_features = (
        control.get("pipeline", {}).get("model_matrix", {}).get("features", [])
    )
    acd_features = acd.get("pipeline", {}).get("model_matrix", {}).get("features", [])
    control_feature_set = control.get("pipeline", {}).get("model_matrix", {}).get("feature_set")
    acd_feature_set = acd.get("pipeline", {}).get("model_matrix", {}).get("feature_set")

    checks = {
        "canonical_to_control_zero_unapproved_diff": not unapproved_canonical_control,
        "control_to_acd_only_whitelisted_diff": not unapproved_acd,
        "control_feature_set_is_core_v1": control_feature_set == EXPECTED_CONTROL_FEATURE_SET,
        "acd_feature_set_is_core_v1_acd_v1": acd_feature_set == EXPECTED_ACD_FEATURE_SET,
        "control_feature_count_is_23": len(control_features) == EXPECTED_CONTROL_FEATURE_COUNT,
        "acd_feature_count_is_150": len(acd_features) == EXPECTED_ACD_FEATURE_COUNT,
        "control_features_subset_of_acd": set(control_features) <= set(acd_features),
    }

    passed = all(checks.values())

    return {
        "gate": "config",
        "passed": passed,
        "checks": checks,
        "canonical_to_control_diff": canonical_to_control,
        "control_to_acd_diff": control_to_acd,
        "unapproved_canonical_to_control_diff": unapproved_canonical_control,
        "unapproved_control_to_acd_diff": unapproved_acd,
        "control_feature_count": len(control_features),
        "acd_feature_count": len(acd_features),
        "paths": {
            "canonical": str(canonical_path),
            "control": str(control_path),
            "acd": str(acd_path),
        },
    }


# ---------------------------------------------------------------------------
# Gate B — dependency audit
# ---------------------------------------------------------------------------

# The fixed, manually-traced import closure for Phases 01-05 + Colab
# training (see Skill-05 Step-1 plan). Not auto-discovered: the audit's job
# is to verify committed requirements files cover THIS known closure, not to
# rediscover the closure itself each run.
CORE_ENTRY_FILES = [
    "scripts/validate_phase1_contract.py",
    "scripts/run_phase2_state_machine_features.py",
    "scripts/run_phase3_control_vs_acd.py",
    "scripts/run_phase4_control_vs_acd.py",
    "scripts/run_phase5_integrity_audit.py",
    "src/state_machine_features.py",
    "src/sequences.py",
    "src/model_matrix.py",
    "src/build_label_policy.py",
    "src/train_model.py",
    "src/features.py",
    "src/common.py",
]
CI_ENTRY_FILES = [
    "tests/test_state_machine_features.py",
    "tests/test_session_labels.py",
]

TRAIN_ONLY_PACKAGES = {"tensorflow", "keras"}

# import-name -> pip-distribution-name, only where they differ or where
# normalization (hyphen/underscore/case) could otherwise mask a match.
IMPORT_TO_PACKAGE = {
    "yaml": "pyyaml",
    "exchange_calendars": "exchange-calendars",
}

STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", ()))
# Fallback for Python versions without sys.stdlib_module_names (<3.10).
STDLIB_MODULES |= {
    "argparse", "ast", "dataclasses", "datetime", "functools", "glob",
    "hashlib", "json", "math", "os", "pathlib", "re", "subprocess", "sys",
    "time", "typing", "warnings", "copy", "__future__",
}


def _local_module_stems() -> set:
    stems = set()
    for pattern in ("src/*.py", "scripts/*.py"):
        for path in REPO_ROOT.glob(pattern):
            stems.add(path.stem)
    return stems


def _top_level_imports(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def _classify_imports(path: Path, local_stems: set) -> dict:
    imports = _top_level_imports(path)
    external = set()
    for name in imports:
        if name in STDLIB_MODULES or name == "__future__":
            continue
        if name in local_stems:
            continue
        external.add(IMPORT_TO_PACKAGE.get(name, name))
    return {"file": str(path.relative_to(REPO_ROOT)), "external": sorted(external)}


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_requirements(path: Path, seen: set | None = None) -> set:
    """Recursively resolves `-r other.txt` includes; returns normalized
    package names (version specifiers stripped)."""

    if seen is None:
        seen = set()
    if path in seen or not path.exists():
        return set()
    seen.add(path)

    packages = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            packages |= _parse_requirements(path.parent / line[3:].strip(), seen)
            continue
        pkg = line
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";"):
            pkg = pkg.split(sep, 1)[0]
        packages.add(_normalize_package_name(pkg))
    return packages


def run_dependency_audit(
    requirements_path: Path,
    requirements_ci_path: Path,
    requirements_train_path: Path,
) -> dict:
    local_stems = _local_module_stems()

    missing_entry_files = []

    core_packages: set = set()
    per_file = []
    for rel in CORE_ENTRY_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            missing_entry_files.append(rel)
            continue
        result = _classify_imports(path, local_stems)
        per_file.append(result)
        core_packages |= set(result["external"])

    ci_packages: set = set()
    for rel in CI_ENTRY_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            missing_entry_files.append(rel)
            continue
        result = _classify_imports(path, local_stems)
        per_file.append(result)
        ci_packages |= set(result["external"])

    core_required = {
        _normalize_package_name(p) for p in core_packages if p not in TRAIN_ONLY_PACKAGES
    }
    train_only_found = {
        _normalize_package_name(p) for p in core_packages if p in TRAIN_ONLY_PACKAGES
    }
    ci_required = {_normalize_package_name(p) for p in ci_packages}

    committed_core = _parse_requirements(requirements_path)
    committed_ci = _parse_requirements(requirements_ci_path)
    committed_train = _parse_requirements(requirements_train_path)

    missing_from_core = sorted(core_required - committed_core)
    missing_from_ci = sorted((ci_required | core_required) - committed_ci)
    missing_train_only = sorted(train_only_found - committed_train)
    # tensorflow/keras must NOT be required by the core (non-training) path.
    train_only_leaked_into_core = sorted(train_only_found & committed_core)
    # requirements-train.txt must cover the full training closure: everything
    # the core pipeline needs (via -r requirements.txt) PLUS the training-only
    # packages — not just the training-only packages in isolation.
    missing_from_train_full = sorted((core_required | train_only_found) - committed_train)

    checks = {
        "all_core_entry_files_exist": not missing_entry_files,
        "core_requirements_cover_core_imports": not missing_from_core,
        "ci_requirements_cover_ci_and_core_imports": not missing_from_ci,
        "train_requirements_cover_training_only_imports": not missing_train_only,
        "train_requirements_cover_core_plus_training_imports": not missing_from_train_full,
        "train_only_packages_not_required_by_core": not train_only_leaked_into_core,
        "no_pip_freeze_style_file_present": not (REPO_ROOT / "requirements-freeze.txt").exists(),
    }
    passed = all(checks.values())

    return {
        "gate": "dependencies",
        "passed": passed,
        "checks": checks,
        "missing_entry_files": missing_entry_files,
        "per_file_external_imports": per_file,
        "core_required_packages": sorted(core_required),
        "ci_required_packages": sorted(ci_required),
        "train_only_packages_found": sorted(train_only_found),
        "missing_from_requirements_train_txt_full": missing_from_train_full,
        "missing_from_requirements_txt": missing_from_core,
        "missing_from_requirements_ci_txt": missing_from_ci,
        "missing_from_requirements_train_txt": missing_train_only,
        "train_only_leaked_into_core_requirements": train_only_leaked_into_core,
    }


# ---------------------------------------------------------------------------
# Gate C — Concourse pipeline/task wiring + requirements-file wiring
# ---------------------------------------------------------------------------

EXPECTED_JOB_CHAIN = [
    ("phase01-contract-validation", []),
    ("phase02-build-state-machine-features", ["phase01-contract-validation"]),
    ("phase03-prepare-control-vs-acd", ["phase02-build-state-machine-features"]),
    ("phase04-prepare-control-vs-acd", ["phase03-prepare-control-vs-acd"]),
    ("phase05-verify-config-and-dependencies", ["phase04-prepare-control-vs-acd"]),
]

# Task outputs deliberately left unmapped/not persisted by a `put`. Any other
# unmapped task output is a wiring gap, not an intentional exception.
INTENTIONALLY_NON_PERSISTED_OUTPUTS = {
    ("phase03-prepare-control-vs-acd", "targets-1m-phase3-view-out"),
}


def _audit_pipeline_wiring(pipeline_path: Path) -> dict:
    ok = True
    detail = []

    if not pipeline_path.exists():
        return {"ok": False, "detail": [f"{pipeline_path} does not exist"]}

    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or {}
    resources = {r["name"] for r in pipeline.get("resources", [])}
    jobs = {j["name"]: j for j in pipeline.get("jobs", [])}

    expected_names = [name for name, _ in EXPECTED_JOB_CHAIN]
    missing_jobs = [name for name in expected_names if name not in jobs]
    if missing_jobs:
        ok = False
        detail.append(f"missing expected jobs: {missing_jobs}")

    # Which job `put`s each resource — derived from the pipeline itself, not
    # hardcoded, so this generalizes to any future job/resource additions.
    resource_producer = {}
    for job_name, job in jobs.items():
        for step in job.get("plan", []):
            if "put" in step:
                resource_producer[step["put"]] = job_name

    for job_name, expected_passed in EXPECTED_JOB_CHAIN:
        if job_name not in jobs:
            continue
        job = jobs[job_name]

        # Job-level: the union of passed constraints across all `get` steps
        # must equal exactly the expected upstream job set (catches extra/
        # wrong dependencies).
        passed_union = {
            p for step in job.get("plan", []) if "get" in step
            for p in step.get("passed", [])
        }
        if passed_union != set(expected_passed):
            ok = False
            detail.append(
                f"{job_name}: passed={sorted(passed_union)} expected={expected_passed}"
            )

        # Per-get: any resource that is itself produced by an upstream job
        # (per resource_producer, derived above) must be individually gated
        # on that producing job's `passed:` — the job-level union check alone
        # cannot catch one correctly-gated get hiding another ungated get of
        # a different upstream-produced resource.
        for step in job.get("plan", []):
            if "get" not in step:
                continue
            resource_name = step["get"]
            producer = resource_producer.get(resource_name)
            if producer is None or producer == job_name:
                continue
            if producer not in step.get("passed", []):
                ok = False
                detail.append(
                    f"{job_name}: get {resource_name!r} is produced by "
                    f"{producer!r} but its own passed: does not include it "
                    f"(passed={step.get('passed', [])})"
                )

    for job_name, job in jobs.items():
        gets_in_job = {s["get"] for s in job.get("plan", []) if "get" in s}
        puts_in_job = {s["put"] for s in job.get("plan", []) if "put" in s}

        for step in job.get("plan", []):
            for key in ("get", "put"):
                if key in step and step[key] not in resources:
                    ok = False
                    detail.append(f"{job_name}: {key} references undeclared resource {step[key]!r}")

            if "task" in step:
                task_rel = step["file"].split("/", 1)[-1] if "/" in step["file"] else step["file"]
                task_path = REPO_ROOT / task_rel
                if not task_path.exists():
                    ok = False
                    detail.append(f"{job_name}: task file does not exist: {task_rel}")
                    continue

                task_cfg = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
                task_inputs = {i["name"] for i in task_cfg.get("inputs", [])}
                task_outputs = {o["name"] for o in task_cfg.get("outputs", [])}
                input_mapping = step.get("input_mapping", {})
                output_mapping = step.get("output_mapping", {})

                if input_mapping and set(input_mapping.keys()) != task_inputs:
                    ok = False
                    detail.append(
                        f"{job_name}: input_mapping keys {sorted(input_mapping)} "
                        f"!= task inputs {sorted(task_inputs)}"
                    )

                # For every declared task input, resolve which resource
                # actually supplies it (mapped name, else the input name
                # itself) and verify THIS job actually has a `get:` step for
                # that resource — a globally declared resource is not
                # enough. market-ml-repo is subject to the same rule.
                for input_name in task_inputs:
                    supplying_resource = input_mapping.get(input_name, input_name)
                    if supplying_resource not in gets_in_job:
                        ok = False
                        detail.append(
                            f"{job_name}: task input {input_name!r} resolves to "
                            f"resource {supplying_resource!r}, but no `get: "
                            f"{supplying_resource}` step exists in this job"
                        )

                # output_mapping keys must themselves be declared task outputs.
                bad_output_keys = set(output_mapping.keys()) - task_outputs
                if bad_output_keys:
                    ok = False
                    detail.append(
                        f"{job_name}: output_mapping keys {sorted(bad_output_keys)} "
                        f"are not declared task outputs {sorted(task_outputs)}"
                    )

                # Every declared task output must be mapped, unless it is on
                # the explicit intentional-exception list — regardless of
                # whether output_mapping is present at all.
                unmapped_outputs = task_outputs - set(output_mapping.keys())
                unexpected_unmapped = {
                    o for o in unmapped_outputs
                    if (job_name, o) not in INTENTIONALLY_NON_PERSISTED_OUTPUTS
                }
                if unexpected_unmapped:
                    ok = False
                    detail.append(
                        f"{job_name}: task outputs {sorted(unexpected_unmapped)} "
                        f"unmapped and not in the intentional-exception list"
                    )

                for out_key, target in output_mapping.items():
                    if target not in resources:
                        ok = False
                        detail.append(f"{job_name}: output_mapping targets undeclared resource {target!r}")
                        continue
                    # Prove persistence, not just declaration: a mapped
                    # output must actually be `put` somewhere in this job.
                    if target not in puts_in_job:
                        ok = False
                        detail.append(
                            f"{job_name}: output_mapping maps {out_key!r} -> {target!r} "
                            f"but no `put: {target}` step exists in this job"
                        )

                for target in input_mapping.values():
                    if target not in resources and target != "market-ml-repo":
                        ok = False
                        detail.append(f"{job_name}: input_mapping targets undeclared resource {target!r}")

    return {"ok": ok, "detail": detail}


def run_concourse_audit(pipeline_path: Path) -> dict:
    ci_dir = REPO_ROOT / "ci" / "concourse"
    findings = []
    all_ok = True

    pipeline_wiring = _audit_pipeline_wiring(pipeline_path)
    all_ok = all_ok and pipeline_wiring["ok"]

    for phase_file in sorted(ci_dir.glob("phase*.yml")):
        text = phase_file.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text) or {}
        run_args = cfg.get("run", {}).get("args", [])
        raw_script = "\n".join(a for a in run_args if isinstance(a, str))
        # Drop bash comment lines (leading `#`, ignoring indentation) so a
        # commented-out `# pytest ...` line isn't mistaken for a real
        # pytest invocation.
        script = "\n".join(
            line for line in raw_script.splitlines()
            if not line.strip().startswith("#")
        )

        pip_targets = re.findall(r"pip install[^\n]*-r\s+([^\s]+)", script)
        needs_pytest = re.search(r"(?<![\w#])pytest\b", script) is not None

        ok = True
        detail = []
        if not pip_targets:
            ok = False
            detail.append("no `pip install -r <file>` line found")
        else:
            for target in pip_targets:
                target_path = REPO_ROOT / target
                if not target_path.exists():
                    ok = False
                    detail.append(f"referenced requirements file does not exist: {target}")
                    continue
                packages = _parse_requirements(target_path)
                if needs_pytest and "pytest" not in packages:
                    ok = False
                    detail.append(
                        f"{phase_file.name} runs pytest but {target} does not provide it"
                    )

        all_ok = all_ok and ok
        findings.append({
            "file": phase_file.name,
            "pip_install_targets": pip_targets,
            "runs_pytest": needs_pytest,
            "ok": ok,
            "detail": detail,
        })

    return {
        "gate": "concourse",
        "passed": all_ok,
        "findings": findings,
        "pipeline_wiring": pipeline_wiring,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["config", "dependencies", "concourse", "all"], default="all")
    parser.add_argument("--canonical-config", type=Path, default=DEFAULT_CANONICAL_CONFIG)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--acd-config", type=Path, default=DEFAULT_ACD_CONFIG)
    parser.add_argument("--requirements", type=Path, default=REPO_ROOT / "requirements.txt")
    parser.add_argument("--requirements-ci", type=Path, default=REPO_ROOT / "requirements-ci.txt")
    parser.add_argument("--requirements-train", type=Path, default=REPO_ROOT / "requirements-train.txt")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--concourse-pipeline", type=Path, default=DEFAULT_CONCOURSE_PIPELINE)
    args = parser.parse_args()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
    }
    gates = []

    if args.mode in ("config", "all"):
        result = run_config_audit(args.canonical_config, args.control_config, args.acd_config)
        gates.append(result)
        print(f"\n[Gate A: config] {'PASS' if result['passed'] else 'FAIL'}")
        for name, ok in result["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    if args.mode in ("dependencies", "all"):
        result = run_dependency_audit(args.requirements, args.requirements_ci, args.requirements_train)
        gates.append(result)
        print(f"\n[Gate B: dependencies] {'PASS' if result['passed'] else 'FAIL'}")
        for name, ok in result["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    if args.mode in ("concourse", "all"):
        result = run_concourse_audit(args.concourse_pipeline)
        gates.append(result)
        print(f"\n[Gate C: concourse] {'PASS' if result['passed'] else 'FAIL'}")
        pipeline_wiring = result["pipeline_wiring"]
        print(f"  [{'PASS' if pipeline_wiring['ok'] else 'FAIL'}] pipeline.yml wiring (01->02->03->04->05)")
        if not pipeline_wiring["ok"]:
            for line in pipeline_wiring["detail"]:
                print(f"      - {line}")
        for finding in result["findings"]:
            status = "PASS" if finding["ok"] else "FAIL"
            print(f"  [{status}] {finding['file']}" + (f" — {finding['detail']}" if finding["detail"] else ""))

    report["gates"] = gates
    passed = all(g["passed"] for g in gates)
    report["overall_passed"] = passed

    args.report_root.mkdir(parents=True, exist_ok=True)
    report_path = args.report_root / "phase5_integrity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {report_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
