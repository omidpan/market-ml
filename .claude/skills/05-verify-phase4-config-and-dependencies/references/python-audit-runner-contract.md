# Phase-5 Python Audit Runner Contract

Preferred entry point:

`scripts/run_phase5_integrity_audit.py`

Recommended modes:

- `config`
- `dependencies`
- `concourse`
- `all`

The runner should be deterministic and read-only with respect to datasets and
models.

It should write a Phase-5-owned report, for example:

`reports/phase5_integrity/phase5_integrity_report.json`

The report should include:

- semantic config diffs;
- invariant checks;
- feature counts;
- external dependency inventory;
- requirements coverage;
- Concourse phase dependency result;
- PASS/FAIL summary.

Exit code 0 = all required gates pass.
Non-zero = at least one required gate fails.

Do not rely on `/tmp`, Claude chat history, or manually pasted lists for the
final proof.
