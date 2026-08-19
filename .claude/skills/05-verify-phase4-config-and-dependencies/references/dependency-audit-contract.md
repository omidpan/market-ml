# Dependency Audit Contract

## Rule

Committed dependency files, not the developer's Conda environment, define
reproducibility.

Never use `pip freeze` as the project dependency specification.

## Audit scope

Inspect external imports required by the persistent entry points used by
Phases 01–05 and later Colab training. Include relevant transitive local-module
imports so a dependency imported in `src/*.py` is not missed merely because
the wrapper does not import it directly.

Classify:

- Python standard library
- repository-local modules
- external packages

Only external packages belong in requirements files.

## Preferred separation

If justified:

`requirements.txt`
- core data/runtime dependencies

`requirements-ci.txt`
- includes/references core requirements
- pytest and CI-only validation dependencies

`requirements-train.txt`
- includes/references core requirements
- full model-training dependencies for Colab

Avoid duplicate version constraints across files where `-r` inclusion is
cleaner.

Do not unnecessarily change existing package version ranges.

## Verification

Perform lightweight import/static checks. Do not run full training merely to
prove dependencies.
