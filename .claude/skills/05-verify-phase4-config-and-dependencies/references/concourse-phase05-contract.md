# Concourse Phase-05 Contract

Create `ci/concourse/phase05.yml` and extend the one real
`ci/concourse/pipeline.yml`.

Dependency order:

`phase01 -> phase02 -> phase03 -> phase04 -> phase05`

Phase 05 is validation-only.

It should invoke the persistent Phase-5 Python audit and any small targeted
tests needed to prove configuration/dependency integrity.

It must not:

- train models;
- call Phase-4 `train-control` or `train-acd`;
- inspect TEST predictions/performance;
- contain core audit logic directly in YAML;
- assume the developer's Conda environment.

Claude should validate YAML syntax and wiring only. Do not run the real
Concourse pipeline during Skill 05.
