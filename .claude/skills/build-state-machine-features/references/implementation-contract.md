# Phase 2 Implementation Contract

## Base universe
The authoritative base is the market-ml feature timestamp universe.

Use a LEFT JOIN from market-ml to ACD/state-machine data.

ACD availability may change feature values and availability flags, but must never decide whether a base row exists.

## Common start
For the first controlled NVDA integration:

`2020-01-03`

## Timestamp convention
- ACD timestamps: UTC, bar-start.
- Convert to the authoritative market-ml timezone/timestamp convention before joining.
- Feature availability must reflect when the completed source bar is observable under market-ml conventions.

## Same-session environment rule
Daily environment values:
- join only to the same `session_date/trading_date`;
- are unavailable before that session's `or_finalized_at`;
- must never carry forward from a prior session.

## Warm-up
Opening-range rows remain in the dataset.

Use explicit availability flags, e.g.:
- `sm_available`
- group-level availability flags where useful

When unavailable:
- numeric values use the approved neutral sentinel;
- categorical values use reserved `unavailable`;
- do not drop the row.

## Feature-set identity
Never overwrite `core_v1`.

Create a new versioned feature-set identity for the combined market-ml + ACD feature table, e.g.:

`core_v1_acd_v1`

The exact identity must be configurable and recorded in metadata.

## Provenance
At minimum record:
- symbol
- signal_config_id / config_id
- regime_config_id
- regime policy name/version
- environment schema version
- category method
- category threshold set id
- state-machine feature schema version
- market-ml feature-set identity
- build timestamp / dataset identity if the project already uses those fields

## First implementation policy
Use exactly one explicitly named regime policy in a given build.

Do not concatenate the 3 policies into one feature vector in the first experiment.

## Safe-first feature policy
Implement only the Phase 1 verified-safe whitelist.

Unresolved fields stay deferred.

Do not infer safety from field names.
