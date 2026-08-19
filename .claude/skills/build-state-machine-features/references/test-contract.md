# Phase 2 Test Contract

Run targeted tests only.

## Required tests

### 1. Source key integrity
For representative inputs:
- no duplicate base keys;
- no duplicate ACD keys after filtering by config/policy;
- expected join cardinality is one-to-one for 1-minute sources.

### 2. Source identity
Compare ACD source OHLCV against authoritative market-ml OHLCV on matched timestamps.

Report:
- matched count
- unmatched count
- duplicate count
- OHLCV mismatch counts
- numeric tolerance used

Fail loudly if the approved identity rule is violated.

### 3. Timestamp and DST
Spot-check:
- winter date
- summer date
- DST transition vicinity where available

Confirm UTC -> America/New_York mapping and bar-start semantics.

### 4. Same-session environment availability
Verify:
- no previous-day environment leakage;
- rows before `or_finalized_at` are unavailable;
- rows at/after availability receive only same-session values.

### 5. Warm-up invariance
Verify opening rows remain present.

ACD unavailable rows must retain the same base timestamps.

### 6. Row-universe invariance
For the common universe:
- baseline/control row count == ACD-enhanced row count;
- timestamps identical and in identical order;
- ACD availability does not remove rows.

### 7. Leakage exclusions
Assert forbidden columns are absent from produced feature schema.

### 8. Provenance
Assert required configuration/policy/schema identifiers are present in metadata/output provenance.

### 9. Sequence rebuild readiness
Verify the produced combined feature table is consumable by the existing sequence/model-matrix path without reusing old sequences.

## Do not run
- full LSTM training
- multi-config grid
- correlation study
- feature importance
- backtest
