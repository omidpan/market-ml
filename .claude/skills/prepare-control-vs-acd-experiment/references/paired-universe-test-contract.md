# Paired-Universe Validation Contract

Before any training, compare CONTROL vs ACD artifacts.

## Required PASS checks

1. Same sequence count overall.
2. Same sequence count by train/validation/test split.
3. Same ordered sequence keys.
4. Same:
   - symbol
   - window_start_datetime
   - window_end_datetime
   - prediction_time
   - target_realized_at
   - trading_date
   - session
   - split
5. Same target values and target availability.
6. Same purge/embargo exclusions.
7. No sequence crosses invalid session boundaries.
8. No pre-2020-01-03 sample appears.
9. CONTROL contains zero ACD feature columns.
10. ACD contains the approved incremental ACD columns.
11. No forbidden ACD columns appear.
12. Any categorical encoding vocabulary/scaler is fit on TRAIN only and then applied unchanged to validation/test.
13. No NaN/inf enters the final numeric model matrix unless current pipeline explicitly supports it.
14. Feature-column order is deterministic and recorded.

## Failure rule

If any sample identity differs between CONTROL and ACD, stop.
Do not train until the difference is explained and removed.
