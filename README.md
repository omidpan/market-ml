# market_ml v1.1
#<b> for more Info you can go to docs</b>
Clean historical market-data foundation for 1-minute OHLCV.

This version intentionally stops **before feature engineering and ML**.
Its job is to make the historical source trustworthy, auditable,
appendable and ready for later multi-timeframe processing.

## Core design

Three concepts stay separate:

1. **market session**
2. **data source**
3. **data-quality / imputation status**

This separation is important because a future overnight provider may have a
different Big Bang date, different quality, different venue rules and
different volume semantics from the current extended-hours feed.

## Market sessions

Timestamps are interpreted in `America/New_York`.

The current 1-minute bars are treated as **bar-start timestamps**.

| Session | Bar-start times |
|---|---|
| `premarket` | 04:00-09:29 |
| `regular` | 09:30-15:59 |
| `aftermarket` | 16:00-19:59 |
| `overnight` | 20:00-03:59 |

For the current source:

```text
premarket   = expected
regular     = expected
aftermarket = expected
overnight   = NOT expected
```

So the validator will never create fake "missing overnight" records for
historical data that never claimed overnight coverage.

## Source provenance

`config/sources.csv` defines what each provider actually supplies.

Current AVGO example:

```text
source_id=primary_extended
coverage_start=2019-05-06 04:00
coverage_sessions=premarket|regular|aftermarket
```

Later an overnight feed can be registered independently:

```text
source_id=overnight_provider
coverage_start=<new provider Big Bang>
coverage_sessions=overnight
```

The raw datasets remain separate.

## Logical trading date

The canonical schema includes both:

```text
datetime
trading_date
```

For current data:

```text
2026-08-11 08:00
    session=premarket
    trading_date=2026-08-11
```

For future overnight data, the intended mapping is:

```text
Sunday 22:00 -> Monday trading_date
Monday 02:00 -> Monday trading_date
```

The code uses the exchange calendar rather than blindly adding one calendar
day, so weekends and holidays can be handled.

## Canonical metadata

Regularized rows include:

```text
symbol
datetime
trading_date
session
source_id

open
high
low
close
volume

is_observed
is_imputed
quality_status
imputation_source_datetime
```

This makes a real zero-volume candle distinguishable from a generated one.

Real source row:

```text
volume=0
is_observed=True
is_imputed=False
```

Synthetic row:

```text
volume=0
is_observed=False
is_imputed=True
```

## Missing-data policy

No backward fill exists.

Default behavior:

```text
small internal gap <= 5 minutes -> eligible for causal fill
leading session gap             -> keep missing
trailing session gap            -> keep missing
fully absent session            -> keep missing
```

When an internal gap is filled:

```text
open  = previous observed close
high  = previous observed close
low   = previous observed close
close = previous observed close
volume = 0
```

The fill source must come from an **earlier observed candle in the same named
session**.

So by default the pipeline will not carry:

```text
premarket -> regular
regular -> aftermarket
previous day -> next premarket
```

across a missing boundary.

## Why session-aware validation matters

Instead of reporting:

```text
missing today = 50
```

the validator can report:

```text
premarket missing   = 50
regular missing     = 0
aftermarket missing = 0
```

This gives much more useful information about data quality.

## Project structure

```text
market_ml_v1/
├── VERSION
├── README.md
├── ARCHITECTURE.md
├── SAMPLE_FINDINGS.md
├── CHANGELOG.md
├── requirements.txt
│
├── config/
│   ├── instruments.csv
│   ├── sources.csv
│   └── pipeline.yaml
│
├── data/
│   ├── raw/
│   │   ├── bootstrap/
│   │   │   └── AVGO/
│   │   └── incoming/
│   │       └── primary_extended/
│   │           └── AVGO/
│   │               └── 1m/
│   │
│   ├── staging/
│   │   └── AVGO/
│   │
│   └── parquet/
│       ├── ohlcv_1m_observed/
│       └── ohlcv_1m_regularized/
│
├── reports/
│   ├── validation/
│   └── regularization/
│
├── state/
│   └── scraper_manifest.jsonl
│
├── src/
│   ├── common.py
│   ├── validate.py
│   ├── regularize.py
│   ├── parquet_store.py
│   ├── ingest.py
│   └── scraper/
│       ├── manifest.py
│       └── README.md
│
├── tests/
└── experiments/
```

## Validation command

```bash
python src/validate.py   --input data/raw/bootstrap/AVGO/AVGO_1m_original.csv   --symbol AVGO   --source-id primary_extended
```

Validation produces:

```text
summary.json
daily_coverage.csv
session_coverage.csv
missing_bars.csv
gaps.csv
duplicates.csv
row_issues.csv
rows_outside_source_claimed_sessions.csv
```

## Regularization command

```bash
python src/regularize.py   --input data/raw/bootstrap/AVGO/AVGO_1m_original.csv   --output data/staging/AVGO/AVGO_1m_regularized.csv   --symbol AVGO   --source-id primary_extended
```

## Parquet command

```bash
python src/parquet_store.py   --input data/staging/AVGO/AVGO_1m_regularized.csv   --symbol AVGO   --root data/parquet/ohlcv_1m_regularized
You also need .env file to give project path, data path and config path
you need to give to the shell if you want to execute scripts
please if you create .env file another location rather than project root you must execute the following to give the path file to the scripts.   ENV_FILE=/private/path/.env
```

Physical Parquet layout:

```text
symbol=AVGO/
└── year=2019/
    └── month=05/
        └── part.parquet
```

`session` and `source_id` remain columns instead of extra folders. This avoids
creating too many small files.

## Live refresh flow

```text
scraper
   ↓
raw/incoming/<SOURCE_ID>/<SYMBOL>/1m/
   ↓
scraper manifest
   ↓
source-aware validation
   ↓
merge/de-duplicate
   ↓
rewrite affected monthly Parquet partition
```

The scraper never writes directly to the canonical historical store.



