# Scraper integration

The scraper is a **source adapter**, not the owner of the canonical
historical dataset.

## Current source

`primary_extended` claims:

- premarket: 04:00-09:29
- regular: 09:30-15:59
- aftermarket: 16:00-19:59

It does **not** claim overnight coverage.

Therefore 20:00-03:59 must never be reported as missing for this source.

## Future overnight provider

If overnight data is obtained later, add a new source in
`config/sources.csv`, for example conceptually:

```text
source_id=overnight_provider
coverage_sessions=overnight
coverage_start=<its own Big Bang>
```

Do not merge raw feeds together.

The canonical layer may combine them later while preserving `source_id`.

## Scraper output path

```text
data/raw/incoming/<SOURCE_ID>/<SYMBOL>/1m/
```

Example:

```text
data/raw/incoming/primary_extended/AVGO/1m/2026-08-09.csv
```

## Scraper must not

- backward-fill
- forward-fill
- calculate ML features
- write directly to canonical Parquet
- silently overwrite finalized source data
- hide the provider identity
