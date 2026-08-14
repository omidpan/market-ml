# Architecture

## Historical bootstrap

```text
immutable raw CSV
       |
       v
schema / timestamp validation
       |
       v
source-aware session validation
       |
       +------> reports
       |
       v
observed canonical data
       |
       v
optional conservative regularization
       |
       v
monthly Parquet
```

## Incremental refresh

```text
scraper
   |
   v
raw/incoming/<source>/<symbol>/1m/
   |
   v
hash + provenance manifest
   |
   v
validate only sessions claimed by that source
   |
   v
timestamp merge / de-duplicate
   |
   v
rewrite affected monthly Parquet partition
```

## Session model

```text
20:00 ---------------- 03:59
          overnight
              |
04:00 ---------------- 09:29
          premarket
              |
09:30 ---------------- 15:59
           regular
              |
16:00 ---------------- 19:59
         aftermarket
```

Current historical source covers only:

```text
04:00 ---------------- 19:59
```

Future overnight data may come from a separate provider.

## Source-of-truth layers

### Raw source truth

The datasource exactly as received.

Do not modify it.

### Observed canonical truth

Validated rows that were actually supplied by the source.

### Regularized analytical grid

Clock-aligned data that may contain generated rows.

Every generated row is explicitly marked.

This layer is useful for time-series continuity but is never confused with
the raw source.

## ML boundary

Feature engineering must consume only data whose availability timestamp is
less than or equal to the model decision time.

That rule will later apply to:

- 5m / 15m / 30m / 1h / 4h resampling
- SMH / SPY contextual features
- rolling indicators
- completed higher-timeframe candles
- feature normalization
- targets

Feature engineering is intentionally not implemented in v1.1.
