# IBKR Historical Tick Scraper

This tick scraper is separate from `ibq_scraper.py`. The existing historical-bar code and its checkpoint method are not changed.

## 1. What the scraper requests

It calls IBKR's `reqHistoricalTicks`, which IBKR describes as Historical Time & Sales. This is historical tick data, not the live `reqTickByTickData` stream.

| YAML `tick_type` | IBKR callback | Main CSV values |
| --- | --- | --- |
| `TRADES` | `historicalTicksLast` | price, size, exchange, conditions, trade flags |
| `BID_ASK` | `historicalTicksBidAsk` | bid/ask prices, bid/ask sizes, quote flags |
| `MIDPOINT` | `historicalTicks` | midpoint and size |

IBKR historical tick timestamps have one-second resolution. Multiple legitimate ticks can therefore have the same timestamp. The scraper preserves them and adds a chronological `sequence` column; it does not incorrectly remove equal-looking trades.

## 2. Session meanings

| YAML session | Contract exchange | `useRth` | Meaning |
| --- | --- | ---: | --- |
| `rth` | `SMART` | `1` | Regular trading hours only |
| `extended` | `SMART` | `0` | All available SMART data, including RTH and pre/post-market; it is not an "outside-RTH-only" filter |
| `overnight` | `OVERNIGHT` | `0` | Direct IBKR overnight venue for supported US stocks/ETFs |

The overnight market is a separate venue. Availability must be tested per symbol and account. A market-data subscription is also required for Historical Time & Sales.

## 3. Place the files

Place these files in your project:

```text
your-project/
├── ibq_tick_scraper.py
└── config/
    └── scraper-config-semicond-ticks.yml
```

The script intentionally uses the same project integration as the bar scraper:

- `config.BASE_DIR` or the parent of `config.DATA_DIR`
- `utils.appenv.APPENV` for the Gateway host, port, and client ID
- the official `ibapi` package

## 4. Start safely

First test only one symbol for a short interval. In the YAML, temporarily comment out the other symbols and use:

```yaml
defaults:
  duration: "1 H"
  sessions: [rth]
  tick_type: TRADES
```

Then run:

```bash
python ibq_tick_scraper.py --config config/scraper-config-semicond-ticks.yml
```

Expected output location:

```text
data/ticks/<group>/<symbol>_<tick_type>_ticks_<session>.csv
```

Example:

```text
data/ticks/semiconductor/nvda_trades_ticks_rth.csv
```

## 5. Checkpoint and restart behavior

IBKR permits at most 1,000 requested tick data points per call, although it can return extra ticks to complete the last second. The scraper requests pages backward in time and writes each successful page atomically before advancing its JSON checkpoint.

If the Gateway or connection fails:

1. Restart IB Gateway if necessary.
2. Run the same command with the unchanged YAML.
3. The job resumes from its last safely stored cursor and page number.

Do not change a job's symbol, duration, session, type, routing, or request size while its checkpoint exists. The scraper rejects a mismatched checkpoint instead of mixing incompatible data.

## 6. Important IBKR constraints

- Maximum request size: 1,000 ticks.
- A request returns only one trading session; multiple requests are necessary.
- IBKR may return more than 1,000 ticks to finish a complete second.
- Historical Time & Sales availability is limited to the latest three years.
- A Level 1/Top of Book subscription is required.
- Historical requests are paced. `BID_ASK` counts as two request units.
- IBKR is not designed as a bulk specialist tick-data vendor. A full busy day across many symbols can require many pages and substantial time/storage.

The supplied pacing configuration uses a shared rolling limiter across all worker threads. Keep it conservative, especially if another program is also requesting historical data through the same account.

## 7. CSV schemas

`TRADES`:

```text
sequence,timestamp_utc,epoch_seconds,price,size,exchange,special_conditions,past_limit,unreported
```

`BID_ASK`:

```text
sequence,timestamp_utc,epoch_seconds,bid_price,ask_price,bid_size,ask_size,bid_past_low,ask_past_high
```

`MIDPOINT`:

```text
sequence,timestamp_utc,epoch_seconds,midpoint,size
```

All request cursors and output timestamps are UTC to avoid Gateway-login timezone and daylight-saving ambiguity.

## 8. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Error 200 / no security definition | Ambiguous or unsupported contract | Add the correct `primary_exchange`, or verify that the symbol supports the selected venue |
| Error 162/165 with no data | No ticks in that interval/venue or unavailable historical data | Test a recent RTH interval first; for `overnight`, verify symbol eligibility |
| Market-data subscription error | Required Level 1 subscription is missing | Verify the subscription in IB Gateway/TWS for the requested contract |
| Long pacing wait | The rolling historical-request budget is full | Let the scraper wait; do not start another historical downloader |
| Checkpoint mismatch | YAML job settings changed after partial download | Restore the original settings, or deliberately remove only that job's checkpoint and hidden parts directory to restart it |
| Completed checkpoint but missing CSV | Output was moved/deleted after completion | Restore the CSV or remove only that job's checkpoint and rerun |

For the first validation, use recent `rth` `TRADES` data for one highly liquid symbol. After that works, test `extended`, and test `overnight` separately because it uses a different venue.

Do not run the bar scraper and tick scraper simultaneously if both use the same IB API client ID. If you deliberately run more than one historical-data process, use different client IDs and remember that IBKR pacing is account-wide while this scraper's limiter can coordinate only its own process.

## 9. Official IBKR references

- [Historical Time & Sales introduction](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-time-sales/introduction)
- [`reqHistoricalTicks` request parameters](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-time-sales/requesting-time-and-sales-data)
- [Historical tick callback methods](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-time-sales/receiving-time-and-sales-data)
- [Historical request pacing rules](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/pacing-violations-for-small-bars-30-secs-or-less)
- [IBKR overnight venue routing](https://www.interactivebrokers.com/docs/tws-api/doc/orders/place-order/trading-the-overnight-session)
