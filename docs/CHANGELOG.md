# Changelog

## 1.1.0

Session-aware historical-data design.

### Added

- `premarket` session: 04:00-09:29
- `regular` session: 09:30-15:59
- `aftermarket` session: 16:00-19:59
- future `overnight` session: 20:00-03:59
- `config/sources.csv`
- source-specific coverage contracts
- `source_id` canonical metadata
- `trading_date` canonical metadata
- exchange-calendar-aware future overnight trading-date mapping
- session-level coverage reports
- source-aware scraper manifest
- separate raw incoming path by source

### Changed

- current historical feed claims only premarket, regular and aftermarket
- overnight is not treated as missing for the current feed
- regularization does not cross named market-session boundaries by default
- synthetic OHLC is flat at the previous observed close
- session-leading/trailing/full-session gaps stay unresolved by default
- Parquet remains monthly by symbol/year/month to avoid tiny files
