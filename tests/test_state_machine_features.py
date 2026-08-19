\
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from state_machine_features import (
    COMMON_MODELING_START,
    ENVIRONMENT_COLUMN_RENAME,
    REGIME_COLUMN_RENAME,
    _read_acd_table,
    _read_features_1m,
    _to_market_ml_timestamp,
    build_combined_feature_table,
    build_state_machine_features_1m,
    validate_source_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_ROOT = REPO_ROOT / "data" / "parquet" / "features_1m"
ACD_ROOT = Path.home() / "acd_experiments_local"
SYMBOL = "nvda"
CONFIG_ID = "nvda__or5__atr14__a010__c020__rules-v2"
REGIME_CONFIG_ID = (
    "nvda__or5__atr14__a010__c020__rules-v2"
    "__env-e1__thr-quantile-frozen-6ee45849cb"
    "__regime-moderate_or_anchored-r1__score-r1"
)
POLICY_NAME = "moderate_or_anchored"
POLICY_VERSION = "r1"
TIMEZONE = "America/New_York"

FORBIDDEN_COLUMNS = {
    "or_width_class", "buffer_width_class", "spacing_class",
    "environment_id", "or_width_category",
    "minutes_since_open",
}

# Small window kept for schema/behavior tests to stay fast; the identity
# test below intentionally uses the full history.
_NARROW_START = pd.Timestamp("2024-01-01").date()
_NARROW_END = pd.Timestamp("2024-01-31").date()

_cached_narrow_build = None


def _narrow_build() -> pd.DataFrame:
    global _cached_narrow_build

    if _cached_narrow_build is None:
        acd_run_root = ACD_ROOT / SYMBOL / CONFIG_ID
        state_trace = _read_acd_table(
            acd_run_root,
            "state/state_trace_1min.parquet",
            [
                "timestamp", "symbol", "config_id",
                "open", "high", "low", "close", "volume",
                "opening_range_finalized", "zone_available",
                "state", "prev_state", "state_changed",
                "price_zone", "prev_price_zone",
            ],
            filter_column="config_id",
            filter_value=CONFIG_ID,
        )
        state_trace["datetime"] = _to_market_ml_timestamp(
            state_trace["timestamp"], TIMEZONE
        )
        base_full = _read_features_1m(FEATURES_ROOT, SYMBOL)
        narrow_datetimes = base_full[
            (base_full["trading_date"] >= _NARROW_START)
            & (base_full["trading_date"] <= _NARROW_END)
        ]["datetime"]
        state_trace_narrow = state_trace[
            state_trace["datetime"].isin(narrow_datetimes)
        ]
        assert len(state_trace_narrow) > 0, (
            "no ACD rows found in the narrow test window; "
            "adjust _NARROW_START/_NARROW_END."
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            build_state_machine_features_1m(
                features_root=FEATURES_ROOT,
                acd_root=ACD_ROOT,
                output_root=Path(tmp_dir),
                symbol=SYMBOL,
                config_id=CONFIG_ID,
                regime_config_id=REGIME_CONFIG_ID,
                regime_policy_name=POLICY_NAME,
                regime_policy_version=POLICY_VERSION,
                timezone=TIMEZONE,
                parquet_compression="zstd",
                start_date=_NARROW_START,
                end_date=_NARROW_END,
            )
            import glob

            import pyarrow.parquet as pq

            written = sorted(
                glob.glob(str(Path(tmp_dir) / "**" / "*.parquet"), recursive=True)
            )
            frames = [pq.ParquetFile(f).read().to_pandas() for f in written]
            _cached_narrow_build = pd.concat(frames, ignore_index=True)

    return _cached_narrow_build


# --- 1. Source key integrity ---

def test_no_duplicate_base_keys():
    base = _read_features_1m(
        FEATURES_ROOT, SYMBOL, start_date=_NARROW_START, end_date=_NARROW_END
    )
    assert not base.duplicated(["datetime"]).any()


def test_no_duplicate_acd_keys_after_filter():
    acd_run_root = ACD_ROOT / SYMBOL / CONFIG_ID
    state_trace = _read_acd_table(
        acd_run_root,
        "state/state_trace_1min.parquet",
        ["timestamp", "config_id"],
        filter_column="config_id",
        filter_value=CONFIG_ID,
    )
    assert not state_trace.duplicated(["timestamp"]).any()


def test_one_to_one_join_cardinality():
    acd_run_root = ACD_ROOT / SYMBOL / CONFIG_ID
    state_trace = _read_acd_table(
        acd_run_root,
        "state/state_trace_1min.parquet",
        ["timestamp", "config_id"],
        filter_column="config_id",
        filter_value=CONFIG_ID,
    )
    state_trace["datetime"] = _to_market_ml_timestamp(
        state_trace["timestamp"], TIMEZONE
    )
    base = _read_features_1m(
        FEATURES_ROOT, SYMBOL, start_date=_NARROW_START, end_date=_NARROW_END
    )
    narrow_state_trace = state_trace[
        state_trace["datetime"].isin(base["datetime"])
    ]
    # Raises if either side has a duplicate key for the merge.
    narrow_state_trace.merge(
        base[["datetime"]], on="datetime", how="inner", validate="one_to_one"
    )


# --- 2. Source identity (approved common overlapping range only —
#         trading_date >= common_modeling_start; the ACD tree itself starts
#         ~2019-12-23, before the approved range, so a true full-history
#         comparison would include dates outside the approved contract) ---

def test_ohlcv_identity_matched_rows_common_range():
    acd_run_root = ACD_ROOT / SYMBOL / CONFIG_ID
    state_trace = _read_acd_table(
        acd_run_root,
        "state/state_trace_1min.parquet",
        ["timestamp", "symbol", "config_id", "open", "high", "low",
         "close", "volume"],
        filter_column="config_id",
        filter_value=CONFIG_ID,
    )
    state_trace["datetime"] = _to_market_ml_timestamp(
        state_trace["timestamp"], TIMEZONE
    )
    base = _read_features_1m(
        FEATURES_ROOT, SYMBOL, start_date=COMMON_MODELING_START
    )
    state_trace = state_trace[
        state_trace["datetime"].between(base["datetime"].min(), base["datetime"].max())
    ]

    report = validate_source_identity(state_trace, base)

    assert report.acd_duplicate_keys == 0
    assert report.base_duplicate_keys == 0
    assert report.unmatched_acd_rows == 0
    assert all(count == 0 for count in report.mismatch_counts.values())


# --- 3. Timestamp and DST ---

def test_utc_to_et_winter_date():
    series = pd.Series(
        [pd.Timestamp("2024-01-15 14:30:00", tz="UTC")]
    )
    converted = _to_market_ml_timestamp(series, TIMEZONE)
    assert str(converted.iloc[0].time()) == "09:30:00"


def test_utc_to_et_summer_date():
    series = pd.Series(
        [pd.Timestamp("2024-07-15 13:30:00", tz="UTC")]
    )
    converted = _to_market_ml_timestamp(series, TIMEZONE)
    assert str(converted.iloc[0].time()) == "09:30:00"


def test_utc_to_et_dst_transition():
    # 2024-03-10: US spring-forward DST transition.
    before = pd.Series([pd.Timestamp("2024-03-08 14:30:00", tz="UTC")])
    after = pd.Series([pd.Timestamp("2024-03-11 13:30:00", tz="UTC")])
    assert str(_to_market_ml_timestamp(before, TIMEZONE).iloc[0].time()) == "09:30:00"
    assert str(_to_market_ml_timestamp(after, TIMEZONE).iloc[0].time()) == "09:30:00"


# --- 4. Same-session environment availability ---

def test_environment_unavailable_before_or_finalized_at():
    built = _narrow_build()
    early = built[built["prediction_time"].dt.time < pd.Timestamp("09:36").time()]
    assert (early["sm_env_available"] == False).any()  # noqa: E712


def test_no_prior_session_environment_leakage():
    built = _narrow_build()
    for column in ENVIRONMENT_COLUMN_RENAME.values():
        unavailable = built.loc[~built["sm_env_available"], column]
        assert (unavailable == 0.0).all()


# --- 5. Warm-up invariance ---

def test_opening_range_rows_present_with_sm_available_false():
    built = _narrow_build()
    assert (~built["sm_available"]).any()
    assert built["sm_available"].any()


# --- 6. Row-universe invariance ---

def test_row_count_matches_base_universe():
    built = _narrow_build()
    base = _read_features_1m(
        FEATURES_ROOT, SYMBOL, start_date=_NARROW_START, end_date=_NARROW_END
    )
    assert len(built) == len(base)


def test_identical_timestamp_order():
    built = _narrow_build()
    base = _read_features_1m(
        FEATURES_ROOT, SYMBOL, start_date=_NARROW_START, end_date=_NARROW_END
    )
    assert list(built["datetime"]) == list(
        base.sort_values("datetime")["datetime"]
    )


# --- 7. Leakage exclusions ---

def test_forbidden_columns_absent_from_schema():
    built = _narrow_build()
    assert not (FORBIDDEN_COLUMNS & set(built.columns))


# --- 8. Provenance ---

def test_provenance_columns_present():
    built = _narrow_build()
    required = {
        "signal_config_id", "regime_config_id", "regime_policy_name",
        "regime_policy_version", "environment_schema_version",
        "category_method", "category_threshold_set_id",
        "state_machine_feature_schema_version",
        "market_ml_feature_set_identity", "build_timestamp",
    }
    assert required <= set(built.columns)
    assert (built["signal_config_id"] == CONFIG_ID).all()
    assert (built["regime_config_id"] == REGIME_CONFIG_ID).all()


# --- 9. Sequence rebuild readiness ---

def test_combined_schema_is_superset_of_base_metadata_columns():
    built = _narrow_build()
    required_for_sequences = {
        "datetime", "prediction_time", "feature_available_at",
        "trading_date", "session", "symbol", "source_id",
    }
    assert required_for_sequences <= set(built.columns)


# --- build_combined_feature_table: real end-to-end test, using a window
#     that straddles common_modeling_start (2020-01-03) so the "no
#     pre-common-start rows" behavior is actually exercised. ---

_BOUNDARY_START = pd.Timestamp("2019-12-15").date()
_BOUNDARY_END = pd.Timestamp("2020-01-10").date()

_cached_combined_build = None


def _narrow_combined_build():
    global _cached_combined_build

    if _cached_combined_build is None:
        import glob
        import tempfile

        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as sm_dir, \
                tempfile.TemporaryDirectory() as combined_dir:
            build_state_machine_features_1m(
                features_root=FEATURES_ROOT,
                acd_root=ACD_ROOT,
                output_root=Path(sm_dir),
                symbol=SYMBOL,
                config_id=CONFIG_ID,
                regime_config_id=REGIME_CONFIG_ID,
                regime_policy_name=POLICY_NAME,
                regime_policy_version=POLICY_VERSION,
                timezone=TIMEZONE,
                parquet_compression="zstd",
                start_date=_BOUNDARY_START,
                end_date=_BOUNDARY_END,
            )
            build_combined_feature_table(
                features_root=FEATURES_ROOT,
                state_machine_root=Path(sm_dir),
                output_root=Path(combined_dir),
                symbol=SYMBOL,
                config_id=CONFIG_ID,
                regime_config_id=REGIME_CONFIG_ID,
                parquet_compression="zstd",
                start_date=_BOUNDARY_START,
                end_date=_BOUNDARY_END,
            )
            written = sorted(
                glob.glob(
                    str(Path(combined_dir) / "**" / "*.parquet"), recursive=True
                )
            )
            frames = [pq.ParquetFile(f).read().to_pandas() for f in written]
            _cached_combined_build = pd.concat(frames, ignore_index=True)

    return _cached_combined_build


def test_combined_row_count_matches_control_common_universe():
    combined = _narrow_combined_build()
    control = _read_features_1m(
        FEATURES_ROOT, SYMBOL,
        start_date=COMMON_MODELING_START, end_date=_BOUNDARY_END,
    )
    assert len(combined) == len(control)


def test_combined_timestamps_match_control_common_universe():
    combined = _narrow_combined_build()
    control = _read_features_1m(
        FEATURES_ROOT, SYMBOL,
        start_date=COMMON_MODELING_START, end_date=_BOUNDARY_END,
    )
    assert list(combined["datetime"]) == list(
        control.sort_values("datetime")["datetime"]
    )


def test_combined_has_no_pre_common_start_rows():
    combined = _narrow_combined_build()
    assert (combined["trading_date"] >= COMMON_MODELING_START).all()
    # The requested window starts 2019-12-15, well before
    # common_modeling_start (2020-01-03) — confirm that boundary is
    # actually exercised, not vacuously true because nothing was in range.
    assert combined["trading_date"].min() < pd.Timestamp("2020-01-10").date()


def test_combined_schema_is_core_v1_superset_plus_approved_acd_features():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from model_matrix import DEFAULT_FEATURES

    combined = _narrow_combined_build()

    assert set(DEFAULT_FEATURES) <= set(combined.columns)
    assert {"sm_available", "sm_env_available", "sm_regime_available"} <= set(
        combined.columns
    )
    assert "acd_today_confirmed_count" in combined.columns
    assert "acd_reliability_10d" in combined.columns

    removed_fields = {
        "acd_today_direction_score", "acd_today_resolved_success_count",
        "acd_reliability_20d", "acd_direction_3d", "acd_whipsaw_score",
    }
    assert not (removed_fields & set(combined.columns))
    assert not (FORBIDDEN_COLUMNS & set(combined.columns))
