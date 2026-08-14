from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from common import (
    CANONICAL_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    OHLCV,
    load_config,
)


REQUIRED_STORAGE_COLUMNS = list(
    CANONICAL_COLUMNS
)

KEY_COLUMNS = ["datetime", "source_id"]
WAP_OHLC_TOLERANCE = 0.005
PARTITION_META_FILENAME = "_partition_meta.json"
PARTITION_META_VERSION = 1


def normalize_symbol(
    symbol: str,
) -> str:
    return (
        str(symbol)
        .strip()
        .lower()
    )


def _parse_bool_series(
    values: pd.Series,
    column: str,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        values.dtype
    ):
        return values.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )
    parsed = normalized.map(
        mapping
    )

    if parsed.isna().any():
        examples = (
            values.loc[
                parsed.isna()
            ]
            .astype(str)
            .head(5)
            .tolist()
        )

        raise ValueError(
            "Invalid boolean values in "
            f"{column}: {examples}"
        )

    return parsed.astype(bool)


def _normalize_datetime(
    values: pd.Series,
    timezone: str,
) -> pd.Series:
    """
    Normalize canonical timestamps without mixed-DST-offset warnings.

    Canonical CSV timestamps may contain both -04:00 and -05:00 over a year.
    Offset-aware strings are parsed through UTC, then converted back to the
    configured market timezone. Naive strings are interpreted as local wall
    time in the configured timezone.
    """
    text = values.astype("string").str.strip()

    explicit_offset = text.str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})$",
        case=False,
        regex=True,
        na=False,
    )

    # A single canonical datetime column should not mix naive and aware values.
    if explicit_offset.any() and not explicit_offset.all():
        raise ValueError(
            "Datetime column mixes timezone-aware and naive values."
        )

    if explicit_offset.all():
        parsed = pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        return parsed.dt.tz_convert(timezone)

    parsed = pd.to_datetime(
        values,
        errors="coerce",
    )

    if not pd.api.types.is_datetime64_dtype(parsed.dtype):
        # Last-resort row-wise parse for unusual but valid naive inputs.
        result = []
        for value in values:
            try:
                timestamp = pd.Timestamp(value)
                if timestamp.tzinfo is not None:
                    raise ValueError(
                        "Unexpected timezone-aware value in naive datetime column."
                    )
                timestamp = timestamp.tz_localize(
                    timezone,
                    ambiguous="raise",
                    nonexistent="raise",
                )
                result.append(timestamp)
            except Exception:
                result.append(pd.NaT)

        return pd.Series(
            result,
            index=values.index,
        )

    return parsed.dt.tz_localize(
        timezone,
        ambiguous="raise",
        nonexistent="raise",
    )


def normalize_canonical_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    timezone: str,
) -> pd.DataFrame:
    """
    Normalize and validate canonical observed/regularized rows.

    Zero volume is never used as an imputation marker.

    Observed and imputed rows require complete OHLCV/WAP/bar_count.
    Unresolved regularization-grid rows may keep those fields null.
    """
    data = df.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    missing = [
        column
        for column
        in REQUIRED_STORAGE_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Input is not canonical yet. "
            f"Missing columns: {missing}"
        )

    data["datetime"] = (
        _normalize_datetime(
            data["datetime"],
            timezone,
        )
    )

    if data[
        "datetime"
    ].isna().any():
        raise ValueError(
            "Canonical input contains "
            "invalid datetime values."
        )

    data["calendar_date"] = (
        pd.to_datetime(
            data["calendar_date"],
            errors="coerce",
        ).dt.date
    )

    data["trading_date"] = (
        pd.to_datetime(
            data["trading_date"],
            errors="coerce",
        ).dt.date
    )

    if (
        data[
            "calendar_date"
        ].isna().any()
        or data[
            "trading_date"
        ].isna().any()
    ):
        raise ValueError(
            "Invalid calendar_date or "
            "trading_date exists."
        )

    actual_calendar_date = (
        data["datetime"]
        .dt.tz_convert(timezone)
        .dt.date
    )

    if not data[
        "calendar_date"
    ].eq(
        actual_calendar_date
    ).all():
        raise ValueError(
            "calendar_date disagrees with "
            "the local datetime date."
        )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["source_id"] = (
        data["source_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["session"] = (
        data["session"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected_symbol = (
        normalize_symbol(
            symbol
        )
    )

    foreign_symbols = sorted(
        set(
            data.loc[
                data["symbol"].ne(
                    expected_symbol
                ),
                "symbol",
            ].unique()
        )
    )

    if foreign_symbols:
        raise ValueError(
            "Input contains symbols "
            f"outside {expected_symbol!r}: "
            f"{foreign_symbols}"
        )

    if (
        data[
            "source_id"
        ].eq("").any()
        or data[
            "session"
        ].eq("").any()
    ):
        raise ValueError(
            "Blank source_id or session exists."
        )

    data["is_observed"] = (
        _parse_bool_series(
            data["is_observed"],
            "is_observed",
        )
    )

    data["is_imputed"] = (
        _parse_bool_series(
            data["is_imputed"],
            "is_imputed",
        )
    )

    impossible_flags = (
        data["is_observed"]
        & data["is_imputed"]
    )

    if impossible_flags.any():
        raise ValueError(
            "A row cannot be both "
            "observed and imputed."
        )

    numeric_columns = [
        *OHLCV,
        *MICROSTRUCTURE_COLUMNS,
    ]

    for column in numeric_columns:
        data[column] = (
            pd.to_numeric(
                data[column],
                errors="coerce",
            )
        )

    complete_required = (
        data["is_observed"]
        | data["is_imputed"]
    )

    incomplete_required = (
        data.loc[
            complete_required,
            numeric_columns,
        ].isna()
    )

    if (
        incomplete_required
        .any()
        .any()
    ):
        bad = (
            incomplete_required
            .sum()
        )
        bad = bad[
            bad > 0
        ].to_dict()

        raise ValueError(
            "Observed/imputed rows have "
            "missing market values: "
            f"{bad}"
        )

    volume_present = (
        data["volume"].notna()
    )

    if (
        data.loc[
            volume_present,
            "volume",
        ] < 0
    ).any():
        raise ValueError(
            "Negative volume exists."
        )

    count_present = (
        data[
            "bar_count"
        ].notna()
    )

    if (
        data.loc[
            count_present,
            "bar_count",
        ] < 0
    ).any():
        raise ValueError(
            "Negative bar_count exists."
        )

    non_integer_count = (
        count_present
        & ~data[
            "bar_count"
        ].eq(
            data[
                "bar_count"
            ].round()
        )
    )

    if non_integer_count.any():
        raise ValueError(
            "Non-integer bar_count exists."
        )

    data["bar_count"] = (
        data["bar_count"]
        .round()
        .astype("Int64")
    )

    price_complete = (
        data[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    high_bad = (
        price_complete
        & (
            data["high"]
            < data[
                [
                    "open",
                    "low",
                    "close",
                ]
            ].max(axis=1)
        )
    )

    low_bad = (
        price_complete
        & (
            data["low"]
            > data[
                [
                    "open",
                    "high",
                    "close",
                ]
            ].min(axis=1)
        )
    )

    if (
        high_bad.any()
        or low_bad.any()
    ):
        raise ValueError(
            "OHLC integrity failure."
        )

    wap_checkable = (
        data["wap"].notna()
        & data["low"].notna()
        & data["high"].notna()
    )

    wap_distance = pd.concat(
        [
            (data["low"] - data["wap"]).clip(lower=0),
            (data["wap"] - data["high"]).clip(lower=0),
        ],
        axis=1,
    ).max(axis=1)

    wap_bad = (
        wap_checkable
        & (
            wap_distance
            > (WAP_OHLC_TOLERANCE + 1e-12)
        )
    )

    if wap_bad.any():
        max_distance = float(
            wap_distance.loc[wap_bad].max()
        )
        raise ValueError(
            "WAP exceeds displayed OHLC range by more than "
            f"${WAP_OHLC_TOLERANCE:.3f}; "
            f"max_excess=${max_distance:.6f}."
        )

    if (
        "imputation_source_datetime"
        in data.columns
    ):
        source_values = data[
            "imputation_source_datetime"
        ]

        non_null = (
            source_values.notna()
            & source_values.astype(str)
            .str.strip()
            .ne("")
        )

        normalized_source = (
            pd.Series(
                pd.NaT,
                index=data.index,
                dtype=(
                    "datetime64[ns, "
                    f"{timezone}]"
                ),
            )
        )

        if non_null.any():
            parsed_source = (
                _normalize_datetime(
                    source_values.loc[
                        non_null
                    ],
                    timezone,
                )
            )

            if parsed_source.isna().any():
                raise ValueError(
                    "Invalid "
                    "imputation_source_datetime "
                    "exists."
                )

            normalized_source.loc[
                non_null
            ] = parsed_source

        data[
            "imputation_source_datetime"
        ] = normalized_source

    ordered_columns = [
        column
        for column in CANONICAL_COLUMNS
        if column in data.columns
    ]
    ordered_columns += sorted(
        column
        for column in data.columns
        if column not in ordered_columns
    )

    return (
        data[
            ordered_columns
        ]
        .sort_values(
            KEY_COLUMNS
        )
        .reset_index(drop=True)
    )


def dedupe_or_raise(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorized duplicate handling.

    Exact duplicate full rows collapse immediately.
    If one (datetime, source_id) key still has multiple different payloads,
    the write stops.

    This avoids the old Python loop over every duplicate timestamp.
    """
    if df.empty:
        return df.reset_index(drop=True)

    deduped = (
        df.drop_duplicates(
            keep="first"
        )
        .reset_index(drop=True)
    )

    conflicts = deduped.duplicated(
        KEY_COLUMNS,
        keep=False,
    )

    if conflicts.any():
        examples = (
            deduped.loc[
                conflicts,
                KEY_COLUMNS,
            ]
            .drop_duplicates()
            .head(5)
            .to_dict(
                orient="records"
            )
        )
        raise ValueError(
            "Conflicting canonical rows exist for "
            "(datetime, source_id). "
            f"Examples: {examples}"
        )

    return (
        deduped
        .sort_values(
            KEY_COLUMNS
        )
        .reset_index(drop=True)
    )


def _ordered_columns(
    df: pd.DataFrame,
) -> list[str]:
    canonical = [
        column
        for column in CANONICAL_COLUMNS
        if column in df.columns
    ]
    extras = sorted(
        column
        for column in df.columns
        if column not in canonical
    )
    return [
        *canonical,
        *extras,
    ]


def _frame_fingerprint(
    df: pd.DataFrame,
) -> str:
    """
    Logical SHA-256 fingerprint of one normalized monthly frame.

    Used only as an acceleration index. The Parquet data remains the source
    of truth; if metadata/file stat checks fail, the store falls back to a
    full read and exact comparison.
    """
    ordered = (
        df[
            _ordered_columns(df)
        ]
        .sort_values(
            KEY_COLUMNS
        )
        .reset_index(drop=True)
    )

    digest = hashlib.sha256()

    digest.update(
        "\x1f".join(
            ordered.columns
        ).encode("utf-8")
    )
    digest.update(b"\x00")

    row_hashes = (
        pd.util.hash_pandas_object(
            ordered,
            index=False,
            categorize=True,
        )
        .to_numpy(
            dtype="uint64",
            copy=False,
        )
    )

    digest.update(
        row_hashes.tobytes()
    )

    return digest.hexdigest()


def _read_partition_meta(
    meta_path: Path,
) -> dict | None:
    if not meta_path.exists():
        return None

    try:
        payload = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    return payload


def _metadata_fast_match(
    *,
    meta: dict | None,
    final_path: Path,
    incoming: pd.DataFrame,
    incoming_hash: str,
) -> bool:
    """
    Fast no-op detection.

    We trust a sidecar only when:
      - its logical hash/row count/schema match the incoming normalized month;
      - the Parquet file's size and mtime still match the sidecar.

    If any check fails, we read the Parquet and verify exactly.
    """
    if meta is None:
        return False

    if meta.get(
        "version"
    ) != PARTITION_META_VERSION:
        return False

    if meta.get(
        "logical_sha256"
    ) != incoming_hash:
        return False

    if int(
        meta.get(
            "row_count",
            -1,
        )
    ) != len(incoming):
        return False

    if meta.get(
        "columns"
    ) != _ordered_columns(
        incoming
    ):
        return False

    try:
        stat = final_path.stat()
    except FileNotFoundError:
        return False

    if int(
        meta.get(
            "parquet_size_bytes",
            -1,
        )
    ) != int(stat.st_size):
        return False

    if int(
        meta.get(
            "parquet_mtime_ns",
            -1,
        )
    ) != int(stat.st_mtime_ns):
        return False

    return True


def _atomic_write_json(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}."
        f"{os.getpid()}.tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            temp_path,
            path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_partition_meta(
    *,
    final_path: Path,
    meta_path: Path,
    payload: pd.DataFrame,
    logical_hash: str | None = None,
) -> None:
    if logical_hash is None:
        logical_hash = (
            _frame_fingerprint(
                payload
            )
        )

    stat = final_path.stat()

    meta = {
        "version": (
            PARTITION_META_VERSION
        ),
        "logical_sha256": (
            logical_hash
        ),
        "row_count": int(
            len(payload)
        ),
        "columns": (
            _ordered_columns(
                payload
            )
        ),
        "parquet_size_bytes": int(
            stat.st_size
        ),
        "parquet_mtime_ns": int(
            stat.st_mtime_ns
        ),
    }

    _atomic_write_json(
        meta,
        meta_path,
    )


def _frames_exactly_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> bool:
    if set(
        left.columns
    ) != set(
        right.columns
    ):
        return False

    columns = _ordered_columns(
        left
    )

    left_cmp = (
        left[
            columns
        ]
        .sort_values(
            KEY_COLUMNS
        )
        .reset_index(drop=True)
    )
    right_cmp = (
        right[
            columns
        ]
        .sort_values(
            KEY_COLUMNS
        )
        .reset_index(drop=True)
    )

    return left_cmp.equals(
        right_cmp
    )


def _merge_upsert(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """
    Vectorized safe upsert.

    Existing rows are never silently changed or deleted.
    Overlapping keys must have identical payloads.
    Only genuinely new keys are appended.
    """
    existing = dedupe_or_raise(
        existing
    )
    incoming = dedupe_or_raise(
        incoming
    )

    if set(
        existing.columns
    ) != set(
        incoming.columns
    ):
        missing_from_existing = sorted(
            set(incoming.columns)
            - set(existing.columns)
        )
        missing_from_incoming = sorted(
            set(existing.columns)
            - set(incoming.columns)
        )

        raise ValueError(
            "Parquet schema mismatch. "
            "A full rebuild is required for this partition. "
            f"missing_from_existing={missing_from_existing}, "
            f"missing_from_incoming={missing_from_incoming}"
        )

    columns = _ordered_columns(
        incoming
    )
    existing = existing[
        columns
    ]
    incoming = incoming[
        columns
    ]

    existing_keys = pd.MultiIndex.from_frame(
        existing[
            KEY_COLUMNS
        ]
    )
    incoming_keys = pd.MultiIndex.from_frame(
        incoming[
            KEY_COLUMNS
        ]
    )

    overlap_mask = incoming_keys.isin(
        existing_keys
    )

    if overlap_mask.any():
        existing_indexed = (
            existing.set_index(
                KEY_COLUMNS,
                drop=False,
            )
        )
        incoming_overlap = (
            incoming.loc[
                overlap_mask
            ]
            .set_index(
                KEY_COLUMNS,
                drop=False,
            )
        )

        overlap_index = (
            incoming_overlap.index
        )

        existing_overlap = (
            existing_indexed.loc[
                overlap_index,
                columns,
            ]
        )

        incoming_overlap = (
            incoming_overlap[
                columns
            ]
        )

        payload_columns = [
            column
            for column in columns
            if column not in KEY_COLUMNS
        ]

        left = existing_overlap[
            payload_columns
        ]
        right = incoming_overlap[
            payload_columns
        ]

        equal_cells = (
            left.eq(right)
            | (
                left.isna()
                & right.isna()
            )
        ).fillna(False)

        conflict_rows = (
            ~equal_cells.all(
                axis=1
            )
        )

        if conflict_rows.any():
            examples = [
                {
                    "datetime": str(
                        key[0]
                    ),
                    "source_id": str(
                        key[1]
                    ),
                }
                for key in (
                    conflict_rows.loc[
                        conflict_rows
                    ]
                    .index
                    .tolist()[:5]
                )
            ]

            raise ValueError(
                "Existing Parquet conflicts with incoming "
                "canonical rows. No data was overwritten. "
                f"Examples: {examples}"
            )

    new_mask = (
        ~incoming_keys.isin(
            existing_keys
        )
    )

    new_count = int(
        new_mask.sum()
    )

    if new_count == 0:
        return (
            existing,
            0,
        )

    payload = pd.concat(
        [
            existing,
            incoming.loc[
                new_mask
            ],
        ],
        ignore_index=True,
    )

    payload = (
        dedupe_or_raise(
            payload
        )
    )

    return (
        payload,
        new_count,
    )


def atomic_write_parquet(
    payload: pd.DataFrame,
    final_path: Path,
    *,
    compression: str,
) -> None:
    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        final_path.with_name(
            f".{final_path.name}."
            f"{os.getpid()}.tmp"
        )
    )

    try:
        payload.to_parquet(
            temp_path,
            index=False,
            engine="pyarrow",
            compression=compression,
        )

        os.replace(
            temp_path,
            final_path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_monthly_parquet(
    df: pd.DataFrame,
    root: Path,
    *,
    symbol: str,
    compression: str,
    timezone: str,
) -> list[Path]:
    """
    Optimized monthly canonical Parquet upsert.

    Fast path on repeated identical runs:
      normalized incoming month
        -> logical fingerprint
        -> sidecar/stat match
        -> SKIP without reading/re-writing the Parquet file

    Safe fallback:
      read existing month
        -> exact vectorized compare
        -> conflict-safe upsert of only new keys
        -> atomic rewrite only when data actually changed
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required "
            "for Parquet storage."
        ) from exc

    incoming = (
        normalize_canonical_frame(
            df,
            symbol=symbol,
            timezone=timezone,
        )
    )

    incoming = dedupe_or_raise(
        incoming
    )

    incoming["_year"] = (
        incoming["datetime"]
        .dt.tz_convert(timezone)
        .dt.year
    )
    incoming["_month"] = (
        incoming["datetime"]
        .dt.tz_convert(timezone)
        .dt.month
    )

    normalized_symbol = (
        normalize_symbol(
            symbol
        )
    )

    written: list[
        Path
    ] = []

    skipped = 0

    for (
        year,
        month,
    ), group in incoming.groupby(
        [
            "_year",
            "_month",
        ],
        sort=True,
    ):
        month_label = (
            f"{int(year):04d}-"
            f"{int(month):02d}"
        )

        output_dir = (
            root
            / f"symbol={normalized_symbol}"
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
        )

        final_path = (
            output_dir
            / "part.parquet"
        )

        meta_path = (
            output_dir
            / PARTITION_META_FILENAME
        )

        month_incoming = (
            group
            .drop(
                columns=[
                    "_year",
                    "_month",
                ]
            )
            .copy()
        )

        month_incoming = (
            dedupe_or_raise(
                month_incoming
            )
        )

        incoming_hash = (
            _frame_fingerprint(
                month_incoming
            )
        )

        if final_path.exists():
            meta = (
                _read_partition_meta(
                    meta_path
                )
            )

            if _metadata_fast_match(
                meta=meta,
                final_path=final_path,
                incoming=month_incoming,
                incoming_hash=incoming_hash,
            ):
                print(
                    f"[{month_label}] "
                    "unchanged - fast skipped "
                    f"({len(month_incoming):,} rows)"
                )
                skipped += 1
                continue

            existing = (
                pd.read_parquet(
                    final_path,
                    engine="pyarrow",
                )
            )

            existing = (
                normalize_canonical_frame(
                    existing,
                    symbol=symbol,
                    timezone=timezone,
                )
            )

            existing = (
                dedupe_or_raise(
                    existing
                )
            )

            if _frames_exactly_equal(
                existing,
                month_incoming,
            ):
                # Bootstrap/repair the sidecar so the NEXT identical run can
                # skip without decoding the Parquet file.
                _write_partition_meta(
                    final_path=final_path,
                    meta_path=meta_path,
                    payload=existing,
                    logical_hash=incoming_hash,
                )

                print(
                    f"[{month_label}] "
                    "unchanged - verified and skipped "
                    f"({len(existing):,} rows)"
                )
                skipped += 1
                continue

            payload, new_count = (
                _merge_upsert(
                    existing,
                    month_incoming,
                )
            )

            if new_count == 0:
                # Incoming is an identical subset of the existing partition.
                # Upsert semantics keep existing historical rows.
                payload_hash = (
                    _frame_fingerprint(
                        payload
                    )
                )
                _write_partition_meta(
                    final_path=final_path,
                    meta_path=meta_path,
                    payload=payload,
                    logical_hash=payload_hash,
                )

                print(
                    f"[{month_label}] "
                    "no new rows - skipped "
                    f"({len(payload):,} stored rows)"
                )
                skipped += 1
                continue

            atomic_write_parquet(
                payload,
                final_path,
                compression=compression,
            )

            payload_hash = (
                _frame_fingerprint(
                    payload
                )
            )
            _write_partition_meta(
                final_path=final_path,
                meta_path=meta_path,
                payload=payload,
                logical_hash=payload_hash,
            )

            print(
                f"[{month_label}] "
                f"updated +{new_count:,} rows "
                f"-> {len(payload):,} total"
            )
            written.append(
                final_path
            )
            continue

        atomic_write_parquet(
            month_incoming,
            final_path,
            compression=compression,
        )

        _write_partition_meta(
            final_path=final_path,
            meta_path=meta_path,
            payload=month_incoming,
            logical_hash=incoming_hash,
        )

        print(
            f"[{month_label}] "
            f"created {len(month_incoming):,} rows"
        )

        written.append(
            final_path
        )

    print(
        "Parquet store summary: "
        f"written={len(written)}, "
        f"skipped={skipped}"
    )

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized conflict-safe upsert of canonical "
            "observed/regularized market rows into monthly Parquet."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical CSV input.",
    )

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/pipeline.yaml"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(
        args.config
    )

    frame = pd.read_csv(
        args.input
    )

    write_monthly_parquet(
        frame,
        args.root,
        symbol=args.symbol,
        compression=(
            config.parquet_compression
        ),
        timezone=config.timezone,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
