from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json

from common import sha256_file


def record_fetch(
    manifest_path: Path,
    *,
    source_id: str,
    symbol: str,
    trading_date: str,
    source_file: Path,
    status: str,
    claimed_sessions: list[str],
    requested_start: str | None = None,
    requested_end: str | None = None,
) -> None:
    """
    Append one immutable scraper-provenance record.

    source_id is mandatory so multiple providers can coexist later.
    """
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "recorded_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "source_id": source_id,
        "symbol": symbol.upper(),
        "trading_date": trading_date,
        "claimed_sessions": (
            claimed_sessions
        ),
        "status": status,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "source_file": str(
            source_file.resolve()
        ),
        "source_sha256": (
            sha256_file(source_file)
            if source_file.exists()
            else None
        ),
    }

    with manifest_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(row) + "\n"
        )
