\
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from common import load_config, session_for_timestamp


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "pipeline.yaml"
)


def test_session_boundaries():
    config = load_config(CONFIG)
    tz = "America/New_York"

    cases = {
        "2019-05-06 04:00": "premarket",
        "2019-05-06 09:29": "premarket",
        "2019-05-06 09:30": "regular",
        "2019-05-06 15:59": "regular",
        "2019-05-06 16:00": "aftermarket",
        "2019-05-06 19:59": "aftermarket",
        "2019-05-06 20:00": "overnight",
        "2019-05-07 03:59": "overnight",
    }

    for value, expected in cases.items():
        timestamp = pd.Timestamp(
            value
        ).tz_localize(tz)

        assert (
            session_for_timestamp(
                timestamp,
                config,
            )
            == expected
        )
