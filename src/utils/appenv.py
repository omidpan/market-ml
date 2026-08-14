from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


# appenv.py is expected at:
#   <project_root>/src/utils/appenv.py
#
# Therefore parents[2] resolves to <project_root>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# By default, use exactly one project-level .env file:
#   <project_root>/.env
#
# ENV_FILE can override this when needed:
#   ENV_FILE=/private/path/.env python ...
ENV_FILE = Path(
    os.environ.get(
        "ENV_FILE",
        str(PROJECT_ROOT / ".env"),
    )
).expanduser().resolve()


def load_project_env() -> Path:
    """
    Load the single project .env file.

    Existing shell environment variables take precedence over values
    from .env because override=False.

    This function never changes the process working directory.
    """
    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f"Could not find project .env file: {ENV_FILE}"
        )

    loaded = load_dotenv(
        dotenv_path=ENV_FILE,
        override=False,
    )

    if not loaded:
        logger.debug(
            "dotenv did not add new variables from %s; "
            "they may already be present in the environment.",
            ENV_FILE,
        )

    return ENV_FILE


class APPENV:
    """
    IB application environment.

    Required variables:
        HOST
        PORT

    Optional:
        CLIENT_ID

    If CLIENT_ID is omitted, a random client id is generated to preserve
    the previous behavior.
    """

    def __init__(self) -> None:
        env_path = load_project_env()

        host = os.getenv("HOST")
        port = os.getenv("PORT")
        client_id = os.getenv("CLIENT_ID")

        if host is None or not host.strip():
            raise RuntimeError(
                f"HOST is not defined in {env_path}"
            )

        if port is None or not port.strip():
            raise RuntimeError(
                f"PORT is not defined in {env_path}"
            )

        try:
            parsed_port = int(port)
        except ValueError as exc:
            raise RuntimeError(
                f"PORT must be an integer in {env_path}; got {port!r}"
            ) from exc

        if not (1 <= parsed_port <= 65535):
            raise RuntimeError(
                f"PORT must be between 1 and 65535; got {parsed_port}"
            )

        if client_id is None or not client_id.strip():
            parsed_client_id = random.randint(1, 10000)
        else:
            try:
                parsed_client_id = int(client_id)
            except ValueError as exc:
                raise RuntimeError(
                    f"CLIENT_ID must be an integer; got {client_id!r}"
                ) from exc

            if parsed_client_id < 0:
                raise RuntimeError(
                    f"CLIENT_ID must be non-negative; got {parsed_client_id}"
                )

        self.host = host.strip()
        self.port = parsed_port
        self.client_id = parsed_client_id
        self.env_file = env_path

        logger.info(
            "Loaded IB environment from %s",
            env_path,
        )