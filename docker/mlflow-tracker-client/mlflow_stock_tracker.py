import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================================
# FASTAPI APPLICATION
# =========================================================================

app = FastAPI(
    title="MLflow Stock Tracker API",
    description="Receives training results and logs them to MLflow.",
    version="1.0.0",
)


# =========================================================================
# ENVIRONMENT CONFIGURATION
# =========================================================================

def require_environment_variable(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


MLFLOW_TRACKING_URI = require_environment_variable(
    "MLFLOW_TRACKING_URI"
)

MODEL_RUNS_DIRECTORY = Path(
    require_environment_variable("MODEL_RUNS_DIRECTORY")
).resolve()


# =========================================================================
# REQUEST BODY
# =========================================================================

class LogRunRequest(BaseModel):
    task: str

    bar_size: str = Field(
        alias="barSize"
    )

    experiment_name: str = Field(
        alias="experimentName"
    )

    horizon: int

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    metrics: dict[str, float] = Field(
        default_factory=dict
    )

    model_run_path: str = Field(
        alias="modelRunPath"
    )

    training_run_id: str = Field(
        alias="trainingRunId"
    )

    model_config = {
        "populate_by_name": True
    }


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def validate_file_component(
    value: str,
    field_name: str,
) -> str:
    """
    Validate values used to construct filenames.
    """

    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}: {value}. "
                "Use only letters, numbers, underscores "
                "and hyphens."
            ),
        )

    return value


def require_file(
    file_path: Path,
    description: str,
) -> Path:
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"{description} was not found: "
                f"{file_path}"
            ),
        )

    return file_path


def resolve_model_run_directory(
    model_run_path: str,
    training_run_id: str,
) -> Path:
    """
    Resolve modelRunPath under the trusted MODEL_RUNS_DIRECTORY root.

    No recursive search: the caller supplies the exact relative run
    directory, which is validated and then used directly.
    """

    if "\\" in model_run_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "modelRunPath must use POSIX '/' separators "
                "only."
            ),
        )

    if model_run_path.startswith("/"):
        raise HTTPException(
            status_code=400,
            detail="modelRunPath must be a relative path.",
        )

    if any(
        segment == ".."
        for segment in model_run_path.split("/")
    ):
        raise HTTPException(
            status_code=400,
            detail="modelRunPath must not contain '..' segments.",
        )

    resolved_path = (
        MODEL_RUNS_DIRECTORY / model_run_path
    ).resolve()

    if not resolved_path.is_relative_to(MODEL_RUNS_DIRECTORY):
        raise HTTPException(
            status_code=400,
            detail=(
                "modelRunPath must resolve under "
                "MODEL_RUNS_DIRECTORY."
            ),
        )

    expected_directory_name = f"run_id={training_run_id}"

    if resolved_path.name != expected_directory_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "modelRunPath directory "
                f"({resolved_path.name}) does not match "
                f"trainingRunId ({expected_directory_name})."
            ),
        )

    return resolved_path


# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/")
def application_information():
    return {
        "application": "MLflow Stock Tracker API",
        "status": "running",
    }


@app.post("/log-run")
def log_run(request: LogRunRequest):
    task = validate_file_component(
        request.task,
        "task",
    )

    bar_size = validate_file_component(
        request.bar_size,
        "barSize",
    )

    if request.horizon < 1:
        raise HTTPException(
            status_code=400,
            detail="horizon must be at least 1.",
        )

    run_directory = resolve_model_run_directory(
        request.model_run_path,
        request.training_run_id,
    )

    # Located directly under the caller-supplied, validated run
    # directory — no recursive search.
    model_path = require_file(
        run_directory / "best_model.keras",
        "Keras model",
    )

    complete_metadata = {
        **request.metadata,
        "task": task,
        "bar_size": bar_size,
        "horizon": request.horizon,
        "training_run_id": request.training_run_id,
    }

    metadata_temp_path: Path | None = None

    try:
        # MODEL_RUNS_DIRECTORY is mounted read-only, so metadata is
        # staged in a temp file and logged as an MLflow artifact
        # rather than written back into the mounted run directory.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as metadata_file:
            json.dump(
                complete_metadata,
                metadata_file,
                indent=2,
            )
            metadata_temp_path = Path(metadata_file.name)

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            request.experiment_name
        )

        with mlflow.start_run() as run:
            parameters = {
                "task": task,
                "bar_size": bar_size,
                "horizon": request.horizon,
                "training_run_id": request.training_run_id,
                "feature_count": len(
                    complete_metadata.get(
                        "feature_columns",
                        [],
                    )
                ),
                **request.parameters,
            }

            mlflow.log_params(parameters)

            if request.metrics:
                mlflow.log_metrics(
                    request.metrics
                )

            # Put the Keras model under model/
            mlflow.log_artifact(
                local_path=str(model_path),
                artifact_path="model",
            )

            # Put JSON under metadata/
            mlflow.log_artifact(
                local_path=str(metadata_temp_path),
                artifact_path="metadata",
            )

            return {
                "status": "success",
                "experimentName": request.experiment_name,
                "runId": run.info.run_id,
                "artifactUri": mlflow.get_artifact_uri(),
                "uploadedFiles": [
                    model_path.name,
                ],
            }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"MLflow logging failed: {error}",
        ) from error

    finally:
        if metadata_temp_path is not None:
            metadata_temp_path.unlink(missing_ok=True)