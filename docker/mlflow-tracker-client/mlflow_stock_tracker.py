import json
import os
import re
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

MODELS_DIRECTORY = Path(
    require_environment_variable("MODELS_DIRECTORY")
)

MODEL_SUBDIRECTORY = require_environment_variable(
    "MODEL_SUBDIRECTORY"
)

MODEL_DIRECTORY = (
    MODELS_DIRECTORY / MODEL_SUBDIRECTORY
)


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

    # Expected files inside /models/model
    model_path = require_file(
        MODEL_DIRECTORY
        / f"lstm_model-{task}-{bar_size}.keras",
        "Keras model",
    )

    scaler_path = require_file(
        MODEL_DIRECTORY
        / f"scaler-{task}-{bar_size}.pkl",
        "Scaler",
    )

    complete_metadata = {
        **request.metadata,
        "task": task,
        "bar_size": bar_size,
        "horizon": request.horizon,
    }

    metadata_path = (
        MODELS_DIRECTORY
        / f"feature_meta-{task}-{bar_size}.json"
    )

    try:
        # Save the received metadata
        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                complete_metadata,
                metadata_file,
                indent=2,
            )

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

            # Put Keras and PKL files under model/
            for model_file in [
                model_path,
                scaler_path,
            ]:
                mlflow.log_artifact(
                    local_path=str(model_file),
                    artifact_path="model",
                )

            # Put JSON under metadata/
            mlflow.log_artifact(
                local_path=str(metadata_path),
                artifact_path="metadata",
            )

            return {
                "status": "success",
                "experimentName": request.experiment_name,
                "runId": run.info.run_id,
                "artifactUri": mlflow.get_artifact_uri(),
                "uploadedFiles": [
                    model_path.name,
                    scaler_path.name,
                    metadata_path.name,
                ],
            }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"MLflow logging failed: {error}",
        ) from error