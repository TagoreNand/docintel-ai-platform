"""Optional MLflow experiment tracking with a transparent no-op fallback.

Training scripts call ``with start_run(name) as run: run.log_params(...)`` without
caring whether MLflow is configured. When ``ENABLE_MLFLOW`` is false or the
tracking server is unreachable, a null run absorbs the calls so training never
fails because of missing infrastructure.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class _NullRun:
    def log_params(self, params: dict) -> None:  # noqa: D401
        logger.debug("mlflow disabled; params=%s", params)

    def log_metrics(self, metrics: dict) -> None:
        logger.debug("mlflow disabled; metrics=%s", metrics)

    def log_artifact(self, path: str) -> None:
        logger.debug("mlflow disabled; artifact=%s", path)

    def log_model(self, model, artifact_path: str, registered_model_name: str | None = None) -> None:
        logger.debug("mlflow disabled; model=%s", registered_model_name or artifact_path)


class _MlflowRun:
    def __init__(self, mlflow) -> None:
        self._mlflow = mlflow

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def log_model(self, model, artifact_path: str, registered_model_name: str | None = None) -> None:
        import mlflow.sklearn

        mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)


@contextmanager
def start_run(run_name: str) -> Iterator[object]:
    if not settings.enable_mlflow:
        yield _NullRun()
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=run_name):
            logger.info("MLflow run started", extra={"run_name": run_name})
            yield _MlflowRun(mlflow)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow unavailable (%s); continuing without tracking", exc.__class__.__name__)
        yield _NullRun()
