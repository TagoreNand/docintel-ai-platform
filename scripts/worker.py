"""Entrypoint to run the distributed ingestion worker."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api_gateway"))

from app.worker import run_worker  # noqa: E402

if __name__ == "__main__":
    run_worker()
