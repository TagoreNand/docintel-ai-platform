"""Train and persist all DocIntel ML models (classifier + anomaly detector)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api_gateway"))

from app.ml.train_anomaly import train as train_anomaly  # noqa: E402
from app.ml.train_classifier import train as train_classifier  # noqa: E402


def main() -> None:
    print("==> Training document classifier ...")
    print(json.dumps(train_classifier(), indent=2))
    print("\n==> Training anomaly detector ...")
    print(json.dumps(train_anomaly(), indent=2))
    print("\nModels written to data/models/.")


if __name__ == "__main__":
    main()
