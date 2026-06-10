"""Retrain the document classifier, folding in reviewer-corrected feedback labels."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api_gateway"))

from app.ml.train_classifier import train  # noqa: E402
from app.services.feedback import load_feedback  # noqa: E402


def main() -> None:
    texts, labels = load_feedback()
    print(f"Loaded {len(texts)} reviewer-feedback examples.")
    card = train(extra=(texts, labels)) if texts else train()
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
