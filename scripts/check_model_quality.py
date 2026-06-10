"""CI model-quality gate.

Trains the document classifier and fails (exit code 1) if test accuracy or macro-F1
regress below the configured thresholds (CLASSIFIER_ACCURACY_GATE /
CLASSIFIER_MACRO_F1_GATE). Wire into CI to block merges that degrade the model.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api_gateway"))

from app.core.config import settings  # noqa: E402
from app.ml.train_classifier import train  # noqa: E402


def main() -> None:
    card = train()
    accuracy = card["test_accuracy"]
    macro_f1 = card["test_macro_f1"]
    acc_gate = settings.classifier_accuracy_gate
    f1_gate = settings.classifier_macro_f1_gate

    print(f"test_accuracy={accuracy} (gate >= {acc_gate})")
    print(f"test_macro_f1={macro_f1} (gate >= {f1_gate})")

    if accuracy < acc_gate or macro_f1 < f1_gate:
        print("MODEL QUALITY GATE: FAILED")
        sys.exit(1)
    print("MODEL QUALITY GATE: PASSED")


if __name__ == "__main__":
    main()
