import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api_gateway"
sys.path.insert(0, str(API_ROOT))

from app.main import app  # noqa: E402

output_path = PROJECT_ROOT / "docs" / "openapi.json"
output_path.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print(f"Wrote OpenAPI spec to {output_path}")
