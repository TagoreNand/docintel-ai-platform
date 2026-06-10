#!/usr/bin/env bash
#
# Smoke-test every DocIntel API endpoint end to end.
#
#   Usage:  bash scripts/smoke_endpoints.sh [BASE_URL] [API_KEY]
#   e.g.    bash scripts/smoke_endpoints.sh
#           bash scripts/smoke_endpoints.sh http://localhost:8000/api/v1 keyA
#
# Requires the API to be running (e.g. uvicorn on :8000). Pass an API_KEY only
# when the server runs with ENABLE_AUTH=true.

set -o pipefail

BASE="${1:-http://localhost:8000/api/v1}"
KEY="${2:-}"
AUTH=()
[ -n "$KEY" ] && AUTH=(-H "X-API-Key: $KEY")

# Find a Python interpreter (python3 on macOS/Linux, python on Windows).
PY="$(command -v python3 || command -v python || true)"

extract() {  # extract <json-key-path-expr> ; reads stdin, prints value or ""
  [ -n "$PY" ] && "$PY" -c "$1" 2>/dev/null || true
}

say() { printf "\n=== %s ===\n" "$1"; }

# Fail fast with a friendly message if the server is not up.
if ! curl -sf "$BASE/health" >/dev/null 2>&1; then
  echo "Cannot reach $BASE/health — is the API running? Start it with:"
  echo "    (in your other terminal)  python -m uvicorn app.main:app --port 8000"
  exit 1
fi

say "GET /health";                   curl -s "$BASE/health"; echo
say "GET /ready";                    curl -s "$BASE/ready"; echo
say "GET /system/info";              curl -s "${AUTH[@]}" "$BASE/system/info"; echo
say "POST /documents/ingest-sample"; curl -s "${AUTH[@]}" -X POST "$BASE/documents/ingest-sample"; echo
say "GET /documents";                curl -s "${AUTH[@]}" "$BASE/documents"; echo

DOC_ID="$(curl -s "${AUTH[@]}" "$BASE/documents" | extract 'import sys,json;d=json.load(sys.stdin);print(d[0]["id"] if d else "")')"
say "GET /documents/{id}";           [ -n "$DOC_ID" ] && curl -s "${AUTH[@]}" "$BASE/documents/$DOC_ID"; echo

say "POST /documents/upload"
printf 'Invoice Number: INV-9001\nVendor: Demo Corp\nSubtotal: 100.00\nTax: 10.00\nTotal Amount: 110.00\n' > /tmp/demo_invoice.txt
curl -s "${AUTH[@]}" -F "file=@/tmp/demo_invoice.txt" "$BASE/documents/upload"; echo

say "POST /search/query"
curl -s "${AUTH[@]}" -H "Content-Type: application/json" -X POST "$BASE/search/query" \
  -d '{"question":"What is the invoice total amount?","top_k":3}'; echo

say "GET /review/tasks";             curl -s "${AUTH[@]}" "$BASE/review/tasks"; echo
TASK_ID="$(curl -s "${AUTH[@]}" "$BASE/review/tasks" | extract 'import sys,json;d=json.load(sys.stdin);print(d[0]["id"] if d else "")')"
say "POST /review/tasks/{id}/resolve"
[ -n "$TASK_ID" ] && curl -s "${AUTH[@]}" -H "Content-Type: application/json" \
  -X POST "$BASE/review/tasks/$TASK_ID/resolve" -d '{"outcome":"approved","notes":"validated"}'; echo

say "GET /analytics/overview";       curl -s "${AUTH[@]}" "$BASE/analytics/overview"; echo
say "GET /analytics/drift";          curl -s "${AUTH[@]}" "$BASE/analytics/drift"; echo
say "GET /events/recent";            curl -s "${AUTH[@]}" "$BASE/events/recent"; echo
say "GET /metrics (first 20 lines)"; curl -s "${AUTH[@]}" "$BASE/metrics" | head -20; echo

printf "\nAll endpoints exercised.\n"
