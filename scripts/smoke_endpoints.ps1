<#
  Smoke-test every DocIntel API endpoint end to end (PowerShell).

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\smoke_endpoints.ps1
    powershell -ExecutionPolicy Bypass -File scripts\smoke_endpoints.ps1 http://localhost:8000/api/v1 keyA

  Requires the API to be running (python -m uvicorn app.main:app --port 8000).
  Pass the API key as the 2nd argument when the server runs with ENABLE_AUTH=true.
#>
param(
  [string]$BaseUrl = "http://localhost:8000/api/v1",
  [string]$ApiKey  = ""
)

$headers = @{}
if ($ApiKey) { $headers["X-API-Key"] = $ApiKey }

function Section($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Show($o) { if ($null -ne $o) { $o | ConvertTo-Json -Depth 6 } }

# Call the API; on an HTTP error, print a clean status line instead of a red stack trace.
function Api($Method, $Url, $Body) {
  try {
    if ($Body) {
      return Invoke-RestMethod -Headers $headers -Method $Method -ContentType "application/json" -Body $Body -Uri $Url
    }
    return Invoke-RestMethod -Headers $headers -Method $Method -Uri $Url
  } catch {
    $code = ""
    if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
    Write-Host ("  [HTTP $code] " + $_.ErrorDetails.Message) -ForegroundColor Yellow
    return $null
  }
}

if (-not (Api GET "$BaseUrl/health")) {
  Write-Host "Cannot reach the API. Start it: python -m uvicorn app.main:app --port 8000" -ForegroundColor Red
  exit 1
}
if ($ApiKey) { Write-Host "Using API key for tenant auth." -ForegroundColor DarkGray }

Section "GET /health";                   Show (Api GET "$BaseUrl/health")
Section "GET /ready";                    Show (Api GET "$BaseUrl/ready")
Section "GET /system/info";              Show (Api GET "$BaseUrl/system/info")
Section "POST /documents/ingest-sample"; Show (Api POST "$BaseUrl/documents/ingest-sample")

Section "GET /documents"
$docs = Api GET "$BaseUrl/documents"
Show $docs
$docId = if ($docs) { $docs[0].id } else { "" }

Section "GET /documents/{id}"
if ($docId) { Show (Api GET "$BaseUrl/documents/$docId") }

Section "POST /documents/upload"
$tmp = Join-Path $env:TEMP "demo_invoice.txt"
"Invoice Number: INV-9001`nVendor: Demo Corp`nSubtotal: 100.00`nTax: 10.00`nTotal Amount: 110.00" |
  Set-Content -Encoding ascii $tmp
$curlArgs = @("-s")
if ($ApiKey) { $curlArgs += @("-H", "X-API-Key: $ApiKey") }
$curlArgs += @("-F", "file=@$tmp", "$BaseUrl/documents/upload")
& curl.exe @curlArgs
Write-Host ""

Section "POST /search/query"
$body = @{ question = "What is the invoice total amount?"; top_k = 3 } | ConvertTo-Json
Show (Api POST "$BaseUrl/search/query" $body)

Section "GET /review/tasks"
$tasks = Api GET "$BaseUrl/review/tasks"
Show $tasks
$taskId = if ($tasks) { $tasks[0].id } else { "" }

Section "POST /review/tasks/{id}/resolve"
if ($taskId) {
  $rbody = @{ outcome = "approved"; notes = "validated" } | ConvertTo-Json
  Show (Api POST "$BaseUrl/review/tasks/$taskId/resolve" $rbody)
}

Section "GET /analytics/overview"; Show (Api GET "$BaseUrl/analytics/overview")
Section "GET /analytics/drift";    Show (Api GET "$BaseUrl/analytics/drift")
Section "GET /events/recent";      Show (Api GET "$BaseUrl/events/recent")

Section "GET /metrics (first 20 lines)"
try {
  ((Invoke-WebRequest -UseBasicParsing -Headers $headers "$BaseUrl/metrics").Content -split "`n" | Select-Object -First 20) -join "`n"
} catch {
  Write-Host "  [metrics unavailable]" -ForegroundColor Yellow
}

Write-Host "`nAll endpoints exercised." -ForegroundColor Green
