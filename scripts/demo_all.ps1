#
# scripts/demo_all.ps1 - end-to-end walkthrough (PowerShell port of demo_all.sh).
# Starts the API, exercises EVERY endpoint + auth flow + management CLI + the
# offline eval harness, start to finish.
#
# Usage:
#   .\scripts\demo_all.ps1            # use .venv if present; run everything
#   .\scripts\demo_all.ps1 -Install   # create .venv + pip install -r requirements.txt, then run
#   .\scripts\demo_all.ps1 -NoServer  # use an already-running API at -BaseUrl
#   .\scripts\demo_all.ps1 -SkipOffline   # skip the tests + eval phase
#
# If script execution is blocked:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
param(
  [switch]$Install,
  [switch]$NoServer,
  [switch]$SkipOffline,
  [string]$BaseUrl = "http://localhost:8000",
  [int]$Port       = 8000,
  [string]$ApiKey  = "dev-key-1",
  [string]$Python  = "python"
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Headers = @{ "X-API-Key" = $ApiKey }

function Section($t) { Write-Host ""; Write-Host "======== $t ========" -ForegroundColor Cyan }
function Step($t)    { Write-Host ""; Write-Host "> $t" -ForegroundColor Yellow }
function HttpCode {
  param($Method, $Url, $Hdrs = @{})
  try {
    $r = Invoke-WebRequest -Method $Method -Uri $Url -Headers $Hdrs -TimeoutSec 10 -UseBasicParsing
    return $r.StatusCode
  } catch {
    if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
    return "ERR"
  }
}

# Resolve Python: prefer an existing project venv unless -Python was overridden
$venvPy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if ($Python -eq "python" -and (Test-Path $venvPy)) { $Python = $venvPy }

# Optional one-time setup: create venv + install dependencies
if ($Install) {
  Section "SETUP - create .venv + install dependencies"
  if (-not (Test-Path $venvPy)) { python -m venv .venv }
  $Python = $venvPy
  & $Python -m pip install --upgrade pip
  & $Python -m pip install -r requirements.txt
}

# Preflight: are the core dependencies importable by the chosen Python?
& $Python -c "import fastapi, structlog, uvicorn, pydantic, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "Dependencies are NOT installed for: $Python" -ForegroundColor Red
  Write-Host "Set up once, then re-run this script:" -ForegroundColor Yellow
  Write-Host "    python -m venv .venv"
  Write-Host "    .\.venv\Scripts\Activate.ps1"
  Write-Host "    pip install -r requirements.txt"
  Write-Host "  ...or let this script do it automatically:" -ForegroundColor Yellow
  Write-Host "    .\scripts\demo_all.ps1 -Install"
  exit 1
}
Write-Host "[using Python: $Python]" -ForegroundColor DarkGray

# 0. Setup (reference only)
Section "0 - SETUP (deps, env, models)"
@(
  "  python -m venv .venv ;  .\.venv\Scripts\Activate.ps1",
  "  pip install -r requirements.txt",
  "  pip install -r requirements-ml.txt        # optional: hnswlib ANN + cross-encoder",
  "  Copy-Item .env.example .env",
  "  ollama pull llama3 ;  ollama pull nomic-embed-text   # needed for task COMPLETION"
) | ForEach-Object { Write-Host $_ }

# 1. Offline quality checks
if (-not $SkipOffline) {
  Section "1 - TESTS + EVALUATION (offline, deterministic)"
  Step "pytest backend/tests/ -q";              & $Python -m pytest backend/tests/ -q
  Step "harness --gate";                        & $Python -m backend.eval.harness --gate
  Step "harness --compare --dim 64";            & $Python -m backend.eval.harness --compare --dim 64
  Step "harness --ablation";                    & $Python -m backend.eval.harness --ablation
} else {
  Section "1 - TESTS + EVALUATION (skipped: -SkipOffline)"
}

# 2. Start API
$srv = $null
if (-not $NoServer) {
  Section "2 - START API (auth enabled: API_KEYS=$ApiKey)"
  $env:API_KEYS = $ApiKey
  $startArgs = @{
    FilePath     = $Python
    ArgumentList = @("-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "$Port")
    PassThru     = $true
    WindowStyle  = "Hidden"
  }
  $srv = Start-Process @startArgs
  $up = $false
  for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-RestMethod "$BaseUrl/openapi.json" -TimeoutSec 5 | Out-Null; $up = $true; break } catch { Start-Sleep 1 }
  }
  if ($up) { Write-Host "[API up at $BaseUrl  pid $($srv.Id)]" -ForegroundColor Green }
  else     { Write-Host "[API failed to start - see logs above]" -ForegroundColor Red }
}

try {
  # 3. System endpoints
  Section "3 - SYSTEM ENDPOINTS"
  Step "GET /            (research UI)"; Write-Host ("HTTP " + (HttpCode "GET" "$BaseUrl/"))
  Step "GET /health";                    Invoke-RestMethod "$BaseUrl/health" | ConvertTo-Json -Compress
  Step "GET /metrics     (Prometheus)"
  (Invoke-RestMethod "$BaseUrl/metrics") -split "\r?\n" | Select-String "^rag_" | Select-Object -First 6
  Step "GET /openapi.json paths"
  (Invoke-RestMethod "$BaseUrl/openapi.json").paths.PSObject.Properties.Name | Sort-Object
  Step "GET /docs        (Swagger)";     Write-Host ("HTTP " + (HttpCode "GET" "$BaseUrl/docs"))

  # 4. Auth (API key -> JWT)
  Section "4 - AUTH (exchange API key for a JWT, then use both schemes)"
  Step "POST /v1/auth/token"
  $tokBody = @{ api_key = $ApiKey } | ConvertTo-Json
  $tok = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/auth/token" -ContentType "application/json" -Body $tokBody
  $tok | ConvertTo-Json -Compress
  $jwt = $tok.access_token
  Step "GET /v1/tasks  Bearer / X-API-Key / no-auth"
  Write-Host ("  Bearer    -> HTTP " + (HttpCode "GET" "$BaseUrl/v1/tasks" @{ Authorization = "Bearer $jwt" }))
  Write-Host ("  X-API-Key -> HTTP " + (HttpCode "GET" "$BaseUrl/v1/tasks" $Headers))
  Write-Host ("  no auth   -> HTTP " + (HttpCode "GET" "$BaseUrl/v1/tasks") + "  (expect 401)")

  # 5. Task lifecycle
  Section "5 - TASK LIFECYCLE"
  Step "POST /v1/tasks  (create + enqueue -> 202)"
  $taskBody = @{ goal = "How does RLHF improve large language models?"; max_steps = 8; sources = @("wikipedia", "arxiv") } | ConvertTo-Json
  $task = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/tasks" -Headers $Headers -ContentType "application/json" -Body $taskBody
  $tid  = $task.task_id
  $task | ConvertTo-Json -Depth 5
  Step "GET /v1/tasks   (list, scoped to caller)"
  (Invoke-RestMethod "$BaseUrl/v1/tasks?limit=5" -Headers $Headers).tasks | Select-Object status, task_id, goal | Format-Table -AutoSize
  Step "GET /v1/tasks/{id}  (poll x3)"
  for ($i = 0; $i -lt 3; $i++) {
    $t = Invoke-RestMethod "$BaseUrl/v1/tasks/$tid" -Headers $Headers
    Write-Host ("  status={0,-9} steps={1} citations={2}" -f $t.status, $t.steps.Count, $t.citations.Count)
    Start-Sleep 2
  }
  Step "GET /v1/tasks/{id}/citations"
  Write-Host ("  citations: " + (Invoke-RestMethod "$BaseUrl/v1/tasks/$tid/citations" -Headers $Headers).citations.Count)
  Step "DELETE /v1/tasks/{id}  (-> 204)"
  Write-Host ("HTTP " + (HttpCode "DELETE" "$BaseUrl/v1/tasks/$tid" $Headers))

  # 6. Management CLI
  Section "6 - MANAGEMENT CLI (scripts/cli.py)"
  Step "cli.py health"; & $Python scripts/cli.py --api-key $ApiKey health
  Step "cli.py submit"; & $Python scripts/cli.py --api-key $ApiKey submit "Compare PPO and DPO for preference fine-tuning"
  Step "cli.py tasks";  & $Python scripts/cli.py --api-key $ApiKey tasks
  Step "cli.py stats";  & $Python scripts/cli.py stats

  # 7. /metrics after activity
  Section "7 - /metrics AFTER ACTIVITY"
  (Invoke-RestMethod "$BaseUrl/metrics") -split "\r?\n" | Select-String "^rag_(tasks_total|active_tasks|retrieval|llm_latency_count|groundedness_count)" | Select-Object -First 8
}
finally {
  if ($srv) { Stop-Process -Id $srv.Id -ErrorAction SilentlyContinue; Write-Host "[stopped API pid $($srv.Id)]" }
}

# 8. Other run modes (reference)
Section "8 - OTHER RUN MODES (reference)"
@(
  "  python -m backend.workers.rq_worker",
  "  docker compose -f deploy/docker/docker-compose.yml up --build -d",
  "  kubectl apply -f deploy/k8s/   ;  kubectl get pods -n rag-ops -w"
) | ForEach-Object { Write-Host $_ }
Section "DONE - every endpoint exercised"
Write-Host "Note: tasks reach succeeded only with Ollama running (ollama serve + models)."
