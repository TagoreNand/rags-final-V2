#!/usr/bin/env bash
# ─── scripts/demo_all.sh ──────────────────────────────────────────────────────
# End-to-end walkthrough: starts the API, exercises EVERY endpoint, the auth
# flow, the management CLI, and the offline eval harness — start to finish.
#
#   ./scripts/demo_all.sh              # start its own server, run everything, stop it
#   ./scripts/demo_all.sh --no-server  # use an already-running server at $BASE_URL
#
# Env overrides: BASE_URL, PORT, API_KEY, PYTHON, SKIP_OFFLINE=1 (skip tests+eval)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

BASE_URL="${BASE_URL:-http://localhost:8000}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-dev-key-1}"          # auth is enabled for this demo
PY="${PYTHON:-python3}"
START_SERVER=1; [ "${1:-}" = "--no-server" ] && START_SERVER=0

cyan(){ printf '\n\033[1;36m======== %s ========\033[0m\n' "$*"; }
step(){ printf '\n\033[1;33m$ %s\033[0m\n' "$*"; }
SRV_PID=""
cleanup(){ [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null && echo "[stopped API pid $SRV_PID]"; }
trap cleanup EXIT

# ── 0. Setup ──────────────────────────────────────────────────────────────────
cyan "0 · SETUP (deps, env, models)"
cat <<'NOTE'
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pip install -r requirements-ml.txt        # optional: hnswlib ANN + cross-encoder
  cp .env.example .env
  ollama pull llama3                         # needed for full task COMPLETION
  ollama pull nomic-embed-text
NOTE

# ── 1. Offline quality checks (no server needed) ───────────────────────────────
if [ "${SKIP_OFFLINE:-0}" != "1" ]; then
cyan "1 · TESTS + EVALUATION (offline, deterministic)"
step "pytest backend/tests/ -q"
"$PY" -m pytest backend/tests/ -q -p no:cacheprovider 2>/dev/null | tail -1
step "python -m backend.eval.harness --gate"
"$PY" -m backend.eval.harness --gate 2>/dev/null | tail -5
step "python -m backend.eval.harness --compare --dim 64   (retrieval stage ablation)"
"$PY" -m backend.eval.harness --compare --dim 64 2>/dev/null | tail -6
step "python -m backend.eval.harness --ablation            (reflection A/B)"
"$PY" -m backend.eval.harness --ablation 2>/dev/null | tail -7
else
  cyan "1 · TESTS + EVALUATION (skipped: SKIP_OFFLINE=1)"
fi

# ── 2. Start API ────────────────────────────────────────────────────────────────
if [ "$START_SERVER" -eq 1 ]; then
  cyan "2 · START API  (auth enabled: API_KEYS=$API_KEY)"
  step "API_KEYS=$API_KEY uvicorn backend.app:app --host 0.0.0.0 --port $PORT"
  API_KEYS="$API_KEY" "$PY" -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT" >/tmp/demo_uvicorn.log 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 20); do curl -s "$BASE_URL/health" >/dev/null 2>&1 && break; sleep 1; done
  echo "[API up at $BASE_URL pid $SRV_PID]"
fi

# ── 3. System / unauthenticated endpoints ──────────────────────────────────────
cyan "3 · SYSTEM ENDPOINTS"
step "GET /            (research UI)";        curl -s -o /dev/null -w 'HTTP %{http_code}  (%{size_download} bytes HTML)\n' "$BASE_URL/"
step "GET /health";                            curl -s "$BASE_URL/health"; echo
step "GET /metrics     (Prometheus)";          curl -s "$BASE_URL/metrics" | grep -E '^rag_|^# HELP rag_tasks' | head -8
step "GET /openapi.json";                      curl -s "$BASE_URL/openapi.json" | "$PY" -c "import sys,json;d=json.load(sys.stdin);print('title:',d['info']['title'],d['info']['version']);print('paths:',', '.join(sorted(d['paths'])))"
step "GET /docs        (Swagger UI)";          curl -s -o /dev/null -w 'HTTP %{http_code}\n' "$BASE_URL/docs"

# ── 4. Auth flow (API key  →  JWT bearer) ───────────────────────────────────────
cyan "4 · AUTH  (exchange API key for a JWT, then use both schemes)"
step "POST /v1/auth/token   {\"api_key\":\"$API_KEY\"}"
TOKEN_JSON=$(curl -s -X POST "$BASE_URL/v1/auth/token" -H 'Content-Type: application/json' -d "{\"api_key\":\"$API_KEY\"}")
echo "$TOKEN_JSON" | "$PY" -m json.tool
TOKEN=$(echo "$TOKEN_JSON" | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)
step "GET /v1/tasks  with  Authorization: Bearer <jwt>"
curl -s -o /dev/null -w 'HTTP %{http_code}\n' "$BASE_URL/v1/tasks" -H "Authorization: Bearer $TOKEN"
step "GET /v1/tasks  with  X-API-Key (no key = 401)"
curl -s -o /dev/null -w 'with key:    HTTP %{http_code}\n' "$BASE_URL/v1/tasks" -H "X-API-Key: $API_KEY"
curl -s -o /dev/null -w 'without key: HTTP %{http_code}\n' "$BASE_URL/v1/tasks"

# ── 5. Task lifecycle (all /v1/tasks endpoints) ─────────────────────────────────
cyan "5 · TASK LIFECYCLE"
step "POST /v1/tasks   (create + enqueue → 202)"
CREATE=$(curl -s -X POST "$BASE_URL/v1/tasks" -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" \
  -d '{"goal":"How does RLHF improve large language models?","max_steps":8,"sources":["wikipedia","arxiv"]}')
echo "$CREATE" | "$PY" -m json.tool
TID=$(echo "$CREATE" | "$PY" -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
echo "[task_id=$TID]"
step "GET /v1/tasks          (list, scoped to caller)"
curl -s "$BASE_URL/v1/tasks?limit=5" -H "X-API-Key: $API_KEY" | "$PY" -c "import sys,json;d=json.load(sys.stdin);print('total:',d['total']);[print(' ',t['status'],t['task_id'][:8],t['goal'][:50]) for t in d['tasks']]"
step "GET /v1/tasks/$TID    (poll status — 3x)"
for i in 1 2 3; do
  curl -s "$BASE_URL/v1/tasks/$TID" -H "X-API-Key: $API_KEY" | "$PY" -c "import sys,json;d=json.load(sys.stdin);print(f'  poll: status={d[\"status\"]:>9}  steps={len(d[\"steps\"])}  citations={len(d[\"citations\"])}')"
  sleep 2
done
step "GET /v1/tasks/$TID/citations"
curl -s "$BASE_URL/v1/tasks/$TID/citations" -H "X-API-Key: $API_KEY" | "$PY" -c "import sys,json;d=json.load(sys.stdin);print('citations:',len(d.get('citations',[])))"
step "DELETE /v1/tasks/$TID   (cancel / soft-delete → 204)"
curl -s -o /dev/null -w 'HTTP %{http_code}\n' -X DELETE "$BASE_URL/v1/tasks/$TID" -H "X-API-Key: $API_KEY"

# ── 6. Management CLI ───────────────────────────────────────────────────────────
cyan "6 · MANAGEMENT CLI (scripts/cli.py)"
step "cli.py health";  "$PY" scripts/cli.py --api-key "$API_KEY" health  || true
step "cli.py submit";  "$PY" scripts/cli.py --api-key "$API_KEY" submit "Compare PPO and DPO for preference fine-tuning" || true
step "cli.py tasks";   "$PY" scripts/cli.py --api-key "$API_KEY" tasks || true
step "cli.py stats (direct DB)"; "$PY" scripts/cli.py stats || true

# ── 7. Metrics reflect the activity ─────────────────────────────────────────────
cyan "7 · /metrics AFTER ACTIVITY"
curl -s "$BASE_URL/metrics" | grep -E '^rag_(tasks_total|active_tasks|retrieval|llm_latency_count|groundedness_count)' | head -8

# ── 8. Other run modes (not executed here) ──────────────────────────────────────
cyan "8 · OTHER RUN MODES (reference)"
cat <<'NOTE'
  # Background worker (durable queue; needs Redis — otherwise tasks run in a thread)
  python -m backend.workers.rq_worker

  # Full stack via Docker (API :8000 · RQ dashboard :9181)
  docker compose -f deploy/docker/docker-compose.yml up --build -d
  docker exec rag-ollama ollama pull llama3
  docker exec rag-ollama ollama pull nomic-embed-text

  # Kubernetes
  kubectl apply -f deploy/k8s/00-namespace-config.yaml
  kubectl apply -f deploy/k8s/01-redis.yaml
  kubectl apply -f deploy/k8s/02-ollama.yaml
  kubectl apply -f deploy/k8s/03-api.yaml
  kubectl apply -f deploy/k8s/04-worker.yaml
  kubectl apply -f deploy/k8s/05-rbac-network.yaml
  kubectl get pods -n rag-ops -w
NOTE
cyan "DONE — every endpoint exercised"
echo "Note: tasks reach 'succeeded' only with Ollama running (ollama serve + models)."
