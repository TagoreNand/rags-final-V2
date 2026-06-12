#!/usr/bin/env bash
# ─── scripts/run.sh ───────────────────────────────────────────────────────────
# Convenience script to start, test, or manage the rag-ops-v2 system.
#
# Usage:
#   ./scripts/run.sh dev        # local dev (no Docker)
#   ./scripts/run.sh worker     # start RQ worker (needs Redis)
#   ./scripts/run.sh docker     # full stack via Docker Compose
#   ./scripts/run.sh test       # run pytest suite
#   ./scripts/run.sh pull-models # pull required Ollama models
#   ./scripts/run.sh demo       # submit a demo task via curl

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Load env ──────────────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a; source .env; set +a
    info "Loaded .env"
elif [ -f ".env.example" ]; then
    warn ".env not found – copying .env.example to .env"
    cp .env.example .env
    set -a; source .env; set +a
fi

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
LLM_MODEL="${LLM_MODEL:-llama3}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

cmd="${1:-help}"

case "$cmd" in

# ── dev ───────────────────────────────────────────────────────────────────────
dev)
    info "Starting API in development mode (inline workers, no Docker required)"
    info "Ensure Ollama is running: ollama serve"
    pip install -q -r requirements.txt
    mkdir -p data
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
    ;;

# ── worker ────────────────────────────────────────────────────────────────────
worker)
    info "Starting RQ worker (requires Redis at $REDIS_URL)"
    pip install -q -r requirements.txt
    python -m backend.workers.rq_worker
    ;;

# ── docker ───────────────────────────────────────────────────────────────────
docker)
    info "Starting full stack via Docker Compose"
    docker compose -f deploy/docker/docker-compose.yml up --build -d
    info "Services:"
    echo "  API:          http://localhost:8000"
    echo "  API docs:     http://localhost:8000/docs"
    echo "  RQ dashboard: http://localhost:9181"
    echo "  Health:       http://localhost:8000/health"
    info "Pull Ollama models inside container:"
    echo "  docker exec rag-ollama ollama pull $LLM_MODEL"
    echo "  docker exec rag-ollama ollama pull $EMBED_MODEL"
    ;;

# ── docker-down ───────────────────────────────────────────────────────────────
docker-down)
    docker compose -f deploy/docker/docker-compose.yml down
    ;;

# ── test ─────────────────────────────────────────────────────────────────────
test)
    info "Running test suite"
    pip install -q -r requirements.txt
    python -m pytest backend/tests/ -v --tb=short "${@:2}"
    ;;

# ── pull-models ──────────────────────────────────────────────────────────────
pull-models)
    info "Pulling Ollama models: $LLM_MODEL + $EMBED_MODEL"
    ollama pull "$LLM_MODEL"
    ollama pull "$EMBED_MODEL"
    info "Models ready."
    ;;

# ── demo ─────────────────────────────────────────────────────────────────────
demo)
    GOAL="${2:-Explain how transformer neural networks work and their impact on NLP}"
    info "Submitting demo task: $GOAL"
    RESP=$(curl -sf -X POST http://localhost:8000/v1/tasks \
        -H "Content-Type: application/json" \
        -d "{\"goal\": \"$GOAL\", \"max_steps\": 6}" 2>&1) || die "API not reachable. Start with: ./scripts/run.sh dev"
    TASK_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")
    info "Task created: $TASK_ID"
    echo ""
    info "Poll status:"
    echo "  curl http://localhost:8000/v1/tasks/$TASK_ID | python3 -m json.tool"
    echo ""
    info "Watch until done:"
    echo "  watch -n 2 'curl -s http://localhost:8000/v1/tasks/$TASK_ID | python3 -m json.tool | grep status'"
    ;;

# ── help ─────────────────────────────────────────────────────────────────────
help|*)
    echo ""
    echo "  Agentic RAG Ops v2 - Management Script"
    echo ""
    echo "  Commands:"
    echo "    dev          Start API in dev mode (auto-reload, no Docker)"
    echo "    worker       Start RQ background worker"
    echo "    docker       Build + start full stack with Docker Compose"
    echo "    docker-down  Stop Docker Compose stack"
    echo "    test         Run pytest suite"
    echo "    pull-models  Pull Ollama LLM + embedding models"
    echo "    demo [goal]  Submit a demo task to the running API"
    echo ""
    ;;
esac
