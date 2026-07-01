#!/usr/bin/env bash
# Clawback — local run. Works with zero API keys (deterministic letters +
# demo unlock). Add STRIPE_SECRET_KEY for live payments and ANTHROPIC_API_KEY
# for LLM-polished letters.
set -e
cd "$(dirname "$0")"
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -q -r requirements.txt
echo "Clawback running at http://localhost:8000"
uvicorn app.main:app --reload --port "${PORT:-8000}"
