#!/usr/bin/env bash
# Sets up the venv, installs deps, and starts the API locally.
# Run this from the municipal-rail/ directory:
#   ./scripts/run_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d venv ]; then
  echo "Creating venv..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "Starting the API on http://localhost:8000 (Ctrl+C to stop)."
echo "API docs: http://localhost:8000/docs"
echo ""
echo "In another terminal, run:"
echo "  ./scripts/seed_demo_data.sh        # one-time: onboard a municipality + sample data"
echo "  npx ngrok http 8000                # if you need PayGate's webhook to reach you"
echo ""

uvicorn app.main:app --reload
