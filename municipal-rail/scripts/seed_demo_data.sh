#!/usr/bin/env bash
# Onboards "Swartland Local Municipality" and ingests the sample billing
# export, against a locally running API (see run_local.sh). Idempotent
# enough to re-run: ingest just re-upserts the same accounts.
#
# Usage:
#   ./scripts/seed_demo_data.sh                       # against localhost:8000
#   API_BASE=https://xxxx.ngrok-free.app ./scripts/seed_demo_data.sh   # against a tunnel
set -euo pipefail
cd "$(dirname "$0")/.."

API_BASE="${API_BASE:-http://localhost:8000}"

echo "Seeding demo data against $API_BASE ..."

MUNI_RESPONSE=$(curl -s -X POST "$API_BASE/municipalities" \
  -H "Content-Type: application/json" \
  -d '{"name": "Swartland Local Municipality"}')
echo "Municipality: $MUNI_RESPONSE"

MUNI_ID=$(echo "$MUNI_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id', 1))" 2>/dev/null || echo 1)

curl -s -X POST "$API_BASE/municipalities/$MUNI_ID/column-mapping" \
  -H "Content-Type: application/json" \
  -d "{
    \"municipality_id\": $MUNI_ID,
    \"mapping\": {
      \"account_number\": \"AcctNo\",
      \"resident_name\": \"CustomerName\",
      \"resident_contact\": \"Phone\",
      \"address\": \"PropertyAddress\",
      \"account_type\": \"Service\",
      \"balance\": \"OutstandingBalance\"
    }
  }" > /dev/null
echo "Column mapping set for municipality $MUNI_ID."

INGEST_RESPONSE=$(curl -s -X POST "$API_BASE/municipalities/$MUNI_ID/ingest" \
  -F "file=@sample_data/sample_billing_export.csv")
echo "Ingest result: $INGEST_RESPONSE"

echo ""
echo "Done. Try the dashboard:"
echo "  open dashboard/index.html?api=$API_BASE"
echo "Municipality ID: $MUNI_ID, sample account number: SW-00123"
