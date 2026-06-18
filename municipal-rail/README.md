# Municipal Rail — Sprint 1 & 2

The integration adapter, account/billing engine, and reconciliation +
audit trail described in the architecture plan. This is the backend
infrastructure layer — not a citizen-facing app. A WhatsApp flow or web
dashboard (Sprint 3) calls this API; it doesn't replace it.

## Setup

```bash
cd municipal-rail
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## Walkthrough: onboarding a municipality and ingesting its billing export

1. **Create the municipality**

```bash
curl -X POST http://localhost:8000/municipalities \
  -H "Content-Type: application/json" \
  -d '{"name": "Swartland Local Municipality"}'
```

2. **Set its column mapping** — every municipality exports billing data
   with different column names. This is the piece that means onboarding
   a new municipality is configuration, not a new integration build.
   The sample export uses these columns:

```bash
curl -X POST http://localhost:8000/municipalities/1/column-mapping \
  -H "Content-Type: application/json" \
  -d '{
    "municipality_id": 1,
    "mapping": {
      "account_number": "AcctNo",
      "resident_name": "CustomerName",
      "resident_contact": "Phone",
      "address": "PropertyAddress",
      "account_type": "Service",
      "balance": "OutstandingBalance"
    }
  }'
```

3. **Ingest the sample billing export**

```bash
curl -X POST http://localhost:8000/municipalities/1/ingest \
  -F "file=@sample_data/sample_billing_export.csv"
```

4. **List the normalized accounts**

```bash
curl http://localhost:8000/municipalities/1/accounts
```

5. **Record a payment** (Sprint 4 will trigger this from a real PSP
   webhook instead of a manual call — the ledger logic is the same)

```bash
curl -X POST http://localhost:8000/payments \
  -H "Content-Type: application/json" \
  -d '{"account_id": 1, "amount": 200, "reference": "PSP-TEST-001"}'
```

6. **Pull the audit export** — the post-Madlanga selling point made real

```bash
curl http://localhost:8000/municipalities/1/audit-export
```

7. **Log a fault/service ticket**

```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"municipality_id": 1, "account_id": 1, "category": "water", "description": "No water since this morning"}'
```

## What's built (Sprints 1 & 2)

- **Integration Adapter Layer** (`app/ingest.py`) — reads any CSV/Excel
  billing export and normalizes it via a per-municipality column mapping
  into the canonical schema. Re-ingesting the same export monthly builds
  history rather than overwriting it.
- **Account & Billing Engine** (`app/models.py`) — normalized accounts,
  balances, and a transaction ledger per account.
- **Reconciliation + Audit Trail** (`app/reconciliation.py`) — payments
  update balances and write an immutable transaction record; the audit
  export is a flat, exportable record of every transaction, ready to hand
  to a CFO or the Auditor-General.
- **Tickets** — basic fault/service request logging with a citizen-facing
  reference number, tied to an account where available.

## What's next

- Sprint 3: a thin reference web dashboard + WhatsApp flow mockup that
  call this API — for demos, not as the product itself.
- Sprint 4: replace the manual `/payments` endpoint with a real PSP
  webhook (Peach Payments or PayGate sandbox).
