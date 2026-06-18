# Municipal Rail — Sprint 1, 2 & 3

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

## Reference UI (Sprint 3) — what you demo, not the product

Two static, no-build-step pages call the same API directly in a browser.
They're a reference for what a real front end (web dashboard or WhatsApp
BSP integration) would do — not something to ship to residents as-is.

1. With the API running on `localhost:8000`, open
   `municipal-rail/dashboard/index.html` directly in a browser (e.g.
   `open municipal-rail/dashboard/index.html`). Enter a municipality ID
   and account number (e.g. `1` and `SW-00123` from the sample export)
   to see balance, payment history, a mocked "pay now", and fault logging.
2. Open `municipal-rail/dashboard/whatsapp_mockup.html` the same way for a
   WhatsApp-style chat mockup of the same flows (`balance`, `pay`,
   `report fault`) — a stand-in for a real WhatsApp Business API/BSP
   integration, which is Sprint 4+ scope.

Both pages call `http://localhost:8000` by default; the API has CORS
enabled (`app/main.py`) so they can be opened as local files. Behind the
scenes, both use the same account/payment/ticket endpoints — including a
new `GET /municipalities/{id}/accounts/lookup?account_number=...` for
looking accounts up the way a resident actually identifies their account
(by account number on their bill, not our internal database id).

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

- Sprint 4: replace the manual `/payments` endpoint with a real PSP
  webhook (Peach Payments or PayGate sandbox), and replace the WhatsApp
  mockup with a real WhatsApp Business API/BSP sandbox integration.
