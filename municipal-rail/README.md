# Municipal Rail — Sprint 1, 2, 3 & 4

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

5. **Record a payment manually** (for admin/cash-at-office entries —
   resident-initiated payments go through `/payments/initiate` instead,
   see Sprint 4 below)

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

## PSP integration (Sprint 4) — PayGate PayWeb3

Resident-initiated payments now go through a real PSP instead of a
manual call:

```bash
curl -X POST http://localhost:8000/payments/initiate \
  -H "Content-Type: application/json" \
  -d '{"account_id": 1, "amount": 200, "return_url": "https://example.org/return"}'
```

This returns a `redirect_url` to PayGate's hosted payment page. PayGate
then calls our `/payments/notify` webhook server-to-server once the
resident pays — that webhook, not the browser redirect, is the only
thing that updates the account balance, and it's checksum-verified and
idempotent against PayGate retrying the notification.

The dashboard and WhatsApp mockup (Sprint 3) both call
`/payments/initiate` for their "pay now" / "pay" flows and poll
`/payments/{id}/status` afterwards.

**Sandbox setup notes:**
- Defaults to PayGate's published test merchant
  (`PAYGATE_ID=10011072154`, key `secret`) — override with the
  `PAYGATE_ID` / `PAYGATE_ENCRYPTION_KEY` env vars for a real account.
- The checksum algorithm and ledger/idempotency logic
  (`app/psp.py`, `app/reconciliation.py`) are unit-tested in
  `tests/test_psp.py` (`pytest tests/`) and were also exercised
  end-to-end by simulating a PayGate notify POST directly. The actual
  HTTP round trip to PayGate's initiate endpoint has NOT been verified
  against a live sandbox session — this couldn't be done from the
  environment this was built in, which has no outbound internet access
  beyond a small allowlist. Test it from a normal machine first (see
  below) before relying on it.

### Testing the real PayGate flow locally

This needs to run somewhere with normal internet access (your laptop —
not a network-locked CI/cloud sandbox), plus a tunnel so PayGate can
reach your local webhook.

1. **Start the API**:
   ```bash
   ./scripts/run_local.sh
   ```
   This creates the venv, installs dependencies, and starts uvicorn on
   `localhost:8000`.
2. **Tunnel it** (in another terminal) so PayGate's servers can call your
   notify webhook:
   ```bash
   npx ngrok http 8000
   ```
   Note the `https://xxxx.ngrok-free.app` URL it gives you — PayGate
   only ever talks to that tunnel URL, not `localhost`, so `NOTIFY_URL`
   (derived from the request's own base URL in `app/main.py`) needs
   requests to come in through the tunnel.
3. **Seed the demo data through the tunnel**, so the municipality and
   accounts exist on the running instance:
   ```bash
   API_BASE=https://xxxx.ngrok-free.app ./scripts/seed_demo_data.sh
   ```
4. **Open the dashboard pointed at the tunnel** — both reference pages
   read an `?api=` query param, so no manual edits needed:
   ```
   dashboard/index.html?api=https://xxxx.ngrok-free.app
   ```
   Look up account `SW-00123` and click **Pay now**.
5. You should land on PayGate's actual hosted sandbox payment page.
   Complete a test payment there (PayGate's sandbox accepts dummy card
   details — see their PayWeb3 docs for the current test card numbers).
6. PayGate will call your `/payments/notify` webhook via the tunnel.
   Confirm in the API logs that the checksum verified and the
   transaction was recorded, then refresh the dashboard — the balance
   should reflect the payment.
7. If anything fails at step 4 (PayGate initiate) or step 6 (the
   notify checksum), that's the signal to re-check the exact field
   list/order against PayGate's current PayWeb3 integration guide —
   checksum mismatches are the most common failure mode with this kind
   of API, and the field set in `app/psp.py` was written from the
   documented spec but never confirmed against a live response.

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

- Verify the PayGate PayWeb3 integration against a live sandbox session
  (this environment can't make outbound calls to PayGate to test that
  part directly — see Sprint 4 notes above).
- Replace the WhatsApp mockup with a real WhatsApp Business API/BSP
  sandbox integration.
