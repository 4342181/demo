"""
The Rail — core platform entry point.

Exposes the integration adapter, account/billing engine, reconciliation +
audit trail, and ticket logging as an API that any front end (your own
reference UI, a WhatsApp flow, or a partner platform like GovChat / My
Smart City) can plug into.
"""
import uuid

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import models, schemas, ingest, reconciliation, psp
from .database import Base, engine, get_db

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Municipal Rail API", version="0.1.0")

# The reference dashboard and WhatsApp mockup (Sprint 3) are static pages
# opened directly in a browser, so they call this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Municipalities & onboarding ----------

@app.post("/municipalities")
def create_municipality(payload: schemas.MunicipalityCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Municipality).filter_by(name=payload.name).first()
    if existing:
        raise HTTPException(400, "Municipality already exists")

    municipality = models.Municipality(name=payload.name)
    db.add(municipality)
    db.commit()
    db.refresh(municipality)
    return {"id": municipality.id, "name": municipality.name}


@app.get("/municipalities")
def list_municipalities(db: Session = Depends(get_db)):
    return db.query(models.Municipality).all()


@app.post("/municipalities/{municipality_id}/column-mapping")
def set_column_mapping(municipality_id: int, payload: schemas.ColumnMappingCreate, db: Session = Depends(get_db)):
    municipality = db.query(models.Municipality).get(municipality_id)
    if not municipality:
        raise HTTPException(404, "Municipality not found")

    if municipality.column_mapping:
        municipality.column_mapping.mapping = payload.mapping
    else:
        db.add(models.ColumnMapping(municipality_id=municipality_id, mapping=payload.mapping))

    db.commit()
    return {"municipality_id": municipality_id, "mapping": payload.mapping}


# ---------- Integration Adapter Layer (Module 1) ----------

@app.post("/municipalities/{municipality_id}/ingest")
async def ingest_billing_export(
    municipality_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    municipality = db.query(models.Municipality).get(municipality_id)
    if not municipality:
        raise HTTPException(404, "Municipality not found")

    file_bytes = await file.read()
    try:
        result = ingest.ingest_accounts(db, municipality, file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return result


# ---------- Account & Billing Engine (Module 2) ----------

@app.get("/municipalities/{municipality_id}/accounts")
def list_accounts(municipality_id: int, db: Session = Depends(get_db)):
    return db.query(models.Account).filter_by(municipality_id=municipality_id).all()


@app.get("/municipalities/{municipality_id}/accounts/lookup")
def lookup_account(municipality_id: int, account_number: str, db: Session = Depends(get_db)):
    """Citizen-facing lookup by account number — what a resident actually has
    on their bill, as opposed to our internal database id."""
    account = (
        db.query(models.Account)
        .filter_by(municipality_id=municipality_id, account_number=account_number.strip())
        .first()
    )
    if not account:
        raise HTTPException(404, "Account not found")
    return account


@app.get("/accounts/{account_id}")
def get_account(account_id: int, db: Session = Depends(get_db)):
    account = db.query(models.Account).get(account_id)
    if not account:
        raise HTTPException(404, "Account not found")
    return account


@app.get("/accounts/{account_id}/history")
def get_account_history(account_id: int, db: Session = Depends(get_db)):
    account = db.query(models.Account).get(account_id)
    if not account:
        raise HTTPException(404, "Account not found")

    transactions = (
        db.query(models.Transaction)
        .filter_by(account_id=account_id)
        .order_by(models.Transaction.date.desc())
        .all()
    )
    return {"account": account, "transactions": transactions}


# ---------- Payment Processing (Module 3) ----------

@app.post("/payments")
def record_payment_manual(payload: schemas.PaymentCreate, db: Session = Depends(get_db)):
    """Manual/admin entry point — e.g. cash paid at a municipal office.
    Resident-initiated payments should go through /payments/initiate
    instead, which routes through the PSP."""
    account = db.query(models.Account).get(payload.account_id)
    if not account:
        raise HTTPException(404, "Account not found")

    transaction = reconciliation.record_payment(
        db, account, payload.amount, payload.reference, payload.description
    )
    return {"transaction_id": transaction.id, "new_balance": account.balance}


@app.post("/payments/initiate")
async def initiate_payment(payload: schemas.PaymentInitiate, request: Request, db: Session = Depends(get_db)):
    """
    Starts a PSP (PayGate PayWeb3) payment. Creates a PendingPayment,
    calls PayGate's initiate endpoint, and returns the URL the resident's
    browser should be redirected to. The account balance is NOT updated
    here — only the /payments/notify webhook, once PayGate confirms the
    transaction, can do that.
    """
    account = db.query(models.Account).get(payload.account_id)
    if not account:
        raise HTTPException(404, "Account not found")

    reference = f"PG-{uuid.uuid4().hex[:12]}"
    notify_url = str(request.base_url).rstrip("/") + "/payments/notify"

    try:
        result = await psp.initiate_payment(
            reference=reference,
            amount=payload.amount,
            return_url=payload.return_url,
            notify_url=notify_url,
            email=payload.email,
        )
    except Exception as exc:
        raise HTTPException(502, f"Could not initiate payment with PSP: {exc}")

    pending = models.PendingPayment(
        account_id=account.id,
        amount=payload.amount,
        reference=reference,
        pay_request_id=result["pay_request_id"],
        status="pending",
    )
    db.add(pending)
    db.commit()

    return {"pending_payment_id": pending.id, "redirect_url": result["redirect_url"]}


@app.post("/payments/notify")
async def payment_notify(request: Request, db: Session = Depends(get_db)):
    """
    PayGate's server-to-server webhook, called once the resident completes
    (or abandons) the payment on PayGate's side. This — not the browser
    redirect — is the only trusted source for confirming a payment, so the
    ledger is only ever updated here, after the checksum is verified.
    """
    form = await request.form()
    fields = dict(form)

    if not psp.verify_notify_checksum(fields):
        raise HTTPException(400, "Checksum verification failed")

    pending = (
        db.query(models.PendingPayment)
        .filter_by(pay_request_id=fields.get("PAY_REQUEST_ID"))
        .first()
    )
    if not pending:
        raise HTTPException(404, "Unknown PAY_REQUEST_ID")

    if pending.status == "completed":
        return {"status": "already_processed"}

    if not psp.is_approved(fields):
        pending.status = "failed"
        db.commit()
        return {"status": "failed"}

    account = db.query(models.Account).get(pending.account_id)
    transaction = reconciliation.record_payment(
        db, account, pending.amount, pending.reference, "Resident payment via PayGate"
    )
    pending.status = "completed"
    pending.transaction_id = transaction.id
    db.commit()

    return {"status": "completed"}


@app.get("/payments/{pending_payment_id}/status")
def payment_status(pending_payment_id: int, db: Session = Depends(get_db)):
    """Lets the dashboard/WhatsApp flow poll for confirmation after the
    resident returns from PayGate, since the notify webhook may land
    slightly after the browser redirect."""
    pending = db.query(models.PendingPayment).get(pending_payment_id)
    if not pending:
        raise HTTPException(404, "Pending payment not found")
    return {"status": pending.status, "reference": pending.reference}


# ---------- Audit & Compliance Trail (Module 4) ----------

@app.get("/municipalities/{municipality_id}/audit-export")
def audit_export(municipality_id: int, db: Session = Depends(get_db)):
    municipality = db.query(models.Municipality).get(municipality_id)
    if not municipality:
        raise HTTPException(404, "Municipality not found")
    return reconciliation.build_audit_export(db, municipality_id)


# ---------- Tickets / fault reporting ----------

@app.post("/tickets")
def create_ticket(payload: schemas.TicketCreate, db: Session = Depends(get_db)):
    ticket = models.Ticket(
        municipality_id=payload.municipality_id,
        account_id=payload.account_id,
        category=payload.category,
        description=payload.description,
        reference_number=ingest.generate_ticket_reference(),
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/municipalities/{municipality_id}/tickets")
def list_tickets(municipality_id: int, db: Session = Depends(get_db)):
    return db.query(models.Ticket).filter_by(municipality_id=municipality_id).all()


@app.patch("/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: int, status: str, db: Session = Depends(get_db)):
    ticket = db.query(models.Ticket).get(ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    ticket.status = status
    db.commit()
    return ticket
