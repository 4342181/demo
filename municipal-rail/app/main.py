"""
The Rail — core platform entry point.

Exposes the integration adapter, account/billing engine, reconciliation +
audit trail, and ticket logging as an API that any front end (your own
reference UI, a WhatsApp flow, or a partner platform like GovChat / My
Smart City) can plug into.
"""
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from . import models, schemas, ingest, reconciliation
from .database import Base, engine, get_db

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Municipal Rail API", version="0.1.0")


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


# ---------- Payment Processing (Module 3 — mocked until PSP integration) ----------

@app.post("/payments")
def record_payment(payload: schemas.PaymentCreate, db: Session = Depends(get_db)):
    account = db.query(models.Account).get(payload.account_id)
    if not account:
        raise HTTPException(404, "Account not found")

    transaction = reconciliation.record_payment(
        db, account, payload.amount, payload.reference, payload.description
    )
    return {"transaction_id": transaction.id, "new_balance": account.balance}


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
