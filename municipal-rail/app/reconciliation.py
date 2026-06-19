"""
Reconciliation + Audit Trail — Module 2 & 4.

Records payments against accounts and produces an immutable, exportable
audit trail: every transaction is logged, time-stamped, and traceable.
"""
import datetime

from sqlalchemy.orm import Session

from . import models


def record_payment(
    db: Session,
    account: models.Account,
    amount: float,
    reference: str,
    description: str = "Resident payment",
) -> models.Transaction:
    """
    Records a payment against an account and updates its balance.
    Idempotent on `reference`: PSPs retry webhook notifications, so if a
    transaction with this source_reference already exists, that existing
    transaction is returned instead of double-crediting the account.
    """
    existing = (
        db.query(models.Transaction)
        .filter_by(account_id=account.id, source_reference=reference, type="payment")
        .first()
    )
    if existing:
        return existing

    account.balance = round(account.balance - amount, 2)
    account.updated_at = datetime.datetime.utcnow()

    transaction = models.Transaction(
        account_id=account.id,
        date=datetime.datetime.utcnow(),
        type="payment",
        amount=amount,
        description=description,
        source_reference=reference,
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction


def record_verification(
    db: Session,
    registration: models.IndigentRegistration,
    method: str,
    result: str,
    notes: str | None = None,
    performed_by: str | None = None,
) -> models.VerificationEvent:
    """
    Records a verification check (ID lookup, income re-assessment, manual
    review, data cleanse) against an indigent registration. Each event is
    immutable once written — the running history across events is the
    audit trail tenders ask for ("who/what/when").
    """
    event = models.VerificationEvent(
        registration_id=registration.id,
        method=method,
        result=result,
        notes=notes,
        performed_by=performed_by,
    )
    db.add(event)

    if result == "fail":
        registration.status = "rejected"
    elif result == "flagged" and registration.status == "active":
        registration.status = "pending"
    elif result == "pass" and registration.status in ("pending", "expired"):
        registration.status = "active"
    registration.updated_at = datetime.datetime.utcnow()

    db.commit()
    db.refresh(event)
    return event


def build_indigent_audit_export(db: Session, municipality_id: int) -> list[dict]:
    """
    Flat, exportable audit record of every verification event run against
    every indigent registration for a municipality — the same shape as
    `build_audit_export` below, but for the indigent register rather than
    the billing ledger.
    """
    rows = (
        db.query(models.VerificationEvent, models.IndigentRegistration)
        .join(
            models.IndigentRegistration,
            models.VerificationEvent.registration_id == models.IndigentRegistration.id,
        )
        .filter(models.IndigentRegistration.municipality_id == municipality_id)
        .order_by(models.VerificationEvent.occurred_at.desc())
        .all()
    )

    return [
        {
            "verification_event_id": event.id,
            "occurred_at": event.occurred_at.isoformat(),
            "applicant_id_number": registration.applicant_id_number,
            "applicant_name": registration.applicant_name,
            "registration_status": registration.status,
            "method": event.method,
            "result": event.result,
            "performed_by": event.performed_by,
            "notes": event.notes,
        }
        for event, registration in rows
    ]


def build_audit_export(db: Session, municipality_id: int) -> list[dict]:
    """
    Produces a flat, exportable audit record: every transaction across
    every account for a municipality, with enough context that a CFO
    could hand this directly to the Auditor-General.
    """
    rows = (
        db.query(models.Transaction, models.Account)
        .join(models.Account, models.Transaction.account_id == models.Account.id)
        .filter(models.Account.municipality_id == municipality_id)
        .order_by(models.Transaction.date.desc())
        .all()
    )

    return [
        {
            "transaction_id": txn.id,
            "date": txn.date.isoformat(),
            "account_number": account.account_number,
            "resident_name": account.resident_name,
            "type": txn.type,
            "amount": txn.amount,
            "description": txn.description,
            "source_reference": txn.source_reference,
        }
        for txn, account in rows
    ]
