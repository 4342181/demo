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
