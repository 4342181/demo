import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.orm import relationship

from .database import Base


class Municipality(Base):
    """A municipal customer onboarded onto the rail."""
    __tablename__ = "municipalities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    accounts = relationship("Account", back_populates="municipality")
    column_mappings = relationship("ColumnMapping", back_populates="municipality")
    indigent_registrations = relationship("IndigentRegistration", back_populates="municipality")

    def column_mapping_for(self, register_type: str) -> "ColumnMapping | None":
        return next(
            (m for m in self.column_mappings if m.register_type == register_type), None
        )


class ColumnMapping(Base):
    """
    Maps a municipality's own export column names onto our canonical schema.
    Every municipality exports billing data differently — this is the piece
    that lets the same ingestion pipeline absorb any of them without a
    custom integration per municipality.

    `register_type` distinguishes which canonical schema this mapping is
    for ("billing" or "indigent"), since a municipality's billing export
    and its indigent-register export rarely share a source system, let
    alone the same columns.
    """
    __tablename__ = "column_mappings"

    id = Column(Integer, primary_key=True, index=True)
    municipality_id = Column(Integer, ForeignKey("municipalities.id"))
    register_type = Column(String, nullable=False, default="billing")

    # canonical_field -> source CSV column name, e.g.
    # {"account_number": "AcctNo", "resident_name": "CustomerName", ...}
    mapping = Column(JSON, nullable=False)

    municipality = relationship("Municipality", back_populates="column_mappings")

    __table_args__ = (
        UniqueConstraint("municipality_id", "register_type", name="uq_municipality_register_type"),
    )


class Account(Base):
    """A single resident municipal account (water/electricity/rates/general)."""
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    municipality_id = Column(Integer, ForeignKey("municipalities.id"), nullable=False)

    account_number = Column(String, nullable=False, index=True)
    resident_name = Column(String, nullable=False)
    resident_contact = Column(String, nullable=True)
    address = Column(String, nullable=True)
    account_type = Column(String, nullable=True)  # water, electricity, rates, general
    balance = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    municipality = relationship("Municipality", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account")
    tickets = relationship("Ticket", back_populates="account")

    __table_args__ = (
        UniqueConstraint("municipality_id", "account_number", name="uq_municipality_account"),
    )


class Transaction(Base):
    """
    A charge or payment on an account. This is the reconciliation ledger —
    every entry is immutable once written, which is what makes the audit
    trail meaningful.
    """
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)

    date = Column(DateTime, nullable=False)
    type = Column(String, nullable=False)  # "charge" or "payment"
    amount = Column(Float, nullable=False)
    description = Column(String, nullable=True)

    # traceability back to the original source row, for audit purposes
    source_reference = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    account = relationship("Account", back_populates="transactions")


class PendingPayment(Base):
    """
    A payment initiated through the PSP (PayGate PayWeb3) but not yet
    confirmed. Created when the resident clicks "pay", resolved when the
    PSP's notify webhook confirms or fails it. Tracking this separately
    from Transaction means a resident closing their browser mid-payment
    doesn't leave the ledger in an ambiguous state.
    """
    __tablename__ = "pending_payments"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)

    amount = Column(Float, nullable=False)
    reference = Column(String, nullable=False, unique=True)  # our reference, sent to PSP
    pay_request_id = Column(String, nullable=True, index=True)  # PSP's id, set after initiate
    status = Column(String, default="pending")  # pending, completed, failed
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    account = relationship("Account")


class IndigentRegistration(Base):
    """
    A resident's registration on the municipality's indigent register —
    qualifying them for subsidized/free basic services. This is the object
    municipalities keep re-tendering verification systems for: registers
    decay (households' income changes, people move, IDs are never checked
    against Home Affairs), so without recurring verification the register
    itself becomes the audit finding ("indigent register lacks a vetting
    system").
    """
    __tablename__ = "indigent_registrations"

    id = Column(Integer, primary_key=True, index=True)
    municipality_id = Column(Integer, ForeignKey("municipalities.id"), nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)

    applicant_id_number = Column(String, nullable=False, index=True)
    applicant_name = Column(String, nullable=False)
    household_income = Column(Float, nullable=True)
    subsidy_category = Column(String, nullable=True)  # water, electricity, rates, full
    status = Column(String, default="pending")  # pending, active, expired, rejected

    registered_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # traceability back to the original source row, for audit purposes
    source_reference = Column(String, nullable=True)

    municipality = relationship("Municipality", back_populates="indigent_registrations")
    account = relationship("Account")
    verification_events = relationship("VerificationEvent", back_populates="registration")

    __table_args__ = (
        UniqueConstraint(
            "municipality_id", "applicant_id_number", name="uq_municipality_applicant"
        ),
    )


class VerificationEvent(Base):
    """
    An immutable record of a single check run against an indigent
    registration — an ID/Home Affairs lookup, an income re-assessment, a
    cross-check against the supplier/government-employee database, or a
    manual review. This is the audit trail tenders explicitly ask for
    ("must provide detailed audit trails... who/what/when").
    """
    __tablename__ = "verification_events"

    id = Column(Integer, primary_key=True, index=True)
    registration_id = Column(Integer, ForeignKey("indigent_registrations.id"), nullable=False)

    method = Column(String, nullable=False)  # id_check, income_check, manual_review, data_cleanse
    result = Column(String, nullable=False)  # pass, fail, flagged
    notes = Column(String, nullable=True)
    performed_by = Column(String, nullable=True)

    occurred_at = Column(DateTime, default=datetime.datetime.utcnow)

    registration = relationship("IndigentRegistration", back_populates="verification_events")


class Ticket(Base):
    """A logged service/fault request tied to an account or resident."""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    municipality_id = Column(Integer, ForeignKey("municipalities.id"), nullable=False)

    category = Column(String, nullable=False)  # water, electricity, roads, waste, other
    description = Column(String, nullable=False)
    status = Column(String, default="open")  # open, in_progress, resolved
    reference_number = Column(String, unique=True, nullable=False)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    account = relationship("Account", back_populates="tickets")
