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
    column_mapping = relationship(
        "ColumnMapping", back_populates="municipality", uselist=False
    )


class ColumnMapping(Base):
    """
    Maps a municipality's own export column names onto our canonical schema.
    Every municipality exports billing data differently — this is the piece
    that lets the same ingestion pipeline absorb any of them without a
    custom integration per municipality.
    """
    __tablename__ = "column_mappings"

    id = Column(Integer, primary_key=True, index=True)
    municipality_id = Column(Integer, ForeignKey("municipalities.id"), unique=True)

    # canonical_field -> source CSV column name, e.g.
    # {"account_number": "AcctNo", "resident_name": "CustomerName", ...}
    mapping = Column(JSON, nullable=False)

    municipality = relationship("Municipality", back_populates="column_mapping")


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
