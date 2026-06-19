"""
Integration Adapter Layer — Module 1.

Takes whatever billing export a municipality already produces (a CSV or
Excel dump from Munsoft, SAP, an Oracle report, whatever) and normalizes
it into the canonical Account + Transaction schema, using a per-municipality
column mapping so no two onboardings require custom code.
"""
import io
import uuid
import datetime

import pandas as pd
from sqlalchemy.orm import Session

from . import models

CANONICAL_ACCOUNT_FIELDS = [
    "account_number", "resident_name", "resident_contact",
    "address", "account_type", "balance",
]
CANONICAL_TRANSACTION_FIELDS = [
    "account_number", "date", "type", "amount", "description",
]

REQUIRED_ACCOUNT_FIELDS = {"account_number", "resident_name", "balance"}

CANONICAL_INDIGENT_FIELDS = [
    "applicant_id_number", "applicant_name", "account_number",
    "household_income", "subsidy_category", "status", "expires_at",
]
REQUIRED_INDIGENT_FIELDS = {"applicant_id_number", "applicant_name"}


def read_tabular_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Reads a CSV or Excel file into a DataFrame, regardless of which format
    the municipality happens to export."""
    # dtype=str prevents pandas from inferring numeric types for columns like
    # phone numbers or account codes, which silently strips leading zeros.
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), dtype=str)
    return pd.read_csv(io.BytesIO(file_bytes), dtype=str)


def apply_mapping(df: pd.DataFrame, mapping: dict, canonical_fields: list[str]) -> pd.DataFrame:
    """Renames a municipality's own column names onto our canonical schema.

    `mapping` is canonical_field -> source_column_name. Any canonical field
    not present in the mapping (and not required) is left absent.
    """
    rename_lookup = {source_col: canonical for canonical, source_col in mapping.items()}
    df = df.rename(columns=rename_lookup)

    available = [f for f in canonical_fields if f in df.columns]
    return df[available]


def validate_account_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_ACCOUNT_FIELDS - set(df.columns)
    if missing:
        raise ValueError(
            f"Mapping is missing required field(s): {', '.join(sorted(missing))}. "
            "Check the municipality's column mapping configuration."
        )


def validate_indigent_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_INDIGENT_FIELDS - set(df.columns)
    if missing:
        raise ValueError(
            f"Mapping is missing required field(s): {', '.join(sorted(missing))}. "
            "Check the municipality's indigent column mapping configuration."
        )


def ingest_accounts(
    db: Session,
    municipality: models.Municipality,
    file_bytes: bytes,
    filename: str,
) -> dict:
    """
    Ingests a billing export for a municipality: normalizes it via the
    municipality's column mapping, then upserts each row into the Account
    table (matched on account_number) and appends a Transaction row for
    the balance snapshot, so re-ingesting the same export monthly builds
    a real history rather than silently overwriting it.
    """
    column_mapping = municipality.column_mapping_for("billing")
    if not column_mapping:
        raise ValueError(
            f"No billing column mapping configured for municipality '{municipality.name}'. "
            "Set one up before ingesting data."
        )

    raw_df = read_tabular_file(file_bytes, filename)
    df = apply_mapping(raw_df, column_mapping.mapping, CANONICAL_ACCOUNT_FIELDS)
    validate_account_columns(df)

    created, updated = 0, 0

    for row_index, row in df.iterrows():
        account_number = str(row["account_number"]).strip()
        balance = float(row["balance"]) if not pd.isna(row["balance"]) else 0.0

        account = (
            db.query(models.Account)
            .filter_by(municipality_id=municipality.id, account_number=account_number)
            .first()
        )

        if account is None:
            account = models.Account(
                municipality_id=municipality.id,
                account_number=account_number,
                resident_name=str(row.get("resident_name", "")).strip(),
                resident_contact=str(row.get("resident_contact", "") or ""),
                address=str(row.get("address", "") or ""),
                account_type=str(row.get("account_type", "") or "general"),
                balance=balance,
            )
            db.add(account)
            created += 1
        else:
            account.resident_name = str(row.get("resident_name", account.resident_name)).strip()
            account.balance = balance
            account.updated_at = datetime.datetime.utcnow()
            updated += 1

        db.flush()  # ensures account.id is available for the transaction FK

        db.add(models.Transaction(
            account_id=account.id,
            date=datetime.datetime.utcnow(),
            type="balance_snapshot",
            amount=balance,
            description=f"Imported from {filename}",
            source_reference=f"{filename}:row{row_index}",
        ))

    db.commit()
    return {"created": created, "updated": updated, "rows_processed": len(df)}


def ingest_indigent_register(
    db: Session,
    municipality: models.Municipality,
    file_bytes: bytes,
    filename: str,
) -> dict:
    """
    Ingests an indigent register export the same way `ingest_accounts`
    ingests a billing export: normalize via the municipality's indigent
    column mapping, then upsert each row (matched on applicant_id_number)
    so re-ingesting monthly/quarterly builds history instead of overwriting
    it — that history is itself the audit trail a verification tender asks
    for.
    """
    column_mapping = municipality.column_mapping_for("indigent")
    if not column_mapping:
        raise ValueError(
            f"No indigent column mapping configured for municipality '{municipality.name}'. "
            "Set one up before ingesting data."
        )

    raw_df = read_tabular_file(file_bytes, filename)
    df = apply_mapping(raw_df, column_mapping.mapping, CANONICAL_INDIGENT_FIELDS)
    validate_indigent_columns(df)

    created, updated = 0, 0

    for row_index, row in df.iterrows():
        id_number = str(row["applicant_id_number"]).strip()

        account = None
        account_number = row.get("account_number")
        if account_number and not pd.isna(account_number):
            account = (
                db.query(models.Account)
                .filter_by(municipality_id=municipality.id, account_number=str(account_number).strip())
                .first()
            )

        household_income = row.get("household_income")
        household_income = (
            float(household_income) if household_income is not None and not pd.isna(household_income) else None
        )

        registration = (
            db.query(models.IndigentRegistration)
            .filter_by(municipality_id=municipality.id, applicant_id_number=id_number)
            .first()
        )

        if registration is None:
            registration = models.IndigentRegistration(
                municipality_id=municipality.id,
                account_id=account.id if account else None,
                applicant_id_number=id_number,
                applicant_name=str(row.get("applicant_name", "")).strip(),
                household_income=household_income,
                subsidy_category=str(row.get("subsidy_category", "") or "") or None,
                status=str(row.get("status", "") or "pending"),
                source_reference=f"{filename}:row{row_index}",
            )
            db.add(registration)
            created += 1
        else:
            registration.applicant_name = str(row.get("applicant_name", registration.applicant_name)).strip()
            registration.household_income = household_income
            registration.subsidy_category = str(row.get("subsidy_category", "") or "") or registration.subsidy_category
            registration.status = str(row.get("status", "") or registration.status)
            registration.updated_at = datetime.datetime.utcnow()
            updated += 1

    db.commit()
    return {"created": created, "updated": updated, "rows_processed": len(df)}


def generate_ticket_reference() -> str:
    """A short, citizen-readable reference number for a logged fault/service request."""
    return f"TCK-{uuid.uuid4().hex[:8].upper()}"
