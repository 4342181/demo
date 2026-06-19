from pydantic import BaseModel


class MunicipalityCreate(BaseModel):
    name: str


class ColumnMappingCreate(BaseModel):
    municipality_id: int
    mapping: dict[str, str]  # canonical_field -> source_column_name


class PaymentCreate(BaseModel):
    account_id: int
    amount: float
    reference: str
    description: str | None = "Resident payment"


class PaymentInitiate(BaseModel):
    account_id: int
    amount: float
    return_url: str  # where the resident's browser lands after paying
    email: str | None = "resident@example.org"


class TicketCreate(BaseModel):
    municipality_id: int
    account_id: int | None = None
    category: str
    description: str


class VerificationEventCreate(BaseModel):
    method: str  # id_check, income_check, manual_review, data_cleanse
    result: str  # pass, fail, flagged
    notes: str | None = None
    performed_by: str | None = None
