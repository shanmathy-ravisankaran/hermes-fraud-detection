"""Shared request schemas used by single and batch scoring."""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_amount: float = Field(gt=0)
    transaction_dt: int = Field(ge=0)
    product_cd: str
    card4: str = "missing"
    card6: str = "missing"
    email_domain: str = "missing"
