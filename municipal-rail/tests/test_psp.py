"""
Unit tests for the PayGate checksum logic — the part of the PSP
integration that can be verified without an outbound network call.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import psp


def test_build_initiate_fields_includes_valid_checksum():
    fields = psp.build_initiate_fields(
        reference="REF-1",
        amount=100.50,
        return_url="https://example.org/return",
        notify_url="https://example.org/notify",
    )
    checksum = fields.pop("CHECKSUM")
    recomputed = psp._checksum(list(fields.values()))
    assert checksum == recomputed


def test_amount_converted_to_cents():
    fields = psp.build_initiate_fields(
        reference="REF-2",
        amount=100.50,
        return_url="https://example.org/return",
        notify_url="https://example.org/notify",
    )
    assert fields["AMOUNT"] == "10050"


def test_verify_notify_checksum_accepts_correctly_signed_payload():
    fields = {"PAYGATE_ID": "10011072154", "PAY_REQUEST_ID": "abc123", "TRANSACTION_STATUS": "1"}
    fields["CHECKSUM"] = psp._checksum(list(fields.values()))
    assert psp.verify_notify_checksum(fields) is True


def test_verify_notify_checksum_rejects_tampered_payload():
    fields = {"PAYGATE_ID": "10011072154", "PAY_REQUEST_ID": "abc123", "TRANSACTION_STATUS": "1"}
    fields["CHECKSUM"] = psp._checksum(list(fields.values()))
    fields["TRANSACTION_STATUS"] = "0"  # tampered after signing
    assert psp.verify_notify_checksum(fields) is False


def test_is_approved():
    assert psp.is_approved({"TRANSACTION_STATUS": "1"}) is True
    assert psp.is_approved({"TRANSACTION_STATUS": "2"}) is False
    assert psp.is_approved({}) is False
