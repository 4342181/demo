"""
PSP integration — Module 3 (Sprint 4).

Implements PayGate's PayWeb3 protocol: we POST a payment request to
PayGate's initiate endpoint, redirect the resident to PayGate to pay,
and PayGate calls our /payments/notify webhook (server-to-server) to
confirm the result. The ledger is only ever updated from the verified
webhook, never from the redirect — the redirect is just where the
resident's browser ends up, and is not trustworthy on its own.

NOTE: this environment cannot make outbound calls to PayGate's servers
(network is restricted to an allowlist), so the actual HTTP round trip to
PAYGATE_INITIATE_URL has not been exercised against a live sandbox. The
checksum algorithm and field set below follow PayGate's published PayWeb3
integration guide; before going live, re-verify the exact field list,
field order, and date format against the current PayGate developer docs
and a real sandbox transaction, since checksum mismatches are the most
common integration failure with this kind of API.

Sandbox-only test credentials (PayGate's permanent published test
merchant, safe to use for development without registering):
  PAYGATE_ID = 10011072130
  ENCRYPTION_KEY = secret
"""
import hashlib
import datetime
import os

import httpx

PAYGATE_ID = os.environ.get("PAYGATE_ID", "10011072130")
ENCRYPTION_KEY = os.environ.get("PAYGATE_ENCRYPTION_KEY", "secret")

PAYGATE_INITIATE_URL = "https://secure.paygate.co.za/payweb3/initiate.trans"
PAYGATE_PROCESS_URL = "https://secure.paygate.co.za/payweb3/process.trans"


def _checksum(ordered_values: list[str]) -> str:
    """PayWeb3 checksum: MD5 of all field values (in request order,
    excluding the CHECKSUM field itself) concatenated with the encryption
    key."""
    return hashlib.md5(("".join(ordered_values) + ENCRYPTION_KEY).encode("utf-8")).hexdigest()


def build_initiate_fields(
    reference: str,
    amount: float,
    return_url: str,
    notify_url: str,
    email: str = "resident@example.org",
) -> dict:
    """Builds the form fields for the PayWeb3 initiate request, including
    the checksum. `amount` must be in cents per PayGate's spec."""
    fields = {
        "PAYGATE_ID": PAYGATE_ID,
        "REFERENCE": reference,
        "AMOUNT": str(int(round(amount * 100))),
        "CURRENCY": "ZAR",
        "RETURN_URL": return_url,
        "TRANSACTION_DATE": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "LOCALE": "en-za",
        "COUNTRY": "ZAF",
        "EMAIL": email,
        "NOTIFY_URL": notify_url,
    }
    fields["CHECKSUM"] = _checksum(list(fields.values()))
    return fields


async def initiate_payment(
    reference: str,
    amount: float,
    return_url: str,
    notify_url: str,
    email: str = "resident@example.org",
) -> dict:
    """
    Calls PayGate's initiate endpoint and returns the PAY_REQUEST_ID +
    checksum needed to redirect the resident to PAYGATE_PROCESS_URL.
    Raises if PayGate rejects the request or the response checksum fails.
    """
    fields = build_initiate_fields(reference, amount, return_url, notify_url, email)

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(PAYGATE_INITIATE_URL, data=fields)
        response.raise_for_status()

    # Logged so a checksum/field mismatch (PayGate's DATA_CHK) can be
    # diagnosed from the request we sent vs. the raw response we got back.
    print("[psp] initiate request fields:", fields)
    print("[psp] initiate raw response:", response.text)

    parsed = dict(item.split("=", 1) for item in response.text.split("&") if "=" in item)

    if parsed.get("ERROR"):
        raise ValueError(f"PayGate rejected the payment request: {parsed['ERROR']}")

    pay_request_id = parsed["PAY_REQUEST_ID"]
    checksum = parsed["CHECKSUM"]

    # PayWeb3 initiate-response checksum is MD5 of
    # PAYGATE_ID + PAY_REQUEST_ID + REFERENCE + encryption key.
    expected = _checksum([PAYGATE_ID, pay_request_id, parsed.get("REFERENCE", reference)])
    if checksum != expected:
        raise ValueError("PayGate initiate response failed checksum verification")

    return {
        "pay_request_id": pay_request_id,
        "checksum": checksum,
        "redirect_url": f"{PAYGATE_PROCESS_URL}?PAY_REQUEST_ID={pay_request_id}&CHECKSUM={checksum}",
    }


def verify_notify_checksum(fields: dict) -> bool:
    """
    Verifies the checksum on an inbound PayGate notify (webhook) POST.
    `fields` is the parsed form body, including the CHECKSUM field PayGate
    sent. We recompute it over every other field, in the order PayGate
    sent them, and compare.
    """
    if "CHECKSUM" not in fields:
        return False

    received_checksum = fields["CHECKSUM"]
    ordered_values = [v for k, v in fields.items() if k != "CHECKSUM"]
    expected = _checksum(ordered_values)
    return expected == received_checksum


def is_approved(fields: dict) -> bool:
    """PayGate sends TRANSACTION_STATUS=1 for an approved transaction."""
    return fields.get("TRANSACTION_STATUS") == "1"
