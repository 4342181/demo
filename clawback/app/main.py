"""
Clawback API + app server.

Money mechanic:
  * /api/preview  — free. Returns the opening of the letter + how many lines
                    are hidden. This is the hook: the visitor sees that the
                    letter is good and correctly argued before paying.
  * /api/checkout — creates a Stripe Checkout session for the unlock fee.
                    If Stripe isn't configured, returns a demo unlock token so
                    the whole flow is testable end-to-end without keys.
  * /api/full     — returns the complete letter. Requires a paid token: either
                    a real paid Stripe session id, or the demo token.

The letter content itself never depends on payment — gating is purely about
revealing the full text, so the product is honest (you can read most of it
free) while still converting at the moment of peak desire.
"""
import os
import time
import secrets
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import letters, levers

app = FastAPI(title="Clawback", version="1.0.0")

# ---- Rate limiting ---------------------------------------------------------
# Wide-open API endpoints are how a vibe-coded app earns a surprise $30k bill:
# anyone can hammer /api/preview or /api/checkout and run up cost or take the
# server down. This is a lightweight, dependency-free per-IP fixed-window
# limiter. It's per-process (fine for one worker / a demo); a multi-worker
# production deployment should move this to a shared store like Redis.
RATE_LIMIT = int(os.environ.get("CLAWBACK_RATE_LIMIT", "60"))     # requests…
RATE_WINDOW = int(os.environ.get("CLAWBACK_RATE_WINDOW", "60"))   # …per window (s)
_rate_lock = threading.Lock()
_rate_hits: dict[str, list] = {}  # ip -> [window_start_ts, count]


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        with _rate_lock:
            entry = _rate_hits.get(ip)
            if entry is None or now - entry[0] >= RATE_WINDOW:
                _rate_hits[ip] = [now, 1]
            else:
                entry[1] += 1
                if entry[1] > RATE_LIMIT:
                    retry = max(1, int(RATE_WINDOW - (now - entry[0])))
                    return JSONResponse(
                        {"detail": "Too many requests — please slow down."},
                        status_code=429,
                        headers={"Retry-After": str(retry)},
                    )
            # Bound memory: prune stale IPs so the dict can't grow forever.
            if len(_rate_hits) > 10000:
                for k in [k for k, v in _rate_hits.items() if now - v[0] >= RATE_WINDOW]:
                    _rate_hits.pop(k, None)
    return await call_next(request)

HERE = os.path.dirname(__file__)
STATIC_DIR = os.path.join(HERE, "static")

UNLOCK_PRICE = os.environ.get("CLAWBACK_PRICE", "R49")
# Amount in the smallest currency unit (e.g. cents) for Stripe.
UNLOCK_AMOUNT = int(os.environ.get("CLAWBACK_PRICE_MINOR", "4900"))
UNLOCK_CURRENCY = os.environ.get("CLAWBACK_CURRENCY", "zar")
STRIPE_KEY = os.environ.get("STRIPE_SECRET_KEY")

# In-memory record of demo tokens we've handed out. Fine for a single-process
# demo; a real deployment would verify Stripe sessions statelessly (see below).
_DEMO_TOKENS: set[str] = set()


class GenerateRequest(BaseModel):
    # Every field is length-capped so a malicious/huge payload can't exhaust
    # memory or inflate LLM token cost. Pydantic rejects over-limit input with
    # a 422 before it reaches our code — input validation as the first gate.
    scenario: str = Field(max_length=40)
    region: str = Field(max_length=12)
    company: str = Field("", max_length=200)
    reference: str = Field("", max_length=200)
    date: str = Field("", max_length=80)
    amount: str = Field("", max_length=80)
    route: str = Field("", max_length=200)
    product: str = Field("", max_length=200)
    facts: str = Field("", max_length=2000)
    sender_name: str = Field("", max_length=120)
    sender_contact: str = Field("", max_length=160)
    deadline_days: int = Field(14, ge=1, le=90)
    token: str | None = Field(None, max_length=200)  # required only for /api/full


def _to_inputs(req: GenerateRequest) -> letters.LetterInputs:
    return letters.LetterInputs(
        scenario=req.scenario,
        region=req.region,
        company=req.company,
        reference=req.reference,
        date=req.date,
        amount=req.amount,
        route=req.route,
        product=req.product,
        facts=req.facts,
        sender_name=req.sender_name,
        sender_contact=req.sender_contact,
        deadline_days=req.deadline_days,
    )


@app.get("/api/config")
def config():
    """Drives the form: the scenarios, their fields, and the regions."""
    return {
        "scenarios": {
            k: {"label": v["label"], "blurb": v["blurb"], "fields": v["fields"]}
            for k, v in levers.SCENARIOS.items()
        },
        "regions": {k: v["label"] for k, v in levers.REGIONS.items()},
        "price": UNLOCK_PRICE,
        "payments_live": bool(STRIPE_KEY),
    }


@app.post("/api/preview")
def preview(req: GenerateRequest):
    # Preview is free and shows only the opening lines, so generate the fast
    # deterministic draft — never pay the LLM's latency or cost to render a
    # teaser. LLM polish is reserved for the paid full letter (/api/full).
    result = letters.build_deterministic(_to_inputs(req))
    prev = letters.make_preview(result["body"])
    return {
        "subject": result["subject"],
        "preview": prev["preview"],
        "hidden_lines": prev["hidden_lines"],
        "lever_note": result["lever_note"],
        "deadline": result["deadline"],
        "price": UNLOCK_PRICE,
    }


@app.post("/api/checkout")
def checkout(req: GenerateRequest):
    """Start payment. Returns either a Stripe Checkout URL, or — when Stripe
    isn't configured — a demo token so the flow can be completed locally."""
    if not STRIPE_KEY:
        token = "demo_" + secrets.token_urlsafe(16)
        _DEMO_TOKENS.add(token)
        return {"mode": "demo", "token": token}

    try:
        import stripe

        stripe.api_key = STRIPE_KEY
        base = os.environ.get("CLAWBACK_BASE_URL", "http://localhost:8000")
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": UNLOCK_CURRENCY,
                    "product_data": {"name": "Clawback — unlock your demand letter"},
                    "unit_amount": UNLOCK_AMOUNT,
                },
                "quantity": 1,
            }],
            success_url=base + "/?paid={CHECKOUT_SESSION_ID}",
            cancel_url=base + "/?canceled=1",
        )
        return {"mode": "stripe", "checkout_url": session.url}
    except Exception as exc:
        raise HTTPException(502, f"Could not start checkout: {exc}")


def _token_is_paid(token: str | None) -> bool:
    if not token:
        return False
    if token in _DEMO_TOKENS:
        return True
    if STRIPE_KEY and not token.startswith("demo_"):
        try:
            import stripe

            stripe.api_key = STRIPE_KEY
            session = stripe.checkout.Session.retrieve(token)
            return session.get("payment_status") == "paid"
        except Exception:
            return False
    return False


@app.post("/api/full")
def full(req: GenerateRequest):
    """The paid payload: the complete letter. Requires a paid/demo token."""
    if not _token_is_paid(req.token):
        raise HTTPException(402, "Payment required to unlock the full letter.")
    result = letters.build(_to_inputs(req))
    return {
        "subject": result["subject"],
        "body": result["body"],
        "mode": result["mode"],
        "deadline": result["deadline"],
    }


# ---- Static app (served last so /api/* wins) -------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
