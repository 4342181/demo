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
import logging
import secrets
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import letters, levers

logger = logging.getLogger("clawback")

app = FastAPI(title="Clawback", version="1.0.0")


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    # Never leak raw errors/stack traces to clients — that hands an attacker a
    # map of your internals. Log the real cause server-side, return a generic
    # message. (HTTPExceptions we raise deliberately keep their own clean text.)
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        {"detail": "Something went wrong. Please try again."}, status_code=500
    )

# ---- Rate limiting ---------------------------------------------------------
# Wide-open API endpoints are how a vibe-coded app earns a surprise $30k bill:
# anyone can hammer /api/preview or /api/checkout and run up cost or take the
# server down. This is a lightweight, dependency-free per-IP fixed-window
# limiter. It's per-process (fine for one worker / a demo); a multi-worker
# production deployment should move this to a shared store like Redis.
RATE_LIMIT = int(os.environ.get("CLAWBACK_RATE_LIMIT", "60"))     # requests…
RATE_WINDOW = int(os.environ.get("CLAWBACK_RATE_WINDOW", "60"))   # …per window (s)
_rate_lock = threading.Lock()
# Sliding-window counter: ip -> [window_index, count_this_window, count_prev_window].
# A plain fixed window lets a client burst the full limit at the end of one
# window and again at the start of the next (2×limit in seconds). Weighting the
# previous window's count by how much it still overlaps "now" smooths that
# boundary out — with just two counters per IP (cheap, in-process).
_rate_hits: dict[str, list] = {}


def _rate_check(ip: str, now: float) -> bool:
    """True if the request is allowed; records it. Sliding-window counter."""
    window = int(now // RATE_WINDOW)
    weight = (RATE_WINDOW - (now - window * RATE_WINDOW)) / RATE_WINDOW  # 1.0→0 across window
    entry = _rate_hits.get(ip)
    if entry is None or entry[0] < window - 1:        # no usable history
        _rate_hits[ip] = [window, 1, 0]
        return True
    if entry[0] == window - 1:                        # advanced one window: roll current→prev
        entry[0], entry[1], entry[2] = window, 0, entry[1]
    if entry[1] + entry[2] * weight >= RATE_LIMIT:    # weighted estimate over the sliding window
        return False
    entry[1] += 1
    return True


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        with _rate_lock:
            allowed = _rate_check(ip, now)
            if len(_rate_hits) > 10000:               # bound memory: drop stale IPs
                cutoff = int(now // RATE_WINDOW) - 1
                for k in [k for k, v in _rate_hits.items() if v[0] < cutoff]:
                    _rate_hits.pop(k, None)
        if not allowed:
            retry = max(1, int(RATE_WINDOW - (now % RATE_WINDOW)))
            return JSONResponse(
                {"detail": "Too many requests — please slow down."},
                status_code=429,
                headers={"Retry-After": str(retry)},
            )
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

# Tokens already spent on an unlock. One payment = one letter, so a paid token
# is single-use: this stops anyone replaying the same token (or a leaked one)
# to mint unlimited letters from a single payment. Per-process like the rest;
# move to Redis/DB for multi-worker production.
_USED_TOKENS: set[str] = set()


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

    # Whitelist the enums server-side: reject unexpected values with a 422
    # rather than silently coercing them to a default. "Correct AND expected."
    @field_validator("scenario")
    @classmethod
    def _known_scenario(cls, v: str) -> str:
        if v not in levers.SCENARIOS:
            raise ValueError("unknown scenario")
        return v

    @field_validator("region")
    @classmethod
    def _known_region(cls, v: str) -> str:
        if v not in levers.REGIONS:
            raise ValueError("unknown region")
        return v


@lru_cache(maxsize=1)
def _form_schema() -> dict:
    """The scenarios/regions/fields never change at runtime, so build this once
    instead of rebuilding it on every page load (config is the hottest read).
    Cache-aside applied to our busiest endpoint — safe because it's immutable,
    with no DB and no staleness. Payment flags below stay live (we never cache
    payment state)."""
    return {
        "scenarios": {
            k: {"label": v["label"], "blurb": v["blurb"], "fields": v["fields"]}
            for k, v in levers.SCENARIOS.items()
        },
        "regions": {k: v["label"] for k, v in levers.REGIONS.items()},
    }


@app.get("/api/config")
def config():
    """Drives the form: the scenarios, their fields, and the regions."""
    return {**_form_schema(), "price": UNLOCK_PRICE, "payments_live": bool(STRIPE_KEY)}


@app.post("/api/preview")
def preview(req: GenerateRequest):
    # Preview is free and shows only the opening lines, so generate the fast
    # deterministic draft — never pay the LLM's latency or cost to render a
    # teaser. LLM polish is reserved for the paid full letter (/api/full).
    result = letters.build_deterministic(req)
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
    except Exception:
        # Log the real Stripe error for ourselves; tell the client nothing
        # about our internals.
        logger.exception("Stripe checkout failed")
        raise HTTPException(502, "Could not start checkout. Please try again.")


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
    """The paid payload: the complete letter. Requires a paid/demo token,
    which is single-use — it can't be replayed for a second free letter."""
    if req.token in _USED_TOKENS:
        raise HTTPException(402, "This unlock has already been used. Please pay to unlock another letter.")
    if not _token_is_paid(req.token):
        raise HTTPException(402, "Payment required to unlock the full letter.")
    result = letters.build(req)
    _USED_TOKENS.add(req.token)  # spend the token: one payment, one letter
    return {
        "subject": result["subject"],
        "body": result["body"],
        "mode": result["mode"],
        "deadline": result["deadline"],
    }


# ---- Static app (served last so /api/* wins) -------------------------------

@app.get("/healthz")
def healthz():
    """Liveness/readiness probe for load balancers and orchestrators (the
    healthy/unhealthy check that lets an LB route around a dead instance).
    Deliberately outside /api/ so the rate limiter can't 429 the probe and
    make the balancer wrongly mark a healthy instance as down. Cheap and
    unauthenticated by design."""
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
