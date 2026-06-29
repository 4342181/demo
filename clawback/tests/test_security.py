"""
Security regression suite — locks in the protections added across the build so
a future change can't silently undo them. Run: `python -m pytest` from the
clawback/ dir. Each test maps to an invariant in CLAUDE.md.
"""
import os

import pytest
from fastapi.testclient import TestClient

import app.main as m
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

BASE = {"scenario": "flight", "region": "za", "company": "AirX", "date": "1 June 2026"}


@pytest.fixture(autouse=True)
def reset_state():
    # Isolate each test: clear the in-process rate/token state and use
    # permissive defaults unless a test overrides them.
    m._rate_hits.clear()
    m._USED_TOKENS.clear()
    m._DEMO_TOKENS.clear()
    m.RATE_LIMIT = 1000
    m.STRIPE_KEY = None
    yield


def test_no_secrets_in_frontend():
    """Invariant 1: the browser bundle must never contain a real key."""
    html = open(os.path.join(os.path.dirname(m.__file__), "static", "index.html")).read().lower()
    for bad in ("sk_live_", "sk_test_", "anthropic_api_key", "secret_key", "bearer "):
        assert bad not in html, f"possible leaked secret marker in frontend: {bad}"


def test_rate_limiting_trips():
    """Invariant 2: public endpoints are rate-limited."""
    m.RATE_LIMIT = 5
    codes = [client.post("/api/preview", json=BASE).status_code for _ in range(10)]
    assert 429 in codes


def test_sliding_window_blocks_boundary_burst():
    """Invariant 2: a fixed window would allow 2×limit across a window edge;
    the sliding-window counter must not."""
    m.RATE_LIMIT = 100
    W = m.RATE_WINDOW
    m._rate_hits.clear()
    # Fill the limit at the very end of window N…
    t_end = 1000 * W + (W - 0.001)
    allowed_end = sum(m._rate_check("9.9.9.9", t_end) for _ in range(100))
    # …then try again at the very start of window N+1.
    t_start = 1001 * W + 0.001
    allowed_start = sum(m._rate_check("9.9.9.9", t_start) for _ in range(100))
    assert allowed_end == 100            # the first window's budget is usable
    assert allowed_start < 100           # but the boundary burst is throttled
    assert allowed_end + allowed_start <= 100 + 5  # ≈ one window's worth, not two


def test_length_cap_rejected():
    """Invariant 3: oversized input is rejected, not processed."""
    assert client.post("/api/preview", json={**BASE, "facts": "x" * 5000}).status_code == 422


def test_deadline_range_rejected():
    """Invariant 3: out-of-range numeric input is rejected."""
    assert client.post("/api/preview", json={**BASE, "deadline_days": 999}).status_code == 422


def test_enum_whitelist_rejected():
    """Invariant 3: unknown enum values are rejected, not silently coerced."""
    assert client.post("/api/preview", json={**BASE, "scenario": "garbage"}).status_code == 422
    assert client.post("/api/preview", json={**BASE, "region": "mars"}).status_code == 422


def test_full_requires_payment_token():
    """Invariant 5: the paid letter is gated."""
    assert client.post("/api/full", json=BASE).status_code == 402


def test_unlock_token_is_single_use():
    """Invariant 5: a token can't be replayed for a second free letter."""
    token = client.post("/api/checkout", json=BASE).json()["token"]
    assert client.post("/api/full", json={**BASE, "token": token}).status_code == 200
    assert client.post("/api/full", json={**BASE, "token": token}).status_code == 402


def test_errors_do_not_leak_internals():
    """Invariant 4: failures return a generic message, never raw internals."""
    m.STRIPE_KEY = "sk_test_fake"  # forces the Stripe path to fail server-side
    r = client.post("/api/checkout", json=BASE)
    assert r.status_code == 502
    detail = r.json()["detail"].lower()
    for leak in ("module", "import", "traceback", "stripe", "exception"):
        assert leak not in detail


def test_free_preview_never_reaches_the_expensive_ai():
    """Invariant 5 (economic): the costly LLM path (letters.build) must be
    unreachable without payment. If preview ever called build() instead of
    build_deterministic(), the patched build below would blow up — so this
    guards against anyone regressing the free path into a free AI endpoint."""
    import app.letters as L

    def _boom(*a, **k):
        raise AssertionError("LLM path reached on the free preview endpoint")

    original = L.build
    L.build = _boom
    try:
        assert client.post("/api/preview", json=BASE).status_code == 200
    finally:
        L.build = original


def test_healthz_is_ok_and_not_rate_limited():
    """Load-balancer probe must stay 200 even under a tight limit — it lives
    outside /api/ so the rate limiter never touches it."""
    m.RATE_LIMIT = 1
    m._rate_hits.clear()
    codes = {client.get("/healthz").status_code for _ in range(5)}
    assert codes == {200}


def test_llm_circuit_breaker_state_machine():
    """Resilience: repeated LLM failures open the breaker (so we stop waiting
    out the timeout on every request); a later success closes it again."""
    import app.letters as L
    L._llm_failures = 0
    L._llm_open_until = 0.0
    assert L._breaker_allows() is True
    for _ in range(L._LLM_FAIL_THRESHOLD):
        L._breaker_record(False)
    assert L._breaker_allows() is False          # opened after N failures
    L._llm_open_until = 0.0                       # simulate cooldown elapsed → half-open
    assert L._breaker_allows() is True
    L._breaker_record(True)                       # success closes it
    assert L._llm_failures == 0 and L._breaker_allows() is True


def test_analytics_events_validated_and_carry_no_pii():
    """Analytics accepts only whitelisted funnel events, and the schema has no
    free-text field — so personal data physically can't be captured."""
    assert client.post("/api/event", json={"name": "preview_generated", "scenario": "flight"}).status_code == 200
    assert client.post("/api/event", json={"name": "exfiltrate"}).status_code == 422
    assert set(m.EventIn.model_fields) <= {"name", "scenario", "region"}


def test_admin_funnel_enforced_on_backend():
    """RBAC done right: the admin view is gated on the backend by a secret,
    not by hiding a link — and is invisible (404) when not configured."""
    m.ADMIN_TOKEN = None
    assert client.get("/api/admin/funnel").status_code == 404         # disabled, hidden
    m.ADMIN_TOKEN = "s3cret"
    assert client.get("/api/admin/funnel").status_code == 401          # no token
    assert client.get("/api/admin/funnel", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/api/admin/funnel", headers={"X-Admin-Token": "s3cret"}).status_code == 200
    m.ADMIN_TOKEN = None


def test_preview_is_free_and_open():
    """The free hook works without any token (and is deterministic/fast)."""
    r = client.post("/api/preview", json=BASE)
    assert r.status_code == 200
    assert r.json()["hidden_lines"] > 0  # the paywall actually hides content
