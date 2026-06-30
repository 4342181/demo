# Clawback — get your money back

Turn a 60-second form into a professional **demand letter** that cites the
*right* consumer-protection law for the user's country, with a hard deadline
and an escalation threat. Free preview → pay a small one-off fee to unlock the
full, ready-to-send letter.

## Why this can make money fast
- **The value is money.** "Pay R49, get R2 000 back" is the easiest sale there is.
- **Built-in virality.** People share refund wins.
- **SEO long-tail.** "[airline] refund letter", "[bank] dispute letter".
- **Paywall at peak desire** — the visitor sees the letter is good *before* paying.

## Run it (zero keys needed)
```bash
./run.sh          # http://localhost:8000
```
Out of the box it uses a deterministic letter engine (no API key, no network)
and a **demo unlock** (no card). Everything below is optional polish.

### Go live
- **Payments:** set `STRIPE_SECRET_KEY` (+ `CLAWBACK_BASE_URL`) and `/api/checkout`
  switches from demo token to real Stripe Checkout. Price via `CLAWBACK_PRICE`,
  `CLAWBACK_PRICE_MINOR`, `CLAWBACK_CURRENCY`.
- **Better letters:** set `ANTHROPIC_API_KEY` and letters get LLM-polished
  (Claude rewrites the draft while preserving every fact and the legal lever).
  Falls back to the deterministic draft on any error.

## How it's built
| File | Role |
|------|------|
| `app/levers.py`  | The smart core — (scenario × region) → the correct legal lever + escalation path |
| `app/letters.py` | Letter assembly: deterministic spine, optional LLM polish, free-preview truncation |
| `app/main.py`    | FastAPI: `/api/config`, `/api/preview` (free), `/api/checkout`, `/api/full` (paid) |
| `app/static/index.html` | Mobile-first single-page app |

The letter content never depends on payment — the paywall only reveals the
full text, so the product is honest (most of the letter is visible free) while
still converting at the moment of peak desire.

## Security

What's in place, and what production still needs:

- **No secrets in the frontend.** Stripe/Anthropic keys live in server-side env
  vars only; the browser holds none and talks only to our backend.
- **Rate limiting.** Per-IP **sliding-window** limiter on `/api/*`
  (`CLAWBACK_RATE_LIMIT`/`CLAWBACK_RATE_WINDOW`, default 60/min → 429) — the
  sliding window avoids the fixed-window boundary burst (2×limit across a window
  edge). It's per-process — move it to Redis for a multi-worker deployment.
  Behind a reverse proxy/CDN, set `CLAWBACK_TRUST_PROXY=1` so it keys on the
  real client IP (`X-Forwarded-For`) instead of the proxy — off by default
  because that header is spoofable when not behind a trusted proxy.
- **Input validation.** Every request field is length-capped (and
  `deadline_days` range-checked), so oversized payloads can't exhaust memory
  or inflate LLM cost. No SQL injection surface (no DB); user text renders via
  `textContent`, not `innerHTML`.
- **Access control.** The app is intentionally anonymous (no accounts) and
  stores no per-user data, so there's no IDOR / row-level-security surface (no
  `/user/{id}`-style endpoint to swap an ID on). The meaningful gate is the
  paid-token check on `/api/full` — not OAuth/RBAC.
- **Single-use unlock tokens.** A paid token is spent on first use, so it can't
  be replayed (or shared/leaked) to mint unlimited free letters from one
  payment.
- **No leaky errors.** Raw exceptions/stack traces are never returned to
  clients — a catch-all handler logs the real cause server-side and returns a
  generic message, so errors don't hand attackers a map of the internals.
- **Deploy behind HTTPS.** Terminate TLS at your host/reverse proxy; set
  `CLAWBACK_BASE_URL` to the https origin so Stripe redirects stay secure.

**Observability & performance.** Every request is access-logged (method, path,
status, latency) — **metadata only; bodies are never logged**, since letters
contain personal details. Responses are gzip-compressed, and the frontend is a
single self-contained file with no external requests, so it loads fast. Heavy
observability stacks (Prometheus/Datadog/Jaeger) are deliberately omitted as
overkill until there's real traffic.

**Scaling.** The app is stateless (letters are generated per-request), so it
runs fine behind a load balancer — with one caveat: the rate-limit counter and
unlock/demo tokens are in-process, so for *multiple* instances move them to a
shared store (Redis). A `GET /healthz` endpoint (outside `/api/`, so it's never
rate-limited) is provided for load-balancer/orchestrator health checks. There's
no file storage to offload (no object store needed) and no database to cache
beyond the already-memoized static config.

**Load testing.** `python scripts/loadtest.py -n 500 -c 50` stress-tests the
running API (stdlib only). A single instance serves ~680 req/s for the free
preview, and under a concurrent flood the limiter holds exactly at the cap
(e.g. 60×`200` then `429`s). Use it to sanity-check before real traffic.

These are locked in by an automated regression suite — run `python -m pytest`
from `clawback/` (covers rate limiting, input/enum validation, the paid-token
gate + single use, sanitized errors, and no secrets in the frontend bundle).

## Coverage
**Scenarios:** delayed/cancelled flight · bank fee / unauthorised charge ·
faulty product · trapped subscription · accommodation not as described ·
late/undelivered order · other.
**Regions:** South Africa (CPA) · UK (CRA / S75 / UK261) · EU (EU261 /
Consumer Sales Directive) · US (DOT / FCBA / FTC) · other.

> Clawback writes self-help letters from publicly available consumer-protection
> principles. It is not a law firm and does not provide legal advice.
