# Clawback — agent context & security rules

Context for any AI/agent (or human) changing this code. Most security bugs in
AI-built apps come from the agent not knowing the *expected behaviour*, not
from bad code. This file is that expected behaviour. **Read it before editing,
and keep it true when you change things.**

## What this app is
A stateless web app that generates consumer **demand letters**. No user
accounts, no database, no per-user stored data. A free preview is public; the
full letter is unlocked by a one-off payment.

## Access-control matrix
| Endpoint | Who may call | Auth/gate | Returns | Notes |
|----------|--------------|-----------|---------|-------|
| `GET /api/config` | anyone | none | static form config | no secrets, no user data |
| `POST /api/preview` | anyone | rate limit only | first ~8 lines of letter | **deterministic only — never call the LLM here** (free path stays fast/cheap) |
| `POST /api/checkout` | anyone | rate limit only | Stripe URL or demo token | server-side Stripe key only |
| `POST /api/full` | paying user | **valid + unused** token | complete letter | LLM polish allowed here; **token is single-use** |
| `GET /` + static | anyone | none | the SPA | no secrets in frontend |

There is intentionally **no `/user/{id}`-style endpoint** and no per-user data,
so there is no IDOR / row-level-security surface. If you ever add stored,
user-owned resources, you MUST add ownership checks (never trust an ID in the
URL/body to imply authorization) and revisit this matrix.

## Security invariants — do not regress these
1. **Secrets stay server-side.** `STRIPE_SECRET_KEY`, `ANTHROPIC_API_KEY` come
   from env and are never sent to or referenced by the browser. The frontend
   only ever calls our own `/api/*`.
2. **Every `/api/*` endpoint is rate-limited.** Per-IP (not per-user — callers
   are anonymous). New endpoints inherit the middleware; don't bypass it.
3. **Validate all input.** Every request field is length-capped and numeric
   fields range-checked via pydantic `Field(...)`. Add caps for any new field.
4. **Never leak internals in responses.** No raw exceptions, stack traces, or
   dependency names to clients. Log server-side; return a generic message. The
   catch-all handler enforces this — keep it.
5. **Paid actions verify *and consume* a token.** `/api/full` checks the token
   is paid and not already used, then marks it used. One payment → one letter.
6. **Render user text safely.** User-supplied content is shown via `textContent`
   (or downloaded/printed as plain text), never injected as HTML. Only our own
   trusted strings use `innerHTML`. Keep it that way to stay XSS-free.
7. **Deploy behind HTTPS.** Set `CLAWBACK_BASE_URL` to the https origin and
   terminate TLS at the proxy, so Stripe redirects stay secure.

## Known production gaps (intentional, for a demo)
- Rate-limit counter, demo tokens, and used-token set are **in-process**. For
  multi-worker production move them to a shared store (e.g. Redis), or the
  limits/single-use guarantees only hold per worker.
- Demo unlock (no Stripe key) hands out a token without payment — that's for
  local testing. Real deployments must set `STRIPE_SECRET_KEY`.

## When you add a feature
- New `/api` endpoint? → confirm it's covered by rate limiting + input caps,
  returns no raw errors, and add a row to the matrix above.
- Touching payments? → keep verify-then-consume; never make a token reusable.
- Touching the LLM path? → keep the timeout + deterministic fallback; never put
  the LLM on the free `/api/preview` path.
