"""
Letter generation.

Two modes:
  * Deterministic (default, zero external dependencies): assembles a strong,
    correctly-structured demand letter from the inputs and the region/scenario
    lever. This is what runs out of the box, so the product works with no API
    key and no network.
  * LLM-polished (optional): if ANTHROPIC_API_KEY is set, we ask Claude to
    rewrite the deterministic draft into tighter, more natural prose while
    keeping every fact and the legal lever intact. The deterministic draft is
    always the fallback, so an API hiccup degrades to a still-sendable letter.

A good demand letter has a reliable spine, which is why the deterministic
path is genuinely good and not a stub:
    1. Header (sender, recipient, date, subject with reference)
    2. The facts, briefly and in order
    3. The specific right/obligation being invoked (the lever)
    4. An explicit demand with a number and a hard deadline
    5. The escalation path if ignored
    6. A professional close
"""
from __future__ import annotations

import os
import datetime
from dataclasses import dataclass, field

from . import levers


@dataclass
class LetterInputs:
    scenario: str
    region: str
    company: str = ""
    reference: str = ""
    date: str = ""
    amount: str = ""
    route: str = ""
    product: str = ""
    facts: str = ""           # free-text "what happened"
    sender_name: str = ""
    sender_contact: str = ""
    deadline_days: int = 14

    extras: dict = field(default_factory=dict)


def _today() -> str:
    return datetime.date.today().strftime("%-d %B %Y") if os.name != "nt" else datetime.date.today().strftime("%d %B %Y")


def _deadline_date(days: int) -> str:
    d = datetime.date.today() + datetime.timedelta(days=max(1, days))
    return d.strftime("%-d %B %Y") if os.name != "nt" else d.strftime("%d %B %Y")


def _scenario_meta(key: str) -> dict:
    return levers.SCENARIOS.get(key, levers.SCENARIOS["other"])


def build_subject(inp: LetterInputs) -> str:
    meta = _scenario_meta(inp.scenario)
    ref = f" (Ref: {inp.reference})" if inp.reference else ""
    return f"Formal demand: {meta['label'].lower()}{ref}"


def _facts_paragraph(inp: LetterInputs) -> str:
    """Build the 'here is what happened' paragraph from structured fields,
    falling back to the user's free text where given."""
    bits = []
    meta = _scenario_meta(inp.scenario)

    lead = {
        "flight": f"On {inp.date or '[date]'}, my flight ({inp.route or 'the booked route'}) was disrupted.",
        "bank_fee": f"On {inp.date or '[date]'}, a charge of {inp.amount or '[amount]'} was applied to my account ({inp.reference or 'account on file'}) that I dispute.",
        "faulty_product": f"On {inp.date or '[date]'} I purchased {inp.product or 'a product'} (order {inp.reference or 'on file'}), which has proven defective or not as described.",
        "subscription": f"On {inp.date or '[date]'} I cancelled my subscription, yet {inp.amount or 'further charges'} were taken from me afterwards.",
        "accommodation": f"For my booking on {inp.date or '[date]'} ({inp.reference or 'reference on file'}), the accommodation was materially not as described.",
        "delivery": f"I placed order {inp.reference or 'on file'} on {inp.date or '[date]'} and paid {inp.amount or 'in full'}, but it has not been delivered as promised.",
        "other": f"On {inp.date or '[date]'} an issue arose in connection with {inp.reference or 'my dealings with you'}.",
    }.get(inp.scenario, "")
    if lead:
        bits.append(lead)

    if inp.facts.strip():
        bits.append(inp.facts.strip())
    else:
        bits.append(
            "Despite this, the matter has not been put right, and I have been "
            "left out of pocket and inconvenienced."
        )
    return " ".join(bits)


def build_deterministic(inp: LetterInputs) -> dict:
    meta = _scenario_meta(inp.scenario)
    lever = levers.lever_for(inp.region, inp.scenario)
    escalation = levers.escalation_for(inp.region)
    claim = meta["claim"]
    deadline = _deadline_date(inp.deadline_days)

    company = inp.company or "[Company name]"
    sender = inp.sender_name or "[Your name]"
    contact = inp.sender_contact or "[Your email / phone]"

    amount_clause = f" totalling {inp.amount}" if inp.amount else ""

    body = f"""{_today()}

To: {company} — Customer Relations / Disputes

Dear {company},

Re: {build_subject(inp)}

I am writing to formally demand {claim}{amount_clause} in respect of the matter set out below. I would prefer to resolve this directly and quickly, but I am prepared to escalate if I must.

{_facts_paragraph(inp)}

{lever}

Accordingly, I require that you {claim}{amount_clause} no later than {deadline} ({inp.deadline_days} days from the date of this letter). Please confirm in writing how and when this will be done.

If I do not receive a satisfactory response by that date, I will refer this matter to {escalation}, and I reserve the right to recover my costs. This letter serves as formal notice for those purposes.

I look forward to your prompt response.

Yours faithfully,
{sender}
{contact}

— Sent via Clawback. This letter is a self-help template based on publicly available consumer-protection principles and is not legal advice."""

    return {
        "subject": build_subject(inp),
        "body": body,
        "lever_note": lever,
        "deadline": deadline,
        "mode": "deterministic",
    }


# ---- Optional LLM polish ---------------------------------------------------

_POLISH_SYSTEM = (
    "You are an expert consumer-rights advocate and a sharp business writer. "
    "You will be given a draft demand letter. Rewrite it so it is tighter, "
    "calm, confident, and persuasive, in formal British/SA business English. "
    "Hard rules: keep every fact, name, reference, amount and date exactly as "
    "given; keep the legal lever sentence and its citation intact; keep the "
    "explicit demand and the deadline; keep the escalation paragraph; do not "
    "invent facts or new legal claims; keep it under 320 words; return only "
    "the letter text, no preamble."
)


def build(inp: LetterInputs) -> dict:
    """Public entry point. Deterministic by default; LLM-polished if a key is
    configured and the call succeeds."""
    draft = build_deterministic(inp)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return draft

    try:
        import anthropic  # imported lazily so the app runs without the dep

        # Hard timeout so a slow/hung LLM call can't freeze the request worker
        # — we'd rather fall back to the (already strong) deterministic draft
        # than make the user wait. Tunable via CLAWBACK_LLM_TIMEOUT.
        timeout = float(os.environ.get("CLAWBACK_LLM_TIMEOUT", "20"))
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        msg = client.messages.create(
            model=os.environ.get("CLAWBACK_MODEL", "claude-opus-4-8"),
            max_tokens=1200,
            system=_POLISH_SYSTEM,
            messages=[{"role": "user", "content": draft["body"]}],
        )
        polished = "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ).strip()
        if polished:
            draft["body"] = polished
            draft["mode"] = "llm"
    except Exception:
        # Any failure (no network, bad key, rate limit) → keep the strong
        # deterministic draft. The user still gets a sendable letter.
        pass

    return draft


def make_preview(body: str, free_lines: int = 8) -> dict:
    """What an unpaid visitor sees: the opening of the letter, then a cut.
    The paywall lands here, right after they can see the letter is good."""
    lines = body.splitlines()
    shown = "\n".join(lines[:free_lines]).rstrip()
    hidden = max(0, len(lines) - free_lines)
    return {"preview": shown, "hidden_lines": hidden}
