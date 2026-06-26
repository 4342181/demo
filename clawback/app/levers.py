"""
The "smart" core of Clawback: the mapping from (scenario, region) to the
*correct* legal/contractual lever to pull, plus the escalation path.

This is what separates Clawback from a generic "write me a complaint
letter" prompt. A good demand letter wins because it names the specific
right the company is obligated to honour and the specific body you'll
escalate to if they don't. Those differ by both the type of problem and
the jurisdiction, so we keep them as structured data rather than baking
them into prose.

None of this is legal advice; it's a well-sourced starting point a
reasonable consumer could send. Letters say so.
"""

# ---- Scenarios -------------------------------------------------------------
# Each scenario describes a class of problem and the fields we collect for it.
# `claim` is the default remedy we demand; `levers` holds region-specific
# legal hooks looked up at letter-build time.

SCENARIOS = {
    "flight": {
        "label": "Delayed / cancelled flight",
        "blurb": "Airline delayed, cancelled, or bumped you, or lost your bags.",
        "claim": "the compensation and refund I am owed",
        "fields": [
            {"name": "company", "label": "Airline", "placeholder": "e.g. SkyHigh Air"},
            {"name": "reference", "label": "Booking / PNR reference", "placeholder": "e.g. X7K2QP"},
            {"name": "route", "label": "Route", "placeholder": "e.g. Cape Town → Johannesburg"},
            {"name": "date", "label": "Date of flight", "placeholder": "e.g. 12 June 2026"},
            {"name": "amount", "label": "Amount you're claiming (optional)", "placeholder": "e.g. R4 000 / €250"},
        ],
    },
    "bank_fee": {
        "label": "Bank fee / unauthorised charge",
        "blurb": "A fee, double charge, or transaction you didn't authorise.",
        "claim": "a full refund of the disputed amount",
        "fields": [
            {"name": "company", "label": "Bank / card provider", "placeholder": "e.g. First National"},
            {"name": "reference", "label": "Account / card last 4 / ref", "placeholder": "e.g. ****4821"},
            {"name": "date", "label": "Date of the charge", "placeholder": "e.g. 3 June 2026"},
            {"name": "amount", "label": "Disputed amount", "placeholder": "e.g. R899 / $120"},
        ],
    },
    "faulty_product": {
        "label": "Faulty / not-as-described product",
        "blurb": "Something broke, never worked, or wasn't what was advertised.",
        "claim": "a repair, replacement, or refund",
        "fields": [
            {"name": "company", "label": "Retailer / seller", "placeholder": "e.g. GadgetWorld"},
            {"name": "reference", "label": "Order number", "placeholder": "e.g. ORD-99214"},
            {"name": "product", "label": "Product", "placeholder": "e.g. Noise-cancelling headphones"},
            {"name": "date", "label": "Purchase date", "placeholder": "e.g. 20 May 2026"},
            {"name": "amount", "label": "Amount paid", "placeholder": "e.g. R2 499 / $149"},
        ],
    },
    "subscription": {
        "label": "Subscription you couldn't cancel",
        "blurb": "Charged after cancelling, or trapped by a cancel-proof flow.",
        "claim": "a refund of the charges taken after I tried to cancel, and written confirmation of cancellation",
        "fields": [
            {"name": "company", "label": "Company", "placeholder": "e.g. StreamPlus"},
            {"name": "reference", "label": "Account email / ID", "placeholder": "e.g. you@email.com"},
            {"name": "date", "label": "Date you tried to cancel", "placeholder": "e.g. 1 April 2026"},
            {"name": "amount", "label": "Amount wrongly charged", "placeholder": "e.g. R199 ×3"},
        ],
    },
    "accommodation": {
        "label": "Hotel / rental not as described",
        "blurb": "The place wasn't what was advertised, or was cancelled on you.",
        "claim": "a partial or full refund",
        "fields": [
            {"name": "company", "label": "Hotel / platform / host", "placeholder": "e.g. Seaview Lodge"},
            {"name": "reference", "label": "Booking reference", "placeholder": "e.g. BK-55109"},
            {"name": "date", "label": "Stay date", "placeholder": "e.g. 5–8 June 2026"},
            {"name": "amount", "label": "Amount paid", "placeholder": "e.g. R3 600 / $210"},
        ],
    },
    "delivery": {
        "label": "Late / undelivered order",
        "blurb": "You paid, the thing never arrived (or arrived far too late).",
        "claim": "immediate delivery or a full refund",
        "fields": [
            {"name": "company", "label": "Seller / courier", "placeholder": "e.g. QuickShip"},
            {"name": "reference", "label": "Order / tracking number", "placeholder": "e.g. TRK-7781234"},
            {"name": "date", "label": "Order date", "placeholder": "e.g. 15 May 2026"},
            {"name": "amount", "label": "Amount paid", "placeholder": "e.g. R750 / $45"},
        ],
    },
    "other": {
        "label": "Something else",
        "blurb": "Any other refund or service complaint.",
        "claim": "an appropriate remedy",
        "fields": [
            {"name": "company", "label": "Company", "placeholder": "Who you're complaining to"},
            {"name": "reference", "label": "Reference (optional)", "placeholder": "Any account/order ref"},
            {"name": "date", "label": "Relevant date", "placeholder": "e.g. 1 June 2026"},
            {"name": "amount", "label": "Amount at stake (optional)", "placeholder": "e.g. R1 000"},
        ],
    },
}

# ---- Regions ---------------------------------------------------------------
# Per region: the consumer-protection framing, the per-scenario legal lever
# (a sentence the letter drops in), and the body you escalate to.

REGIONS = {
    "za": {
        "label": "South Africa",
        "currency_hint": "R",
        "default_lever": (
            "Under the Consumer Protection Act 68 of 2008, I am entitled to "
            "goods and services of a quality that a reasonable consumer is "
            "entitled to expect, and to a remedy where that standard is not met."
        ),
        "levers": {
            "faulty_product": (
                "Section 56 of the Consumer Protection Act 68 of 2008 gives me "
                "the right, within six months of delivery, to return goods that "
                "are defective or not fit for their intended purpose and to "
                "choose a repair, replacement, or full refund — at no cost to me."
            ),
            "subscription": (
                "Section 14 of the Consumer Protection Act limits automatically "
                "renewing agreements and entitles me to cancel, and the Act "
                "prohibits charging for services I have cancelled."
            ),
            "delivery": (
                "Section 19 of the Consumer Protection Act entitles me to "
                "delivery at the agreed time, failing which I may cancel and "
                "recover what I paid."
            ),
            "accommodation": (
                "Section 54 of the Consumer Protection Act entitles me to "
                "services performed to the standard reasonably expected, and to "
                "a refund or price reduction where they are not."
            ),
        },
        "escalation": (
            "the National Consumer Commission and the relevant industry "
            "ombudsman (and, for card transactions, a formal chargeback through "
            "my bank)"
        ),
    },
    "uk": {
        "label": "United Kingdom",
        "currency_hint": "£",
        "default_lever": (
            "Under the Consumer Rights Act 2015, goods must be of satisfactory "
            "quality, fit for purpose, and as described, and services performed "
            "with reasonable care and skill."
        ),
        "levers": {
            "flight": (
                "Under UK261 (the retained EU261 regulation), I am entitled to "
                "fixed compensation and a refund or re-routing for a cancelled "
                "or significantly delayed flight within the airline's control."
            ),
            "faulty_product": (
                "Under the Consumer Rights Act 2015 I am entitled to a full "
                "refund for faulty goods rejected within 30 days, or a repair, "
                "replacement, or refund thereafter."
            ),
            "bank_fee": (
                "Under Section 75 of the Consumer Credit Act and my chargeback "
                "rights, I am entitled to dispute and recover this amount."
            ),
            "subscription": (
                "Under the Consumer Rights Act and the Consumer Contracts "
                "Regulations, I am entitled to cancel and to a refund of charges "
                "taken after cancellation."
            ),
        },
        "escalation": (
            "the Financial Ombudsman Service or the relevant ADR scheme, a "
            "Section 75 / chargeback claim through my card provider, and a claim "
            "in the small claims court"
        ),
    },
    "eu": {
        "label": "European Union",
        "currency_hint": "€",
        "default_lever": (
            "Under the EU Consumer Sales Directive (2019/771), goods must "
            "conform to the contract, and I am entitled to a remedy where they "
            "do not."
        ),
        "levers": {
            "flight": (
                "Under Regulation (EC) No 261/2004 (EU261), I am entitled to "
                "fixed compensation of €250–€600 and a refund or re-routing for "
                "a cancelled or significantly delayed flight within the "
                "airline's control."
            ),
            "faulty_product": (
                "Under the EU Consumer Sales Directive (2019/771), I am entitled "
                "to a repair, replacement, or refund for goods that do not "
                "conform to the contract."
            ),
            "subscription": (
                "Under the EU Consumer Rights Directive, I am entitled to cancel "
                "and to a refund of charges taken after cancellation."
            ),
        },
        "escalation": (
            "the relevant national consumer authority and the EU Online Dispute "
            "Resolution platform, plus a chargeback through my card provider"
        ),
    },
    "us": {
        "label": "United States",
        "currency_hint": "$",
        "default_lever": (
            "Under the implied warranty of merchantability and applicable state "
            "consumer-protection law, I am entitled to a product or service that "
            "works as represented, and to a remedy where it does not."
        ),
        "levers": {
            "flight": (
                "Under US Department of Transportation rules, I am entitled to a "
                "prompt cash refund (not a voucher) for a cancelled or "
                "significantly changed flight that I chose not to take."
            ),
            "bank_fee": (
                "Under the Fair Credit Billing Act and Regulation E, I am "
                "entitled to dispute this charge and to a chargeback through my "
                "card issuer."
            ),
            "subscription": (
                "Under the FTC's rule on negative-option marketing and "
                "applicable state auto-renewal laws, I am entitled to cancel "
                "easily and to a refund of charges taken after cancellation."
            ),
        },
        "escalation": (
            "the Consumer Financial Protection Bureau or FTC, my state attorney "
            "general's consumer-protection division, a chargeback through my "
            "card issuer, and small claims court"
        ),
    },
    "other": {
        "label": "Somewhere else",
        "currency_hint": "",
        "default_lever": (
            "Under the consumer-protection law applicable to this transaction, I "
            "am entitled to goods and services that match what was promised, and "
            "to a remedy where they do not."
        ),
        "levers": {},
        "escalation": (
            "the relevant consumer-protection regulator or ombudsman, and a "
            "chargeback through my bank or card provider"
        ),
    },
}


def lever_for(region_key: str, scenario_key: str) -> str:
    region = REGIONS.get(region_key, REGIONS["other"])
    return region["levers"].get(scenario_key, region["default_lever"])


def escalation_for(region_key: str) -> str:
    return REGIONS.get(region_key, REGIONS["other"])["escalation"]
