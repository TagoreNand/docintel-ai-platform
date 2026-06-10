"""Synthetic-but-realistic labelled corpus for the document classifier.

Real labelled enterprise documents cannot ship in a public repo, so we generate
a templated corpus with randomised entities per class. The vocabulary mirrors the
fields the rule engine and extractor look for, which gives the TF-IDF +
calibrated logistic-regression model genuine, generalisable signal while keeping
the project fully self-contained and reproducible (fixed RNG seed).
"""

from __future__ import annotations

import random

DOC_TYPES = [
    "invoice",
    "contract",
    "claim_form",
    "bank_statement",
    "resume",
    "compliance_report",
]

_VENDORS = ["Nova Industrial Supplies", "Aurora Systems", "Beacon Logistics",
            "Quantum Components", "Summit Manufacturing", "Vertex Materials"]
_PEOPLE = ["John Smith", "Maria Garcia", "Wei Chen", "Aisha Khan", "David Brown",
           "Priya Nair", "Liam O'Brien", "Sofia Rossi"]
_STATES = ["Delaware", "California", "New York", "Texas", "Illinois", "Washington"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _date(rng: random.Random) -> str:
    return f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}, 20{rng.randint(23, 26)}"


def _money(rng: random.Random, lo: int = 100, hi: int = 90000) -> str:
    return f"{rng.randint(lo, hi):,}.{rng.randint(0, 99):02d}"


def _invoice(rng: random.Random) -> str:
    sub = rng.randint(500, 40000)
    tax = round(sub * 0.1)
    return (
        f"INVOICE\nInvoice Number: INV-{rng.randint(1000, 9999)}\n"
        f"Invoice Date: {_date(rng)}\nDue Date: {_date(rng)}\n"
        f"Bill To: {rng.choice(_PEOPLE)}\nVendor: {rng.choice(_VENDORS)}\n"
        f"PO Number: PO-{rng.randint(100, 999)}\n"
        f"Subtotal: ${sub:,}.00\nTax: ${tax:,}.00\nTotal Amount: ${sub + tax:,}.00\n"
        "Please remit payment to the vendor account by the due date."
    )


def _contract(rng: random.Random) -> str:
    return (
        f"SERVICE AGREEMENT\nThis Agreement is entered into and made effective "
        f"as of {_date(rng)} by and between {rng.choice(_VENDORS)} and "
        f"{rng.choice(_VENDORS)}.\nEffective Date: {_date(rng)}\n"
        f"Termination: Either party may terminate this agreement with 30 days notice.\n"
        f"Renewal: This contract renews annually unless cancelled.\n"
        f"Governing Law: {rng.choice(_STATES)}\n"
        "Confidentiality and indemnification clauses apply to both parties. "
        "The parties agree to the terms and conditions set forth in this clause."
    )


def _claim(rng: random.Random) -> str:
    return (
        f"INSURANCE CLAIM FORM\nClaim ID: CLM-{rng.randint(1000, 9999)}\n"
        f"Claimant: {rng.choice(_PEOPLE)}\nPolicy Number: POL-{rng.randint(10000, 99999)}\n"
        f"Incident Date: {_date(rng)}\nAdjuster: {rng.choice(_PEOPLE)}\n"
        f"Loss Description: Property damage following a severe weather event.\n"
        f"Amount Claimed: ${_money(rng)}\n"
        "The claimant requests reimbursement under the policy coverage terms."
    )


def _bank_statement(rng: random.Random) -> str:
    return (
        f"BANK STATEMENT\nAccount Number: ****{rng.randint(1000, 9999)}\n"
        f"Statement Period: {_date(rng)} to {_date(rng)}\n"
        f"Opening Balance: ${_money(rng, 1000, 50000)}\n"
        f"Closing Balance: ${_money(rng, 1000, 50000)}\n"
        f"Debit: ${_money(rng, 10, 2000)} card purchase\n"
        f"Credit: ${_money(rng, 10, 5000)} direct deposit\n"
        "Available balance reflects all posted transactions for this period."
    )


def _resume(rng: random.Random) -> str:
    return (
        f"{rng.choice(_PEOPLE)}\nemail: candidate@example.com | LinkedIn: in/profile\n"
        "Professional Summary: Experienced engineer with a strong background.\n"
        "Experience: Senior Engineer, led multiple projects and teams.\n"
        "Education: B.S. in Computer Science.\n"
        "Skills: Python, SQL, machine learning, cloud infrastructure.\n"
        "Projects: Built data pipelines and ML models.\nCertifications: AWS Certified."
    )


def _compliance(rng: random.Random) -> str:
    return (
        f"COMPLIANCE AUDIT REPORT\nScope: Review of internal controls.\n"
        f"Risk Rating: {rng.choice(['Low', 'Medium', 'High'])}\n"
        f"Finding: Control gap identified in access management.\n"
        f"Observation: Several accounts lacked periodic review.\n"
        f"Recommendation: Implement quarterly access recertification.\n"
        "Remediation: Owner assigned with a target completion date. "
        "This audit assesses the effectiveness of the control environment."
    )


_GENERATORS = {
    "invoice": _invoice,
    "contract": _contract,
    "claim_form": _claim,
    "bank_statement": _bank_statement,
    "resume": _resume,
    "compliance_report": _compliance,
}


def build_dataset(samples_per_class: int = 160, seed: int = 13) -> tuple[list[str], list[str]]:
    """Return ``(texts, labels)`` for a balanced synthetic training corpus."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[str] = []
    for label, generator in _GENERATORS.items():
        for _ in range(samples_per_class):
            texts.append(generator(rng))
            labels.append(label)
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order]
