from collections import defaultdict


DOCUMENT_PATTERNS = {
    "invoice": [
        "invoice number",
        "invoice #",
        "bill to",
        "due date",
        "subtotal",
        "tax",
        "total amount",
        "vendor",
        "po number",
    ],
    "contract": [
        "agreement",
        "effective date",
        "party",
        "termination",
        "renewal",
        "clause",
        "governing law",
        "confidentiality",
    ],
    "claim_form": [
        "claim id",
        "incident date",
        "claimant",
        "adjuster",
        "loss description",
        "amount claimed",
        "policy number",
    ],
    "bank_statement": [
        "account number",
        "statement period",
        "closing balance",
        "opening balance",
        "debit",
        "credit",
    ],
    "resume": [
        "experience",
        "education",
        "skills",
        "projects",
        "linkedin",
    ],
    "compliance_report": [
        "audit",
        "control",
        "observation",
        "remediation",
        "risk rating",
        "finding",
    ],
}


def classify_document(text: str) -> tuple[str, float, dict[str, int]]:
    lowered = text.lower()
    scores = defaultdict(int)

    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                scores[doc_type] += 1

    if not scores:
        return "unknown", 0.35, {}

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    total = sum(scores.values())
    confidence = round(min(0.99, 0.55 + (best_score / max(total, 1)) * 0.4), 4)
    return best_type, confidence, dict(scores)
