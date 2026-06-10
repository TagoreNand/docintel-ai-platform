import re
from typing import Iterable


def _match(patterns: Iterable[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def extract_entities(doc_type: str, text: str) -> list[dict]:
    rules = {
        "invoice": {
            "invoice_number": [r"invoice(?:\s+number|\s*#)?[:\s]+([A-Z0-9\-]+)"],
            "invoice_date": [r"invoice date[:\s]+([A-Za-z0-9,\-\/ ]+)"],
            "due_date": [r"due date[:\s]+([A-Za-z0-9,\-\/ ]+)"],
            "vendor_name": [r"vendor[:\s]+([A-Za-z0-9&,\.\- ]+)", r"from[:\s]+([A-Za-z0-9&,\.\- ]+)"],
            "po_number": [r"po number[:\s]+([A-Z0-9\-]+)"],
            "subtotal": [r"subtotal[:\s\$]+([0-9,]+(?:\.[0-9]{2})?)"],
            "tax_amount": [r"tax[:\s\$]+([0-9,]+(?:\.[0-9]{2})?)"],
            "total_amount": [r"total amount[:\s\$]+([0-9,]+(?:\.[0-9]{2})?)", r"total[:\s\$]+([0-9,]+(?:\.[0-9]{2})?)"],
        },
        "contract": {
            "counterparty_a": [r"between\s+([A-Za-z0-9 ,.&\-]+)\s+and\s+[A-Za-z0-9 ,.&\-]+"],
            "counterparty_b": [r"between\s+[A-Za-z0-9 ,.&\-]+\s+and\s+([A-Za-z0-9 ,.&\-]+)"],
            "effective_date": [r"effective date[:\s]+([A-Za-z0-9,\-\/ ]+)"],
            "termination_date": [r"termination date[:\s]+([A-Za-z0-9,\-\/ ]+)"],
            "renewal_term": [r"renewal(?: term)?[:\s]+([A-Za-z0-9,\-\/ ]+)"],
            "governing_law": [r"governing law[:\s]+([A-Za-z0-9 ,.&\-]+)"],
            "penalty_clause": [r"penalty clause[:\s]+(.+)", r"late payment penalty[:\s]+(.+)"],
        },
        "claim_form": {
            "claim_id": [r"claim id[:\s]+([A-Z0-9\-]+)"],
            "claimant_name": [r"claimant[:\s]+([A-Za-z ,.\-]+)"],
            "policy_number": [r"policy number[:\s]+([A-Z0-9\-]+)"],
            "incident_date": [r"incident date[:\s]+([A-Za-z0-9,\-\/ ]+)"],
            "amount_claimed": [r"amount claimed[:\s\$]+([0-9,]+(?:\.[0-9]{2})?)"],
            "adjuster": [r"adjuster[:\s]+([A-Za-z ,.\-]+)"],
        },
    }

    selected = rules.get(doc_type, {})
    results: list[dict] = []

    for field_name, patterns in selected.items():
        value = _match(patterns, text)
        if value:
            confidence = 0.92 if field_name.endswith(("number", "date", "id")) else 0.8
            results.append(
                {
                    "field_name": field_name,
                    "field_value": value,
                    "confidence": confidence,
                }
            )

    return results
