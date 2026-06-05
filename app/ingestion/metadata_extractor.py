import re
from typing import Optional


CLAUSE_TYPES = [
    "payment terms",
    "termination",
    "confidentiality",
    "indemnity",
    "liability",
    "governing law",
    "force majeure",
    "warranty",
    "limitation of liability",
]

JURISDICTIONS = [
    "india",
    "new york",
    "delaware",
    "singapore",
    "england",
    "california",
]


def infer_document_type(file_name: str, text: str) -> str:
    combined = f"{file_name} {text}".lower()

    if "master service agreement" in combined or "msa" in combined:
        return "master_service_agreement"
    if "statement of work" in combined or "sow" in combined:
        return "statement_of_work"
    if "invoice" in combined:
        return "invoice"
    if "contract" in combined or "agreement" in combined:
        return "contract"
    return "unknown"


def extract_clause_type(section_title: Optional[str], text: str = "") -> Optional[str]:
    combined = f"{section_title or ''} {text}".lower()
    for clause in CLAUSE_TYPES:
        if clause in combined:
            return clause
    return None


def extract_jurisdiction(text: str) -> Optional[str]:
    lower = text.lower()
    for item in JURISDICTIONS:
        if item in lower:
            return item.title()
    return None


def extract_vendor_name(text: str) -> Optional[str]:
    match = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\.|\n)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
