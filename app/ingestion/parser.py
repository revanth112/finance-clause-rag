
import re
from typing import List, Dict, Any
from app.schemas.document import ParsedPage


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_pages(raw_pages: List[Dict[str, Any]]) -> List[ParsedPage]:
    parsed_pages: List[ParsedPage] = []

    for page in raw_pages:
        parsed_pages.append(
            ParsedPage(
                page_number=page["page_number"],
                text=normalize_text(page["text"])
            )
        )

    return parsed_pages
