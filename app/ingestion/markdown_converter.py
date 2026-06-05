import re
from typing import List
from app.schemas.document import ParsedPage


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    if line.isupper() and len(line) > 4:
        return True

    if re.match(r"^\d+(\.\d+)*\s+.+", line):
        return True

    known_headers = [
        "payment terms",
        "termination",
        "confidentiality",
        "indemnity",
        "liability",
        "governing law",
        "force majeure",
    ]
    return line.lower() in known_headers


def pages_to_markdown(parsed_pages: List[ParsedPage]) -> str:
    lines: List[str] = []

    for page in parsed_pages:
        lines.append(f"## Page {page.page_number}")
        lines.append("")

        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            if is_heading(line):
                lines.append(f"### {line}")
            else:
                lines.append(line)

        lines.append("")

    return "\n".join(lines).strip()
