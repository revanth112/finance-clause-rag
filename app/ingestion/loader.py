"""Document loader for finance clause PDFs and text files."""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class DocumentLoader:
    """Load documents from file system with metadata extraction."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single document file and return content with metadata."""
        path = Path(file_path)
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        content = self._read_file(path)
        metadata = self._extract_metadata(path, content)

        return {
            "content": content,
            "metadata": metadata,
            "file_path": str(path),
        }

    def load_directory(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load all supported documents from a directory."""
        target_dir = Path(directory) if directory else self.data_dir
        documents = []

        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in target_dir.rglob(f"*{ext}"):
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")

        return documents

    def _read_file(self, path: Path) -> str:
        """Read file content based on extension."""
        if path.suffix == ".pdf":
            return self._read_pdf(path)
        elif path.suffix == ".docx":
            return self._read_docx(path)
        else:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text_parts = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(f"<!-- Page {page_num + 1} -->\n{text}")
            doc.close()
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError("PyMuPDF (fitz) required for PDF loading. Install with: pip install pymupdf")

    def _read_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        try:
            import docx
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ImportError("python-docx required. Install with: pip install python-docx")

    def _extract_metadata(self, path: Path, content: str) -> Dict[str, Any]:
        """Extract document metadata from path and content."""
        metadata = {
            "filename": path.name,
            "file_type": path.suffix.lstrip("."),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
            "source": str(path),
        }

        # Extract document type from filename patterns
        filename_lower = path.stem.lower()
        if any(kw in filename_lower for kw in ["contract", "agreement", "clause"]):
            metadata["doc_type"] = "contract"
        elif any(kw in filename_lower for kw in ["policy", "compliance", "regulation"]):
            metadata["doc_type"] = "policy"
        elif any(kw in filename_lower for kw in ["report", "annual", "quarterly"]):
            metadata["doc_type"] = "report"
        else:
            metadata["doc_type"] = "general"

        # Extract year from filename or content
        year_match = re.search(r"(20\d{2})", path.name)
        if year_match:
            metadata["year"] = int(year_match.group(1))

        # Extract company name hints
        company_keywords = ["goldman", "jpmorgan", "morgan", "blackrock", "vanguard", "fidelity"]
        for kw in company_keywords:
            if kw in filename_lower or kw in content[:500].lower():
                metadata["company"] = kw.title()
                break

        return metadata
