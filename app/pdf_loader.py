from typing import List, Dict
import pdfplumber
from .utils import clean_text


def extract_pdf_text(pdf_path: str) -> List[Dict]:
    """Returns list of {page, text} dicts."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append({"page": i, "text": clean_text(txt)})
    return pages
