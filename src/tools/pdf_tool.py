import requests
from pypdf import PdfReader
from io import BytesIO
from langchain.tools import tool

@tool
def read_pdf_from_url(url: str) -> str:
    """Download and extract text from a PDF URL."""
    response = requests.get(url, timeout=30)
    reader = PdfReader(BytesIO(response.content))
    text = ""
    for page in reader.pages[:15]:  # limit to first 15 pages
        text += page.extract_text() or ""
    return text[:8000]  # token safety limit
    