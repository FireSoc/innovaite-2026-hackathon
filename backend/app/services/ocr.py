"""OCR service — extract text from images and PDFs before LLM.

Tesseract is the source of truth for readable text; the LLM only categorizes
and structures. PDFs are rendered to images per page then OCR'd.

If pytesseract or pymupdf are not installed, the app still starts; extract_text
returns "" and the evidence pipeline falls back to LLM-only.
"""

import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# Optional dependencies: app starts even if not installed
try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError as e:
    logger.warning("OCR unavailable: %s. Install pytesseract and Pillow for OCR.", e)
    _OCR_AVAILABLE = False

try:
    import pymupdf
    _PYMUPDF_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning("PDF OCR unavailable: %s. Install pymupdf for PDF support.", e)
    _PYMUPDF_AVAILABLE = False

# MIME types we can OCR directly as images
IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
PDF_MIME_TYPE = "application/pdf"


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run Tesseract on image bytes. Returns raw text."""
    if not _OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(BytesIO(image_bytes))
        # Ensure RGB for consistent OCR
        if img.mode != "RGB":
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng")
        return (text or "").strip()
    except Exception as e:
        logger.warning(f"OCR on image failed: {e}")
        return ""


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Render PDF pages to images and OCR each. Returns concatenated text."""
    if not _PYMUPDF_AVAILABLE or not _OCR_AVAILABLE:
        return ""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=150, alpha=False)
            img_bytes = pix.tobytes("png")
            page_text = _ocr_image_bytes(img_bytes)
            if page_text:
                parts.append(f"[Page {i + 1}]\n{page_text}")
        doc.close()
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning(f"OCR on PDF failed: {e}")
        return ""


def extract_text(filename: str, file_bytes: bytes, mime_type: str) -> str:
    """
    Extract raw text from a single file using OCR.

    Args:
        filename: Original filename (for logging).
        file_bytes: Raw file content.
        mime_type: MIME type (image/* or application/pdf).

    Returns:
        Extracted text, or empty string if OCR fails, dependencies missing, or file type unsupported.
    """
    if not _OCR_AVAILABLE and mime_type in IMAGE_MIME_TYPES:
        return ""
    if mime_type == PDF_MIME_TYPE and not _PYMUPDF_AVAILABLE:
        return ""
    if mime_type in IMAGE_MIME_TYPES:
        return _ocr_image_bytes(file_bytes)
    if mime_type == PDF_MIME_TYPE:
        return _ocr_pdf_bytes(file_bytes)
    logger.warning(f"Unsupported MIME type for OCR: {mime_type} ({filename})")
    return ""
