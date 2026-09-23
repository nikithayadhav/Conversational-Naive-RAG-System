
import shutil
from pathlib import Path

import fitz

from backend.app.config import UPLOAD_FOLDER


# --------------------------------------------------
# Ensure upload directory exists
# --------------------------------------------------

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


def save_pdf(file):
    """
    Save an uploaded PDF to the configured upload directory.

    The filename is expected to be validated and sanitized
    by the upload route before reaching this service.
    """

    filename = Path(file.filename).name

    file_path = UPLOAD_FOLDER / filename

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return file_path


def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF document.
    """

    text_parts = []

    with fitz.open(pdf_path) as document:

        for page in document:

            page_text = page.get_text()

            if page_text:
                text_parts.append(
                    page_text
                )

    return "\n".join(text_parts)

