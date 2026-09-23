from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.config import (
    CHUNKS_FILE,
    FAISS_INDEX_FILE
)

from backend.app.services.pdf_service import (
    save_pdf,
    extract_text_from_pdf
)

from backend.app.services.text_service import clean_text
from backend.app.services.chunk_service import chunk_text

from backend.app.services.embedding_service import (
    generate_embeddings,
    create_faiss_index,
    save_faiss_index
)

from backend.app.services.retrieval_service import (
    save_chunks,
    load_chunks
)


router = APIRouter()


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Upload a PDF, extract and clean its text,
    create chunks and embeddings, and rebuild
    the FAISS index using all uploaded documents.
    """

    # --------------------------------------------------
    # Step 1: Validate file type
    # --------------------------------------------------

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A valid PDF filename is required."
        )

    # Keep only the filename component.
    filename = Path(file.filename).name

    try:

        # --------------------------------------------------
        # Step 2: Check existing chunks
        # --------------------------------------------------

        try:
            existing_chunks = load_chunks(CHUNKS_FILE)

        except FileNotFoundError:
            existing_chunks = []

        # --------------------------------------------------
        # Step 3: Check for duplicate document
        # --------------------------------------------------

        existing_filenames = {
            chunk["filename"]
            for chunk in existing_chunks
            if isinstance(chunk, dict)
            and "filename" in chunk
        }

        if filename in existing_filenames:
            raise HTTPException(
                status_code=409,
                detail=f"{filename} has already been uploaded."
            )

        # --------------------------------------------------
        # Step 4: Save uploaded PDF
        # --------------------------------------------------

        file.filename = filename

        file_path = save_pdf(file)

        # --------------------------------------------------
        # Step 5: Extract PDF text
        # --------------------------------------------------

        pdf_text = extract_text_from_pdf(file_path)

        if not pdf_text.strip():
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text."
            )

        # --------------------------------------------------
        # Step 6: Clean extracted text
        # --------------------------------------------------

        cleaned_text = clean_text(pdf_text)

        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="No usable text was found in the PDF."
            )

        # --------------------------------------------------
        # Step 7: Create chunks
        # --------------------------------------------------

        raw_chunks = chunk_text(cleaned_text)

        if not raw_chunks:
            raise HTTPException(
                status_code=400,
                detail="Unable to create text chunks from the PDF."
            )

        # --------------------------------------------------
        # Step 8: Add document metadata
        # --------------------------------------------------

        new_chunks = [
            {
                "filename": filename,
                "text": chunk
            }
            for chunk in raw_chunks
        ]

        # --------------------------------------------------
        # Step 9: Combine existing and new chunks
        # --------------------------------------------------

        all_chunks = existing_chunks + new_chunks

        # --------------------------------------------------
        # Step 10: Save chunks
        # --------------------------------------------------

        save_chunks(
            all_chunks,
            CHUNKS_FILE
        )

        # --------------------------------------------------
        # Step 11: Generate embeddings
        # --------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = generate_embeddings(texts)

        # --------------------------------------------------
        # Step 12: Create FAISS index
        # --------------------------------------------------

        index = create_faiss_index(
            embeddings
        )

        # --------------------------------------------------
        # Step 13: Save FAISS index
        # --------------------------------------------------

        save_faiss_index(
            index,
            FAISS_INDEX_FILE
        )

        # --------------------------------------------------
        # Step 14: Return success response
        # --------------------------------------------------

        return {
            "status": "success",
            "filename": filename,
            "new_chunks": len(new_chunks),
            "total_chunks": len(all_chunks),
            "embedding_dimension": len(embeddings[0]),
            "vectors_stored": index.ntotal,
            "message": (
                "PDF added successfully and the FAISS "
                "index was rebuilt using all documents."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Unable to process the uploaded PDF."
        ) from exc