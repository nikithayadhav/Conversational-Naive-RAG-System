import os
from backend.app.services.pdf_service import extract_text_from_pdf
from backend.app.services.text_service import clean_text
from backend.app.services.chunk_service import chunk_text
from backend.app.services.embedding_service import (
    generate_embeddings,
    create_faiss_index,
    save_faiss_index
)
from backend.app.services.retrieval_service import save_chunks
from backend.app.config import UPLOAD_FOLDER


UPLOAD_DIR = UPLOAD_FOLDER
VECTOR_DIR = "backend/vector_store"

CHUNKS_PATH = os.path.join(
    VECTOR_DIR,
    "chunks.json"
)

FAISS_PATH = os.path.join(
    VECTOR_DIR,
    "faiss_index.bin"
)


PDF_FILES = [
    "finleafe_unique_features.pdf",
    "InferAPI.pdf"
]


def main():

    all_chunks = []

    print("=" * 60)
    print("REBUILDING VECTOR STORE")
    print("=" * 60)

    for filename in PDF_FILES:

        pdf_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        print()
        print(f"Processing: {filename}")

        # Extract PDF text
        pdf_text = extract_text_from_pdf(
            pdf_path
        )

        print(
            f"Extracted characters: {len(pdf_text)}"
        )

        # Clean text
        cleaned_text = clean_text(
            pdf_text
        )

        print(
            f"Cleaned characters: {len(cleaned_text)}"
        )

        # Create sentence-aware chunks
        raw_chunks = chunk_text(
            cleaned_text,
            chunk_size=500,
            overlap=100
        )

        print(
            f"Created chunks: {len(raw_chunks)}"
        )

        # Add metadata
        document_chunks = [
            {
                "filename": filename,
                "text": chunk
            }
            for chunk in raw_chunks
        ]

        all_chunks.extend(
            document_chunks
        )

    print()
    print("=" * 60)
    print("CHUNK SUMMARY")
    print("=" * 60)

    print(
        "Total chunks:",
        len(all_chunks)
    )

    for filename in PDF_FILES:

        count = sum(
            chunk["filename"] == filename
            for chunk in all_chunks
        )

        print(
            f"{filename}: {count}"
        )

    # Save chunks
    save_chunks(
        all_chunks,
        CHUNKS_PATH
    )

    print()
    print(
        f"Saved chunks to: {CHUNKS_PATH}"
    )

    # Extract text for embeddings
    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print()
    print("Generating embeddings...")

    embeddings = generate_embeddings(
        texts
    )

    print(
        "Embedding dimension:",
        len(embeddings[0])
    )

    # Create FAISS index
    print()
    print("Creating FAISS index...")

    index = create_faiss_index(
        embeddings
    )

    # Save FAISS
    save_faiss_index(
        index,
        FAISS_PATH
    )

    print()
    print("=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)

    print(
        "Total chunks:",
        len(all_chunks)
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print(
        "Embedding dimension:",
        len(embeddings[0])
    )

    print(
        "FAISS file:",
        FAISS_PATH
    )


if __name__ == "__main__":
    main()