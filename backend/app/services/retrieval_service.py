import json
import logging

from backend.app.services.embedding_service import (
    load_faiss_index,
    generate_query_embedding
)

from backend.app.config import (
    TOP_K,
    RELEVANCE_THRESHOLD,
    CHUNKS_FILE,
    FAISS_INDEX_FILE
)

logger = logging.getLogger(__name__)


def search_index(index, query_embedding, top_k=TOP_K):
    """
    Search the FAISS index and return the closest matching chunks.
    """

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return distances, indices


def save_chunks(chunks, file_path):
    """
    Save all chunks into a JSON file.
    """

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_chunks(file_path):
    """
    Load all saved chunks from JSON.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def retrieve_context(
    question: str,
    top_k: int = TOP_K
):
    """
    Retrieve relevant chunks from FAISS.

    Returns:
        {
            "context": "...",
            "sources": [
                {
                    "filename": "...",
                    "chunk_id": ...,
                    "distance": ...
                }
            ]
        }

    Raises:
        ValueError: If top_k is invalid.
        RuntimeError: If retrieval infrastructure fails.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    try:

        # --------------------------------------------------
        # Step 1: Generate embedding for the question
        # --------------------------------------------------

        query_embedding = generate_query_embedding(
            question
        )

        # --------------------------------------------------
        # Step 2: Load FAISS index
        # --------------------------------------------------

        index = load_faiss_index(
            FAISS_INDEX_FILE
        )

        # --------------------------------------------------
        # Step 3: Search FAISS index
        # --------------------------------------------------

        distances, indices = search_index(
            index,
            query_embedding,
            top_k=top_k
        )

        

        logger.debug("FAISS distances: %s", distances)

        logger.debug("FAISS indices: %s", indices)
        

        # --------------------------------------------------
        # Step 4: Load chunks
        # --------------------------------------------------

        chunks = load_chunks(
            CHUNKS_FILE
        )

        # --------------------------------------------------
        # Step 5: Validate vector/chunk consistency
        # --------------------------------------------------

        if index.ntotal != len(chunks):
            raise RuntimeError(
                "FAISS index and chunk store are out of sync."
            )

        retrieved_chunks = []

        sources = []

        seen_chunks = set()

        # --------------------------------------------------
        # Step 6: Process FAISS results
        # --------------------------------------------------

        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx == -1:
                logger.debug("Rejected chunk %s", idx)

                continue

            if distance > RELEVANCE_THRESHOLD:
                logger.debug(
    "Rejected chunk %s with distance %.4f",
    idx,
    distance
)
                
                continue

            chunk = chunks[idx]

            chunk_text = chunk["text"]

            if chunk_text in seen_chunks:
                logger.debug(
    "Skipped duplicate chunk %s",
    idx
)
                continue

            logger.debug(
    "Accepted chunk %s from %s with distance %.4f",
    idx,
    chunk["filename"],
    distance
)

            retrieved_chunks.append(
                chunk_text
            )

            sources.append(
                {
                    "filename": chunk["filename"],
                    "chunk_id": int(idx),
                    "distance": float(distance)
                }
            )

            seen_chunks.add(
                chunk_text
            )

        # --------------------------------------------------
        # Step 7: Combine retrieved chunks
        # --------------------------------------------------

        context = "\n\n".join(
            retrieved_chunks
        )

        # --------------------------------------------------
        # Step 8: Return context and source metadata
        # --------------------------------------------------

        return {
            "context": context,
            "sources": sources
        }

    except Exception as e:

        logger.exception("Retrieval failed")


        raise RuntimeError(
            "Unable to retrieve information from the document store."
        ) from e