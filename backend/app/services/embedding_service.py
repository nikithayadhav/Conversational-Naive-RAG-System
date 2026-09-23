import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# --------------------------------------------------
# Generate document embeddings
# --------------------------------------------------

def generate_embeddings(chunks: list[str]):
    """
    Generate embeddings for document chunks.
    """

    if not chunks:
        return np.empty(
            (0, model.get_sentence_embedding_dimension()),
            dtype="float32"
        )

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


# --------------------------------------------------
# Create FAISS index
# --------------------------------------------------

def create_faiss_index(embeddings):
    """
    Create a FAISS L2 index from embeddings.
    """

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            "Embeddings must be a non-empty 2D array."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# --------------------------------------------------
# Save FAISS index
# --------------------------------------------------

def save_faiss_index(index, file_path):
    """
    Save the FAISS index to disk.
    """

    faiss.write_index(
        index,
        str(file_path)
    )


# --------------------------------------------------
# Load FAISS index
# --------------------------------------------------

def load_faiss_index(file_path):
    """
    Load a FAISS index from disk.
    """

    return faiss.read_index(
        str(file_path)
    )


# --------------------------------------------------
# Generate query embedding
# --------------------------------------------------

def generate_query_embedding(question: str):
    """
    Generate an embedding for a user question.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    return embedding.astype("float32")