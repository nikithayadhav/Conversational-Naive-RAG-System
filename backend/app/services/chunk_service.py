import re


# --------------------------------------------------
# Chunking configuration
# --------------------------------------------------

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
):
    """
    Split text into sentence-aware chunks.

    The function:
    - normalizes whitespace
    - preserves complete sentences where possible
    - keeps chunks around the configured chunk size
    - creates overlap using complete sentences
    - handles sentences larger than the chunk size
    """

    # --------------------------------------------------
    # Validate configuration
    # --------------------------------------------------

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if not text or not text.strip():
        return []

    # --------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # --------------------------------------------------
    # Split text into sentences
    # --------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    current_sentences = []
    current_length = 0

    # --------------------------------------------------
    # Build chunks
    # --------------------------------------------------

    for sentence in sentences:

        sentence_length = len(sentence)

        # --------------------------------------------------
        # Handle a sentence larger than chunk_size
        # --------------------------------------------------

        if sentence_length > chunk_size:

            # Save any existing chunk first
            if current_sentences:
                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_length = 0

            # Split the long sentence into smaller pieces
            for start in range(
                0,
                sentence_length,
                chunk_size
            ):
                chunks.append(
                    sentence[
                        start:start + chunk_size
                    ]
                )

            continue

        # --------------------------------------------------
        # Check whether sentence fits
        # --------------------------------------------------

        if (
            current_sentences
            and current_length + sentence_length + 1
            > chunk_size
        ):

            # Save current chunk
            chunks.append(
                " ".join(current_sentences)
            )

            # --------------------------------------------------
            # Create sentence-level overlap
            # --------------------------------------------------

            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(
                current_sentences
            ):

                previous_length = (
                    len(previous_sentence) + 1
                )

                if (
                    overlap_length + previous_length
                    <= overlap
                ):
                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_length += previous_length

                else:
                    break

            current_sentences = overlap_sentences
            current_length = overlap_length

        # --------------------------------------------------
        # Add current sentence
        # --------------------------------------------------

        current_sentences.append(
            sentence
        )

        current_length += (
            sentence_length + 1
        )

    # --------------------------------------------------
    # Add final chunk
    # --------------------------------------------------

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks