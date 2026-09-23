from groq import Groq

from backend.app.config import (
    GROQ_API_KEY,
    GROQ_MODEL
)


# --------------------------------------------------
# Groq client
# --------------------------------------------------

client = Groq(
    api_key=GROQ_API_KEY
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

DOCUMENT_NOT_FOUND_MESSAGE = (
    "I don't know based on the provided document."
)


# --------------------------------------------------
# Generate response
# --------------------------------------------------

def generate_response(
    question: str,
    context: str,
    session_id: str
):
    """
    Generate an answer using only the retrieved
    document context.

    The conversation history is handled by the
    query rewriting component before retrieval.
    """

    # --------------------------------------------------
    # Step 1: Validate retrieved context
    # --------------------------------------------------

    if not context or not context.strip():
        return DOCUMENT_NOT_FOUND_MESSAGE

    try:

        # --------------------------------------------------
        # Step 2: Build system instructions
        # --------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant.\n\n"

                    "Your ONLY source of factual information is the "
                    "CURRENT RETRIEVED DOCUMENT.\n\n"

                    "STRICT RULES:\n"

                    "1. Answer the CURRENT QUESTION using ONLY the "
                    "CURRENT RETRIEVED DOCUMENT.\n\n"

                    "2. Do NOT use your own knowledge or outside "
                    "information.\n\n"

                    "3. Do NOT invent, guess, or assume information.\n\n"

                    "4. If the CURRENT RETRIEVED DOCUMENT contains "
                    "enough information to answer the question, "
                    "answer directly using that information.\n\n"

                    "5. If the CURRENT RETRIEVED DOCUMENT does not "
                    "contain enough information, respond exactly with:\n"
                    f"{DOCUMENT_NOT_FOUND_MESSAGE}\n\n"

                    "6. Keep the answer clear and directly related "
                    "to the CURRENT QUESTION.\n\n"

                    "7. Do NOT mention the retrieval process unless "
                    "the user asks about it."
                )
            }
        ]

        # --------------------------------------------------
        # Step 3: Add current context and question
        # --------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": (
                    "==============================\n"
                    "CURRENT RETRIEVED DOCUMENT\n"
                    "==============================\n\n"
                    f"{context}\n\n"

                    "==============================\n"
                    "CURRENT QUESTION\n"
                    "==============================\n\n"
                    f"{question}\n\n"

                    "==============================\n"
                    "ANSWERING INSTRUCTION\n"
                    "==============================\n\n"

                    "Answer the CURRENT QUESTION using ONLY "
                    "the CURRENT RETRIEVED DOCUMENT."
                )
            }
        )

        # --------------------------------------------------
        # Step 4: Call Groq
        # --------------------------------------------------
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0
        )

        

        # --------------------------------------------------
        # Step 5: Extract answer
        # --------------------------------------------------

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError(
                "LLM returned an empty response."
            )

        return answer.strip()

    except RuntimeError:
        raise

    except Exception as e:

        print(
            "Groq Error:",
            e
        )

        raise RuntimeError(
            "Unable to generate an answer from the language model."
        ) from e