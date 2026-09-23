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


def rewrite_question(
    question: str,
    history: list
):
    """
    Rewrite the user's latest question into a
    standalone question using conversation history.

    The rewriter:
    - resolves references such as it, its, this, that
    - resolves numbered references such as first, second, third
    - preserves the original meaning
    - does not answer the question
    - returns the original question if rewriting fails
    """

    # --------------------------------------------------
    # Step 1: Validate question
    # --------------------------------------------------

    if not question or not question.strip():
        return question

    try:

        # --------------------------------------------------
        # Step 2: System instructions
        # --------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a question rewriting component in a "
                    "Conversational RAG system.\n\n"

                    "Your ONLY task is to rewrite the user's latest "
                    "question into a standalone question.\n\n"

                    "STRICT RULES:\n\n"

                    "1. Use ONLY information explicitly present in "
                    "the conversation history and the latest question.\n\n"

                    "2. NEVER invent, assume, or introduce a new "
                    "person, company, product, organization, object, "
                    "subject, or entity.\n\n"

                    "3. If the latest question contains a reference "
                    "to something mentioned earlier, resolve that "
                    "reference using the conversation history.\n\n"

                    "4. References include words or phrases such as:\n"
                    "'it', 'its', 'they', 'them', 'this', 'that', "
                    "'the product', 'the subject', 'the third subject', "
                    "'the first one', 'the second one', etc.\n\n"

                    "5. For numbered references such as "
                    "'the first subject', 'the second subject', "
                    "'the third subject', identify the relevant list "
                    "from the conversation history.\n\n"

                    "6. Preserve the exact entity or topic name from "
                    "the conversation history whenever possible.\n\n"

                    "7. Preserve the original meaning of the question.\n\n"

                    "8. If the latest question is already standalone "
                    "and does not contain an unresolved reference, "
                    "return it unchanged.\n\n"

                    "9. Do NOT answer the question.\n\n"

                    "10. Do NOT add explanations.\n\n"

                    "11. Return ONLY the rewritten standalone question.\n\n"

                    "12. If a reference cannot be resolved confidently, "
                    "return the original question unchanged.\n\n"

                    "EXAMPLE 1:\n"
                    "Conversation:\n"
                    "User: What is InferAPI?\n"
                    "Assistant: InferAPI is an API documentation system.\n"
                    "User: What problem does it solve?\n\n"

                    "Correct rewrite:\n"
                    "What problem does InferAPI solve?\n\n"

                    "EXAMPLE 2:\n"
                    "Conversation:\n"
                    "User: What subjects are included in the MCA 1st "
                    "semester examination?\n"
                    "Assistant: The MCA 1st semester examination includes "
                    "Introductory Programming, Digital Systems, "
                    "Mathematical Foundation of Computer Science, "
                    "Accounting and Financial Management, and "
                    "Probability and Statistics.\n"
                    "User: What is the third subject?\n\n"

                    "Correct rewrite:\n"
                    "What is the third subject in the MCA 1st semester "
                    "examination?\n\n"

                    "Incorrect rewrite:\n"
                    "What is the third subject?\n"
                )
            }
        ]

        # --------------------------------------------------
        # Step 3: Add conversation history
        # --------------------------------------------------

        if history:
            messages.extend(history)

        # --------------------------------------------------
        # Step 4: Add current question
        # --------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # --------------------------------------------------
        # Step 5: Ask Groq to rewrite
        # --------------------------------------------------

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0
        )

        # --------------------------------------------------
        # Step 6: Extract response
        # --------------------------------------------------

        rewritten_question = (
            response.choices[0]
            .message.content
            .strip()
        )

        # --------------------------------------------------
        # Step 7: Safety fallback
        # --------------------------------------------------

        if not rewritten_question:
            return question

        return rewritten_question

    except Exception as e:

        # If rewriting fails,
        # continue using the original question.
        return question