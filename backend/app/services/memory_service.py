from backend.app.config import MAX_HISTORY_MESSAGES


# --------------------------------------------------
# In-memory conversation storage
# --------------------------------------------------

conversation_histories = {}


def add_message(
    session_id: str,
    role: str,
    content: str
):
    """
    Add a message to a session's conversation history.
    """

    if not session_id or not session_id.strip():
        raise ValueError(
            "session_id cannot be empty."
        )

    if role not in {"user", "assistant"}:
        raise ValueError(
            "role must be either 'user' or 'assistant'."
        )

    if not content or not content.strip():
        raise ValueError(
            "content cannot be empty."
        )

    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    conversation_histories[session_id].append(
        {
            "role": role,
            "content": content
        }
    )

    # Keep only the most recent messages
    conversation_histories[session_id] = (
        conversation_histories[session_id][
            -MAX_HISTORY_MESSAGES:
        ]
    )


def get_history(session_id: str):
    """
    Return the conversation history for a session.
    """

    if not session_id or not session_id.strip():
        return []

    return conversation_histories.get(
        session_id,
        []
    )


def clear_history(session_id: str):
    """
    Clear the conversation history for a session.
    """

    if not session_id or not session_id.strip():
        return

    conversation_histories.pop(
        session_id,
        None
    )