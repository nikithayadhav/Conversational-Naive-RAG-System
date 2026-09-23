
from fastapi import APIRouter, HTTPException

from backend.app.schemas.query_schema import (
    QueryRequest,
    QueryResponse
)

from backend.app.services.retrieval_service import retrieve_context
from backend.app.services.llm_service import generate_response
from backend.app.services.memory_service import (
    add_message,
    get_history,
    clear_history
)
from backend.app.services.query_rewrite_service import rewrite_question


router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse
)
async def ask_question(request: QueryRequest):
    """
    Process a user question through the conversational RAG pipeline.

    Pipeline:
        1. Retrieve conversation history.
        2. Rewrite the current question into a standalone question.
        3. Retrieve relevant document chunks.
        4. Generate an answer using retrieved context.
        5. Store the conversation messages.
        6. Return the structured response.
    """

    try:
        # --------------------------------------------------
        # Step 1: Retrieve conversation history
        # --------------------------------------------------

        history = get_history(
            request.session_id
        )

    

        # --------------------------------------------------
        # Step 2: Rewrite the question
        # --------------------------------------------------

        rewritten_question = rewrite_question(
            question=request.question,
            history=history
        )

        # --------------------------------------------------
        # Step 3: Retrieve relevant document context
        # --------------------------------------------------

        retrieval_result = retrieve_context(
            rewritten_question
        )

        context = retrieval_result["context"]
        sources = retrieval_result["sources"]

        # --------------------------------------------------
        # Step 4: Generate answer
        # --------------------------------------------------

        answer = generate_response(
            question=rewritten_question,
            context=context,
            session_id=request.session_id
        )

        # --------------------------------------------------
        # Step 5: Store the original user question
        # --------------------------------------------------

        add_message(
            session_id=request.session_id,
            role="user",
            content=request.question
        )

        # --------------------------------------------------
        # Step 6: Store the assistant answer
        # --------------------------------------------------

        add_message(
            session_id=request.session_id,
            role="assistant",
            content=answer
        )

        # --------------------------------------------------
        # Step 7: Return structured response
        # --------------------------------------------------

        return QueryResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            rewritten_question=rewritten_question,
            context=context,
            sources=sources
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query."
        ) from exc


@router.get(
    "/history/{session_id}"
)
async def get_conversation_history(
    session_id: str
):
    """
    Return the conversation history for a session.
    """

    return {
        "session_id": session_id,
        "history": get_history(session_id)
    }


@router.delete(
    "/history/{session_id}"
)
async def clear_conversation_history(
    session_id: str
):
    """
    Clear the conversation history for a session.
    """

    clear_history(
        session_id
    )

    return {
        "session_id": session_id,
        "message": "Conversation history cleared successfully"
    }

