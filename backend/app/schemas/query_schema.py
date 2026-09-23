from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):

    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)

    @field_validator("session_id", "question")
    @classmethod
    def validate_not_blank(cls, value: str):

        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty or whitespace.")

        return value


class Source(BaseModel):

    filename: str
    chunk_id: int
    distance: float


class QueryResponse(BaseModel):

    session_id: str
    question: str
    answer: str
    rewritten_question: str
    context: str
    sources: list[Source]