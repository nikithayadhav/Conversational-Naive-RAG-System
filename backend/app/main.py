from fastapi import FastAPI
from backend.app.routes.upload import router as upload_router
from backend.app.routes.query import router as query_router

app = FastAPI(
    title="Conversational RAG System"
)

app.include_router(upload_router)
app.include_router(query_router)

@app.get("/")
def home():
    return {
        "message": "Welcome to Conversational RAG System"
    }