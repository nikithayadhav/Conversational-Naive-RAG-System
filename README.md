# Conversational Naive RAG System

A conversational Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their contents.

The system supports conversational follow-up questions by using conversation history to rewrite context-dependent questions into standalone questions before retrieval.

## Features

- PDF document upload
- PDF text extraction
- Text cleaning
- Text chunking
- SentenceTransformer embeddings
- FAISS vector storage and semantic retrieval
- Relevance threshold filtering
- Source metadata for retrieved chunks
- Groq LLM integration
- Document-grounded question answering
- Unknown-question handling
- Conversation memory using session IDs
- Query rewriting for follow-up questions
- Pronoun/reference resolution
- Numbered reference resolution
- Session isolation
- Conversation history clearing
- Duplicate PDF upload protection
- PDF file validation

## RAG Pipeline

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Cleaning
    ↓
Text Chunking
    ↓
SentenceTransformer Embeddings
    ↓
FAISS Vector Store
    ↓
User Question
    ↓
Conversation History
    ↓
Query Rewriting
    ↓
Semantic Retrieval
    ↓
Relevance Filtering
    ↓
Retrieved Context
    ↓
Groq LLM
    ↓
Grounded Answer
    ↓
Conversation History