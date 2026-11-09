from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from google import genai
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)

from llmrag.rag import RagResult, answer_question_with_pgvector
from llmrag.schemas import AskRequest, AskResponse, EvidenceResponse
from llmrag.storage import open_document_store

app = FastAPI(title="PaperTrace — Evidence-Grounded Research Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "OK"}


def to_ask_response(result: RagResult) -> AskResponse:
    """Convert a RAG result into the public API response."""
    evidence: list[EvidenceResponse] = []

    for evidence_id, document in enumerate(result.documents, start=1):
        evidence.append(
            EvidenceResponse(
                evidence_id=evidence_id,
                paper_id=document.meta["paper_id"],
                paper_title=document.meta["paper_title"],
                page_number=document.meta["page_number"],
                content=document.content,
            )
        )

    return AskResponse(
        answer=result.answer,
        evidence=evidence,
    )


def get_client() -> Iterator[genai.Client]:
    """Provide a Gemini client for one request."""
    with genai.Client() as client:
        yield client


def get_retriever() -> Iterator[PgvectorEmbeddingRetriever]:
    """Provide a database retriever and release request resources."""
    store = open_document_store()

    try:
        retriever = PgvectorEmbeddingRetriever(document_store=store)
        yield retriever
    finally:
        store.close()


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    client: Annotated[genai.Client, Depends(get_client)],
    retriever: Annotated[
        PgvectorEmbeddingRetriever,
        Depends(get_retriever),
    ],
) -> AskResponse:
    try:
        result = answer_question_with_pgvector(
            request.question,
            client=client,
            retriever=retriever,
            top_k=request.top_k,
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=502,
            detail="Question answering failed to produce a valid answer.",
        ) from error

    return to_ask_response(result)
