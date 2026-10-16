from dataclasses import dataclass

from google import genai
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from llmrag.bm25_baseline import retrieve_examples
from llmrag.generation import generate_answer
from llmrag.prompting import build_evidence_prompt


@dataclass
class RagResult:
    answer: str
    documents: list[Document]


def answer_question(
    question: str,
    *,
    client: genai.Client,
    retriever: InMemoryBM25Retriever | None = None,
    top_k: int = 2,
    model: str = "gemini-3.5-flash-lite",
) -> RagResult:
    """Answer using a supplied retriever or the synthetic collection."""
    if not question.strip():
        raise ValueError("question must not be empty")

    if retriever is None:
        documents = retrieve_examples(question, top_k=top_k)
    else:
        documents = retriever.run(query=question, top_k=top_k)["documents"]

    if not documents:
        return RagResult(
            answer="Insufficient evidence was found to answer the question.",
            documents=[],
        )

    prompt = build_evidence_prompt(question, documents)

    answer = generate_answer(prompt, client=client, model=model)

    return RagResult(answer=answer, documents=documents)
