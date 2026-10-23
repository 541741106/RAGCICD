from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


def collect_retrieval_results(
    cases: list[dict],
    *,
    retriever: InMemoryBM25Retriever,
    top_k: int = 3,
) -> list[dict]:
    """Collect ranked evidence for manual retrieval evaluation."""
    results: list[dict] = []

    for case in cases:
        documents = retriever.run(
            query=case["question"],
            top_k=top_k,
        )["documents"]

        retrieved: list[dict] = []
        for rank, document in enumerate(documents, start=1):
            retrieved.append(
                {
                    "rank": rank,
                    "paper_id": document.meta["paper_id"],
                    "page_number": document.meta["page_number"],
                    "chunk_number": document.meta["chunk_number"],
                    "score": document.score,
                    "content": document.content,
                }
            )

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answerable": case["answerable"],
                "retrieved": retrieved,
            }
        )

    return results
