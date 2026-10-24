import pytest
from haystack import Document

from llmrag.dense_retrieval import build_dense_retriever


def test_dense_retriever_ranks_vectors_and_preserves_sources():
    chunks = [
        Document(content="First text", meta={"paper_id": "a", "page_number": 1}),
        Document(content="Second text", meta={"paper_id": "b", "page_number": 2}),
    ]

    retriever = build_dense_retriever(
        chunks,
        vectors=[[0.0, 1.0], [1.0, 0.0]],
    )

    documents = retriever.run(
        query_embedding=[1.0, 0.0],
        top_k=2,
    )["documents"]

    assert len(documents) == 2
    assert documents[0].content == "Second text"
    assert documents[0].meta == {"paper_id": "b", "page_number": 2}
    assert documents[0].score == pytest.approx(1.0)
    assert chunks[0].embedding is None
    assert chunks[1].embedding is None


def test_dense_retriever_rejects_mismatched_counts():
    chunks = [Document(content="One text")]

    with pytest.raises(ValueError, match="chunk count must match vector count"):
        build_dense_retriever(chunks, vectors=[])
