from unittest.mock import Mock, call

import pytest
from haystack import Document

from llmrag.evaluation import collect_dense_retrieval_results, collect_retrieval_results


def test_collect_retrieval_results_preserves_question_mapping():
    cases = [
        {"id": "q1", "question": "First question?", "answerable": True},
        {"id": "q2", "question": "Second question?", "answerable": False},
    ]
    document = Document(
        content="Retrieved evidence.",
        meta={
            "paper_id": "example-paper",
            "page_number": 2,
            "chunk_number": 3,
        },
        score=1.5,
    )
    retriever = Mock()
    retriever.run.side_effect = [
        {"documents": [document]},
        {"documents": []},
    ]

    results = collect_retrieval_results(
        cases,
        retriever=retriever,
        top_k=2,
    )

    assert retriever.run.call_args_list == [
        call(query="First question?", top_k=2),
        call(query="Second question?", top_k=2),
    ]

    assert results == [
        {
            "id": "q1",
            "question": "First question?",
            "answerable": True,
            "retrieved": [
                {
                    "rank": 1,
                    "paper_id": "example-paper",
                    "page_number": 2,
                    "chunk_number": 3,
                    "score": 1.5,
                    "content": "Retrieved evidence.",
                }
            ],
        },
        {
            "id": "q2",
            "question": "Second question?",
            "answerable": False,
            "retrieved": [],
        },
    ]


def test_collect_dense_results_preserves_question_vector_mapping():
    cases = [
        {"id": "q1", "question": "First?", "answerable": True},
        {"id": "q2", "question": "Second?", "answerable": False},
    ]
    query_vectors = [[1.0, 0.0], [0.0, 1.0]]
    document = Document(
        content="Evidence.",
        meta={
            "paper_id": "paper-a",
            "page_number": 2,
            "chunk_number": 3,
        },
        score=0.8,
    )

    retriever = Mock()
    retriever.run.side_effect = [
        {"documents": [document]},
        {"documents": []},
    ]

    results = collect_dense_retrieval_results(
        cases,
        query_vectors,
        retriever=retriever,
        top_k=2,
    )

    assert retriever.run.call_args_list == [
        call(query_embedding=[1.0, 0.0], top_k=2),
        call(query_embedding=[0.0, 1.0], top_k=2),
    ]

    assert results == [
        {
            "id": "q1",
            "question": "First?",
            "answerable": True,
            "retrieved": [
                {
                    "rank": 1,
                    "paper_id": "paper-a",
                    "page_number": 2,
                    "chunk_number": 3,
                    "score": 0.8,
                    "content": "Evidence.",
                }
            ],
        },
        {
            "id": "q2",
            "question": "Second?",
            "answerable": False,
            "retrieved": [],
        },
    ]


def test_collect_dense_results_rejects_mismatched_counts():
    retriever = Mock()

    with pytest.raises(
        ValueError,
        match="question count must match query vector count",
    ):
        collect_dense_retrieval_results(
            [{"id": "q1", "question": "First?", "answerable": True}],
            [],
            retriever=retriever,
        )

    retriever.run.assert_not_called()
