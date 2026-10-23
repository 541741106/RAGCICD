from unittest.mock import Mock, call

from haystack import Document

from llmrag.evaluation import collect_retrieval_results


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
