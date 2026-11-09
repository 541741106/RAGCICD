from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from haystack import Document

from llmrag.api import app, get_client, get_retriever, to_ask_response
from llmrag.rag import RagResult


def test_health_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_to_ask_response_preserves_evidence_mapping():
    result = RagResult(
        answer="An answer [1].",
        documents=[
            Document(
                content="Evidence A",
                meta={
                    "paper_id": "paper-a",
                    "paper_title": "Paper A",
                    "page_number": 2,
                },
            ),
            Document(
                content="Evidence B",
                meta={
                    "paper_id": "paper-b",
                    "paper_title": "Paper B",
                    "page_number": 5,
                },
            ),
        ],
    )

    response = to_ask_response(result)

    assert response.answer == result.answer

    assert response.model_dump()["evidence"] == [
        {
            "evidence_id": 1,
            "paper_id": "paper-a",
            "paper_title": "Paper A",
            "page_number": 2,
            "content": "Evidence A",
        },
        {
            "evidence_id": 2,
            "paper_id": "paper-b",
            "paper_title": "Paper B",
            "page_number": 5,
            "content": "Evidence B",
        },
    ]


def test_to_ask_response_handles_empty_evidence():
    response_data = to_ask_response(
        RagResult(answer="Insufficient evidence.", documents=[])
    ).model_dump()

    assert response_data == {
        "answer": "Insufficient evidence.",
        "evidence": [],
    }


def test_ask_returns_answer_and_evidence():
    fake_client = Mock()
    fake_client.models.generate_content.return_value.text = "An answer [1]."

    query_vector = [0.1] * 768
    document = Document(
        content="Identity shortcuts pass the input unchanged.",
        meta={
            "paper_id": "example-paper",
            "paper_title": "Example Paper",
            "page_number": 2,
            "chunk_number": 1,
        },
    )
    fake_retriever = Mock(spec=["run"])
    fake_retriever.run.return_value = {"documents": [document]}

    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_retriever] = lambda: fake_retriever

    try:
        with patch(
            "llmrag.rag.embed_texts",
            autospec=True,
            return_value=[query_vector],
        ) as embed:
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": "What are identity shortcuts?",
                        "top_k": 1,
                    },
                )

            embed.assert_called_once_with(
                ["What are identity shortcuts?"],
                client=fake_client,
                task_type="RETRIEVAL_QUERY",
            )
    finally:
        app.dependency_overrides.pop(get_client, None)
        app.dependency_overrides.pop(get_retriever, None)

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "An answer [1]."
    assert data["evidence"] == [
        {
            "evidence_id": 1,
            "paper_id": "example-paper",
            "paper_title": "Example Paper",
            "page_number": 2,
            "content": "Identity shortcuts pass the input unchanged.",
        }
    ]

    fake_retriever.run.assert_called_once_with(query_embedding=query_vector, top_k=1)

    fake_client.models.generate_content.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   ", "top_k": 3},
        {"question": "What is ResNet?", "top_k": 0},
        {"question": "What is ResNet?", "top_k": 11},
    ],
)
def test_ask_rejects_invalid_request(payload):
    fake_client = Mock()
    fake_retriever = Mock(spec=["run"])

    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_retriever] = lambda: fake_retriever

    try:
        with patch("llmrag.rag.embed_texts", autospec=True) as embed:
            with TestClient(app) as client:
                response = client.post("/ask", json=payload)
            embed.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_client, None)
        app.dependency_overrides.pop(get_retriever, None)

    assert response.status_code == 422
    fake_retriever.run.assert_not_called()
    fake_client.models.generate_content.assert_not_called()


def test_ask_returns_502_when_model_response_is_empty():
    fake_client = Mock()
    fake_client.models.generate_content.return_value.text = ""

    query_vector = [0.1] * 768
    document = Document(
        content="Evidence for the question.",
        meta={
            "paper_id": "example-paper",
            "paper_title": "Example Paper",
            "page_number": 2,
        },
    )
    fake_retriever = Mock(spec=["run"])
    fake_retriever.run.return_value = {"documents": [document]}

    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_retriever] = lambda: fake_retriever

    try:
        with (
            patch(
                "llmrag.rag.embed_texts",
                autospec=True,
                return_value=[query_vector],
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/ask",
                json={
                    "question": "A question?",
                    "top_k": 1,
                },
            )
    finally:
        app.dependency_overrides.pop(get_client, None)
        app.dependency_overrides.pop(get_retriever, None)

    assert response.status_code == 502
    assert (
        response.json()["detail"]
        == "Question answering failed to produce a valid answer."
    )
    fake_retriever.run.assert_called_once_with(query_embedding=query_vector, top_k=1)
    fake_client.models.generate_content.assert_called_once()


def test_get_retriever_closes_store():
    store = Mock(spec=["close"])

    with (
        patch(
            "llmrag.api.open_document_store",
            autospec=True,
            return_value=store,
        ) as open_store,
        patch(
            "llmrag.api.PgvectorEmbeddingRetriever",
            autospec=True,
        ) as retriever_class,
    ):
        dependency = get_retriever()

        provided = next(dependency)

        open_store.assert_called_once_with()
        retriever_class.assert_called_once_with(document_store=store)
        assert provided is retriever_class.return_value
        store.close.assert_not_called()

        dependency.close()

        store.close.assert_called_once_with()


def test_ask_returns_502_when_embedding_response_is_invalid():
    fake_client = Mock()
    fake_retriever = Mock(spec=["run"])

    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_retriever] = lambda: fake_retriever

    try:
        with (
            patch(
                "llmrag.rag.embed_texts",
                autospec=True,
                side_effect=RuntimeError("embedding response contains no embeddings"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/ask",
                json={"question": "A question?", "top_k": 1},
            )
    finally:
        app.dependency_overrides.pop(get_client, None)
        app.dependency_overrides.pop(get_retriever, None)

    assert response.status_code == 502
    assert (
        response.json()["detail"]
        == "Question answering failed to produce a valid answer."
    )
    fake_retriever.run.assert_not_called()
    fake_client.models.generate_content.assert_not_called()
