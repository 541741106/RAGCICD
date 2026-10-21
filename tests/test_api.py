from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from haystack import Document

from llmrag.api import app, get_client, to_ask_response
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
    fake_client.models.generate_content.return_value.text = "A semantic label [1]."

    app.dependency_overrides[get_client] = lambda: fake_client

    try:
        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "What does image segmentation assign to each pixel?",
                    "top_k": 1,
                },
            )
    finally:
        app.dependency_overrides.pop(get_client, None)

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "A semantic label [1]."
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["evidence_id"] == 1
    assert data["evidence"][0]["paper_id"] == "example-segmentation"

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
    app.dependency_overrides[get_client] = lambda: fake_client

    try:
        with TestClient(app) as client:
            response = client.post("/ask", json=payload)
    finally:
        app.dependency_overrides.pop(get_client, None)

    assert response.status_code == 422
    fake_client.models.generate_content.assert_not_called()


def test_ask_returns_502_when_model_response_is_empty():
    fake_client = Mock()
    fake_client.models.generate_content.return_value.text = ""

    app.dependency_overrides[get_client] = lambda: fake_client

    try:
        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "What does image segmentation assign to each pixel?",
                    "top_k": 1,
                },
            )
    finally:
        app.dependency_overrides.pop(get_client, None)

    assert response.status_code == 502
    assert response.json()["detail"] == "Generation failed to produce a valid answer."
    fake_client.models.generate_content.assert_called_once()
