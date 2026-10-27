import json
from unittest.mock import patch

import pytest
from haystack import Document

from llmrag.pgvector_ingestion import load_cached_document_vectors, main


def test_load_cached_document_vectors_preserves_order(tmp_path):
    chunks = [
        Document(content="Text A"),
        Document(content="Text B"),
    ]
    expected_vectors = [
        [0.1] * 768,
        [0.2] * 768,
    ]

    cached = {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
        "texts": ["Text A", "Text B"],
        "vectors": expected_vectors,
    }

    cache_path = tmp_path / "embeddings.json"
    cache_path.write_text(
        json.dumps(cached),
        encoding="utf-8",
    )

    vectors = load_cached_document_vectors(chunks, cache_path=cache_path)

    assert vectors == expected_vectors


def test_load_cached_document_vectors_rejects_different_texts(tmp_path):
    cache_path = tmp_path / "embeddings.json"
    cached = {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
        "texts": ["Old text"],
        "vectors": [[0.1] * 768],
    }
    original_text = json.dumps(cached)
    cache_path.write_text(original_text, encoding="utf-8")

    chunks = [Document(content="New text")]

    with pytest.raises(
        ValueError,
        match="embedding cache does not match inputs",
    ):
        load_cached_document_vectors(chunks, cache_path=cache_path)

    assert cache_path.read_text(encoding="utf-8") == original_text


@pytest.mark.parametrize(
    ("vectors", "error_type", "message"),
    [
        (None, TypeError, "cached vectors must be a list"),
        ([], ValueError, "cached vector count does not match chunk count"),
        ([[0.1] * 767], ValueError, "cached vector must have 768 dimensions"),
        (
            [["bad"] + [0.1] * 767],
            ValueError,
            "cached vector values must be finite numbers",
        ),
        (
            [[True] + [0.1] * 767],
            ValueError,
            "cached vector values must be finite numbers",
        ),
        (
            [[float("nan")] + [0.1] * 767],
            ValueError,
            "cached vector values must be finite numbers",
        ),
    ],
)
def test_load_cached_document_vectors_rejects_invalid_vectors(
    tmp_path, vectors, error_type, message
):
    chunks = [Document(content="Text A")]
    cached = {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
        "texts": ["Text A"],
        "vectors": vectors,
    }
    cache_path = tmp_path / "embeddings.json"
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    with pytest.raises(
        error_type,
        match=message,
    ):
        load_cached_document_vectors(chunks, cache_path=cache_path)


def test_main_stops_before_database_when_cache_is_invalid():
    with (
        patch("llmrag.pgvector_ingestion.load_dotenv", autospec=True),
        patch("llmrag.pgvector_ingestion.load_pdf_pages", autospec=True),
        patch("llmrag.pgvector_ingestion.chunk_pages", autospec=True),
        patch(
            "llmrag.pgvector_ingestion.load_cached_document_vectors",
            autospec=True,
        ) as load_vectors,
        patch(
            "llmrag.pgvector_ingestion.open_document_store",
            autospec=True,
        ) as open_store,
        patch(
            "llmrag.pgvector_ingestion.write_embedded_chunks",
            autospec=True,
        ) as write,
    ):
        load_vectors.side_effect = ValueError("embedding cache does not match inputs")

        with pytest.raises(ValueError, match="embedding cache does not match inputs"):
            main()

        open_store.assert_not_called()

        write.assert_not_called()
