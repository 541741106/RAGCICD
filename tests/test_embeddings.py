import json
from unittest.mock import Mock, call, patch

import pytest
from google.genai import types

from llmrag.embeddings import embed_texts, embed_texts_batched, load_or_embed_texts


def test_embed_texts_preserves_order_and_request_config():
    first_vector = [0.1] * 768
    second_vector = [0.2] * 768

    client = Mock()
    client.models.embed_content.return_value = types.EmbedContentResponse(
        embeddings=[
            types.ContentEmbedding(values=first_vector),
            types.ContentEmbedding(values=second_vector),
        ]
    )

    vectors = embed_texts(
        ["Text A", "Text B"],
        client=client,
        task_type="RETRIEVAL_DOCUMENT",
    )

    assert vectors == [first_vector, second_vector]

    client.models.embed_content.assert_called_once()
    arguments = client.models.embed_content.call_args.kwargs

    assert arguments["model"] == "gemini-embedding-001"
    assert arguments["contents"] == ["Text A", "Text B"]
    assert arguments["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert arguments["config"].output_dimensionality == 768


def test_embed_texts_skips_empty_input():
    client = Mock(spec=["models"])

    vectors = embed_texts(
        [],
        client=client,
        task_type="RETRIEVAL_DOCUMENT",
    )

    assert vectors == []
    client.models.embed_content.assert_not_called()


@pytest.mark.parametrize("blank_text", ["", "   ", "\n\t"])
def test_embed_texts_rejects_blank_text(blank_text):
    client = Mock(spec=["models"])

    with pytest.raises(ValueError, match="texts must not contain blank strings"):
        embed_texts(
            ["Valid text", blank_text],
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
        )

    client.models.embed_content.assert_not_called()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            types.EmbedContentResponse(embeddings=None),
            "embedding response contains no embeddings",
        ),
        (
            types.EmbedContentResponse(
                embeddings=[types.ContentEmbedding(values=None)]
            ),
            "embedding response contains no vector values",
        ),
        (
            types.EmbedContentResponse(embeddings=[]),
            "embedding count does not match text count",
        ),
        (
            types.EmbedContentResponse(
                embeddings=[types.ContentEmbedding(values=[0.1] * 767)]
            ),
            "embedding vector must have 768 dimensions",
        ),
    ],
)
def test_embed_texts_rejects_invalid_response(response, message):
    client = Mock()
    client.models.embed_content.return_value = response

    with pytest.raises(RuntimeError, match=message):
        embed_texts(
            ["Valid text"],
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
        )

    client.models.embed_content.assert_called_once()


def test_embed_texts_batched_preserves_batches_and_order():
    client = Mock()

    with patch("llmrag.embeddings.embed_texts", autospec=True) as embed:
        embed.side_effect = [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0]],
        ]

        vectors = embed_texts_batched(
            ["A", "B", "C", "D", "E"],
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=2,
        )

        assert embed.call_args_list == [
            call(
                ["A", "B"],
                client=client,
                task_type="RETRIEVAL_DOCUMENT",
            ),
            call(
                ["C", "D"],
                client=client,
                task_type="RETRIEVAL_DOCUMENT",
            ),
            call(
                ["E"],
                client=client,
                task_type="RETRIEVAL_DOCUMENT",
            ),
        ]
        assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_embed_texts_batched_rejects_invalid_batch_size(batch_size):
    client = Mock()

    with patch("llmrag.embeddings.embed_texts", autospec=True) as embed:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            embed_texts_batched(
                [],
                client=client,
                task_type="RETRIEVAL_DOCUMENT",
                batch_size=batch_size,
            )

        embed.assert_not_called()


def test_embed_texts_batched_validates_all_texts_before_requests():
    client = Mock()

    with patch("llmrag.embeddings.embed_texts", autospec=True) as embed:
        with pytest.raises(
            ValueError,
            match="texts must not contain blank strings",
        ):
            embed_texts_batched(
                ["A", "B", "   "],
                client=client,
                task_type="RETRIEVAL_DOCUMENT",
                batch_size=2,
            )

        embed.assert_not_called()


def test_load_or_embed_texts_reuses_matching_cache(tmp_path):
    cache_path = tmp_path / "embeddings.json"
    expected_vectors = [[0.1] * 768, [0.2] * 768]

    cached = {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
        "texts": ["Text A", "Text B"],
        "vectors": expected_vectors,
    }
    cache_path.write_text(
        json.dumps(cached),
        encoding="utf-8",
    )

    client = Mock(spec=["models"])

    vectors = load_or_embed_texts(
        ["Text A", "Text B"],
        cache_path=cache_path,
        client=client,
        task_type="RETRIEVAL_DOCUMENT",
    )

    assert vectors == expected_vectors
    client.models.embed_content.assert_not_called()


def test_load_or_embed_texts_rejects_different_texts(tmp_path):
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

    client = Mock(spec=["models"])

    with pytest.raises(ValueError, match="embedding cache does not match inputs"):
        load_or_embed_texts(
            ["New text"],
            cache_path=cache_path,
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
        )

    client.models.embed_content.assert_not_called()
    assert cache_path.read_text(encoding="utf-8") == original_text


def test_load_or_embed_texts_creates_and_reuses_cache(tmp_path):
    cache_path = tmp_path / "nested" / "embeddings.json"
    expected_vectors = [[0.1] * 768]
    client = Mock(spec=["models"])

    with patch(
        "llmrag.embeddings.embed_texts_batched",
        autospec=True,
        return_value=expected_vectors,
    ) as embed:
        first = load_or_embed_texts(
            ["Text A"],
            cache_path=cache_path,
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=2,
        )

        second = load_or_embed_texts(
            ["Text A"],
            cache_path=cache_path,
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=2,
        )

        assert first == expected_vectors
        assert second == expected_vectors
        embed.assert_called_once_with(
            ["Text A"],
            client=client,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=2,
        )

    saved = json.loads(cache_path.read_text(encoding="utf-8"))

    assert saved == {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
        "texts": ["Text A"],
        "vectors": expected_vectors,
    }
