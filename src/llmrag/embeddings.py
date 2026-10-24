import json
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types


def embed_texts(
    texts: list[str],
    *,
    client: genai.Client,
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"],
) -> list[list[float]]:
    """Embed texts in input order for the retrieval experiment."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain blank strings")

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )

    if response.embeddings is None:
        raise RuntimeError("embedding response contains no embeddings")

    vectors: list[list[float]] = []

    for embedding in response.embeddings:
        if embedding.values is None:
            raise RuntimeError("embedding response contains no vector values")
        vectors.append(embedding.values)

    if not len(vectors) == len(texts):
        raise RuntimeError("embedding count does not match text count")

    for vector in vectors:
        if len(vector) != 768:
            raise RuntimeError("embedding vector must have 768 dimensions")

    return vectors


def embed_texts_batched(
    texts: list[str],
    *,
    client: genai.Client,
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"],
    batch_size: int = 10,
) -> list[list[float]]:
    """Embed texts in sequential batches while preserving input order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain blank strings")

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        batch_vectors = embed_texts(batch, client=client, task_type=task_type)

        vectors.extend(batch_vectors)

    return vectors


def load_or_embed_texts(
    texts: list[str],
    *,
    cache_path: str | Path,
    client: genai.Client,
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"],
    batch_size: int = 10,
) -> list[list[float]]:
    """Reuse embeddings only when cached inputs and configuration match."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain blank strings")

    path = Path(cache_path)

    expected = {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": task_type,
        "texts": texts,
    }

    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(cached, dict):
            raise ValueError("embedding cache must be a JSON object")

        for key, value in expected.items():
            if cached.get(key) != value:
                raise ValueError("embedding cache does not match inputs")

        vectors = cached.get("vectors")

        if not isinstance(vectors, list):
            raise ValueError("cached vectors must be a list")

        if len(vectors) != len(texts):
            raise ValueError("cached vector count does not match text count")

        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != 768:
                raise ValueError("cached vector must have 768 dimensions")

        return vectors

    vectors = embed_texts_batched(
        texts,
        client=client,
        task_type=task_type,
        batch_size=batch_size,
    )

    cached = {
        **expected,
        "vectors": vectors,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(cached, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)

    return vectors
