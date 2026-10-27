import json
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document

from llmrag.ingestion import chunk_pages, load_pdf_pages
from llmrag.pgvector_storage import write_embedded_chunks
from llmrag.storage import open_document_store


def load_cached_document_vectors(
    chunks: list[Document],
    *,
    cache_path: str | Path,
) -> list[list[float]]:
    """Load existing document embeddings only when inputs match."""
    cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    if not isinstance(cached, dict):
        raise TypeError("embedding cache must be a JSON object")

    expected = {
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
        "texts": [chunk.content for chunk in chunks],
    }

    for key, value in expected.items():
        if value != cached.get(key):
            raise ValueError("embedding cache does not match inputs")

    vectors = cached.get("vectors")

    if not isinstance(vectors, list):
        raise TypeError("cached vectors must be a list")

    if len(vectors) != len(chunks):
        raise ValueError("cached vector count does not match chunk count")

    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != 768:
            raise ValueError("cached vector must have 768 dimensions")
        for value in vector:
            if not type(value) in (int, float) or not isfinite(value):
                raise ValueError("cached vector values must be finite numbers")

    return vectors


def main() -> None:
    """Import the ResNet chunks using existing embedding caches."""
    load_dotenv(".env", override=False)

    pages = load_pdf_pages(
        "data/raw/resnet-1512.03385v1.pdf",
        paper_id="arxiv:1512.03385v1",
        paper_title="Deep Residual Learning for Image Recognition",
        source="https://arxiv.org/abs/1512.03385v1",
    )

    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    vectors = load_cached_document_vectors(
        chunks, cache_path="data/index/resnet-document-embeddings.json"
    )

    store = open_document_store()

    written = write_embedded_chunks(chunks, vectors, store=store)

    print("Written:", written)
    print("Stored documents:", store.count_documents())


if __name__ == "__main__":
    main()
