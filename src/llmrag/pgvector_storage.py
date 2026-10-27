from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


def write_embedded_chunks(
    chunks: list[Document],
    vectors: list[list[float]],
    *,
    store: PgvectorDocumentStore,
) -> int:
    """Write aligned chunks and embeddings without modifying the originals."""
    if len(chunks) != len(vectors):
        raise ValueError("chunk count must match vector count")

    documents: list[Document] = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        documents.append(
            Document(
                id=chunk.id,
                content=chunk.content,
                meta=chunk.meta.copy(),
                embedding=vector,
            )
        )

    written = store.write_documents(
        documents,  # list[Document]
        policy=DuplicatePolicy.OVERWRITE,
    )

    return written
