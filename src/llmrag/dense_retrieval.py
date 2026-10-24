from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def build_dense_retriever(
    chunks: list[Document],
    vectors: list[list[float]],
) -> InMemoryEmbeddingRetriever:
    """Build a cosine retriever from chunks and aligned embeddings."""
    if len(chunks) != len(vectors):
        raise ValueError("chunk count must match vector count")

    documents: list[Document] = []

    for chunk, vector in zip(chunks, vectors):
        documents.append(
            Document(
                content=chunk.content,
                meta=chunk.meta.copy(),
                embedding=vector,
            )
        )

    store = InMemoryDocumentStore(
        embedding_similarity_function="cosine",
    )

    store.write_documents(documents)

    return InMemoryEmbeddingRetriever(document_store=store)
