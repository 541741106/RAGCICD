from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def build_example_documents() -> list[Document]:
    """Build synthetic documents for learning retrieval."""
    return [
        Document(
            content=(
                "This synthetic example describes an object detector. "
                "The detector predicts bounding boxes and object classes."
            ),
            meta={
                "paper_id": "example-detection",
                "paper_title": "Synthetic Detection Paper",
                "page_number": 1,
                "source": "synthetic",
            },
        ),
        Document(
            content=(
                "This synthetic example describes image classification. "
                "The model assigns a class label to an input image."
            ),
            meta={
                "paper_id": "example-classification",
                "paper_title": "Synthetic Classification Paper",
                "page_number": 1,
                "source": "synthetic",
            },
        ),
        Document(
            content=(
                "This synthetic example describes image segmentation. "
                "The model assigns a semantic label to each pixel in an image."
            ),
            meta={
                "paper_id": "example-segmentation",
                "paper_title": "Synthetic Segmentation Paper",
                "page_number": 1,
                "source": "synthetic",
            },
        ),
    ]


def build_example_store() -> InMemoryDocumentStore:
    """Create an in-memory store containing the synthetic documents."""
    store = InMemoryDocumentStore()
    documents = build_example_documents()

    store.write_documents(documents)

    return store


def retrieve_examples(query: str, top_k: int = 2) -> list[Document]:
    """Retrieve synthetic documents using BM25."""
    store = build_example_store()
    retriever = InMemoryBM25Retriever(document_store=store)

    result = retriever.run(query=query, top_k=top_k)
    return result["documents"]
