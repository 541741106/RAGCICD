from pathlib import Path

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pypdf import PdfReader

from llmrag.chunking import chunk_text


def load_pdf_pages(
    path: str | Path,
    *,
    paper_id: str,
    paper_title: str,
    source: str,
) -> list[Document]:
    """Extract raw page text while preserving PDF page indices."""
    documents: list[Document] = []

    with Path(path).open("rb") as pdf_file:
        reader = PdfReader(pdf_file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    content=text,
                    meta={
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "page_number": page_number,
                        "source": source,
                    },
                )
            )

    return documents


def chunk_pages(
    pages: list[Document],
    *,
    chunk_size: int,
    overlap: int = 0,
) -> list[Document]:
    """Split pages into chunks while preserving source metadata."""
    chunk_text("", chunk_size=chunk_size, overlap=overlap)

    chunks: list[Document] = []

    for page in pages:
        texts = chunk_text(
            page.content or "",
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, text in enumerate(texts, start=1):
            metadata = page.meta.copy()

            metadata["chunk_number"] = chunk_number

            chunks.append(
                Document(
                    content=text,
                    meta=metadata,
                )
            )

    return chunks


def build_pdf_retriever(
    path: str | Path,
    *,
    paper_id: str,
    paper_title: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> InMemoryBM25Retriever:
    """Build an in-memory BM25 retriever from one PDF."""
    pages = load_pdf_pages(
        path,
        paper_id=paper_id,
        paper_title=paper_title,
        source=source,
    )

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    store = InMemoryDocumentStore()

    store.write_documents(chunks)

    return InMemoryBM25Retriever(document_store=store)
