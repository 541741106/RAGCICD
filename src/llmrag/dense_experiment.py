import argparse
import json
from importlib.metadata import version
from pathlib import Path

from google import genai
from haystack import Document

from llmrag.dense_retrieval import build_dense_retriever
from llmrag.embeddings import load_or_embed_texts
from llmrag.evaluation import collect_dense_retrieval_results
from llmrag.ingestion import chunk_pages, load_pdf_pages


def run_dense_experiment(
    chunks: list[Document],
    cases: list[dict],
    *,
    client: genai.Client,
    document_cache: str | Path,
    query_cache: str | Path,
    top_k: int = 3,
) -> list[dict]:
    """Run dense retrieval using cached document and query embeddings."""
    document_texts = [chunk.content for chunk in chunks]
    query_texts = [case["question"] for case in cases]

    document_vectors = load_or_embed_texts(
        texts=document_texts,
        cache_path=document_cache,
        client=client,
        task_type="RETRIEVAL_DOCUMENT",
    )

    query_vectors = load_or_embed_texts(
        texts=query_texts,
        cache_path=query_cache,
        client=client,
        task_type="RETRIEVAL_QUERY",
    )

    retriever = build_dense_retriever(chunks, document_vectors)

    results = collect_dense_retrieval_results(
        cases, query_vectors, retriever=retriever, top_k=top_k
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the ResNet dense retrieval experiment using caches."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    document_cache = Path("data/index/resnet-document-embeddings.json")
    query_cache = Path("data/index/resnet-query-embeddings.json")
    output = Path(args.output)

    if not document_cache.is_file() or not query_cache.is_file():
        parser.error("Both embedding cache files must already exist.")
    if output.exists():
        parser.error("Output already exists; choose a new path.")

    cases = json.loads(
        Path("data/evaluation/resnet-dev.json").read_text(encoding="utf-8")
    )
    pages = load_pdf_pages(
        "data/raw/resnet-1512.03385v1.pdf",
        paper_id="arxiv:1512.03385v1",
        paper_title="Deep Residual Learning for Image Recognition",
        source="https://arxiv.org/abs/1512.03385v1",
    )
    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    with genai.Client() as client:
        results = run_dense_experiment(
            chunks,
            cases,
            client=client,
            document_cache=document_cache,
            query_cache=query_cache,
        )

    report = {
        "dataset": "data/evaluation/resnet-dev.json",
        "paper_id": "arxiv:1512.03385v1",
        "haystack_version": version("haystack-ai"),
        "google_genai_version": version("google-genai"),
        "retriever": "InMemoryEmbeddingRetriever",
        "embedding_model": "gemini-embedding-001",
        "dimensions": 768,
        "document_task_type": "RETRIEVAL_DOCUMENT",
        "query_task_type": "RETRIEVAL_QUERY",
        "similarity": "cosine",
        "chunk_size": 1000,
        "overlap": 150,
        "top_k": 3,
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        file.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()
