import argparse

from google import genai

from llmrag.ingestion import build_pdf_retriever
from llmrag.rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about a paper PDF or the synthetic CV collection."
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--pdf", help="Path to a text-based paper PDF")
    parser.add_argument("--paper-id", help="Paper identifier")
    parser.add_argument("--paper-title", help="Paper title")
    parser.add_argument("--source", help="Paper source URL")
    args = parser.parse_args()
    if args.pdf and not all([args.paper_id, args.paper_title, args.source]):
        parser.error("Missing required arguments for PDF processing.")

    retriever = None

    if args.pdf:
        retriever = build_pdf_retriever(
            path=args.pdf,
            paper_id=args.paper_id,
            paper_title=args.paper_title,
            source=args.source,
        )

    with genai.Client() as client:
        result = answer_question(
            args.question,
            client=client,
            retriever=retriever,
            top_k=args.top_k,
        )

    print("Answer:", result.answer)

    for evidence_id, document in enumerate(result.documents, start=1):
        print(
            f"[{evidence_id}] {document.meta['paper_title']} "
            f"(PDF page index: {document.meta['page_number']})"
        )
        print(f"Content: {document.content}\n")


if __name__ == "__main__":
    main()
