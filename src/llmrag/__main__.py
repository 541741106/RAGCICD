import argparse

from google import genai

from llmrag.rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about the synthetic CV document collection."
    )
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    with genai.Client() as client:
        result = answer_question(
            args.question,
            client=client,
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
