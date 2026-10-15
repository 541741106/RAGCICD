from haystack import Document


def build_evidence_prompt(
    question: str,
    documents: list[Document],
) -> str:
    """Build a prompt from a question and retrieved evidence."""
    evidence_blocks: list[str] = []

    for evidence_id, document in enumerate(documents, start=1):
        paper_title = document.meta["paper_title"]
        page_number = document.meta["page_number"]
        content = document.content
        evidence_block = (
            f"[{evidence_id}]\n"
            f"Paper: {paper_title}\n"
            f"PDF page index: {page_number}\n"
            f"Content: {content}"
        )
        evidence_blocks.append(evidence_block)

    evidence_text = "\n\n".join(evidence_blocks)

    prompt = (
        "You are a Computer Vision research assistant.\n\n"
        "Rules:\n"
        "1. Answer using only the provided evidence.\n"
        "2. Cite the provided evidence IDs for each factual claim.\n"
        "3. Do not fabricate information.\n"
        "4. If the evidence is insufficient, explicitly say "
        "'Insufficient evidence was found to answer the question.'\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    return prompt
