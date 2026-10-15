from haystack import Document

from llmrag.prompting import build_evidence_prompt


def test_prompt_preserves_question_and_evidence():
    documents = [
        Document(
            content="First evidence text.",
            meta={"paper_title": "Paper A", "page_number": 3},
        ),
        Document(
            content="Second evidence text.",
            meta={"paper_title": "Paper B", "page_number": 5},
        ),
    ]
    question = "How do the methods differ?"

    prompt = build_evidence_prompt(question, documents)

    assert f"Question: {question}" in prompt

    first_block = (
        "[1]\nPaper: Paper A\nPDF page index: 3\nContent: First evidence text."
    )
    second_block = (
        "[2]\nPaper: Paper B\nPDF page index: 5\nContent: Second evidence text."
    )

    assert first_block in prompt
    assert second_block in prompt

    assert prompt.index(first_block) < prompt.index(second_block)


def test_prompt_contains_answering_rules():
    prompt = build_evidence_prompt("What is the method?", [])

    assert "Answer using only the provided evidence." in prompt
    assert "Cite the provided evidence IDs for each factual claim." in prompt
    assert "Insufficient evidence was found to answer the question." in prompt


def test_prompt_handles_empty_evidence():
    question = "What is the method?"

    prompt = build_evidence_prompt(question, [])

    assert f"Question: {question}" in prompt
    assert "Evidence:" in prompt
    assert "[1]" not in prompt
