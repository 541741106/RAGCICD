from unittest.mock import Mock, patch

import pytest
from haystack import Document
from pypdf import PdfWriter

from llmrag.ingestion import chunk_pages, load_pdf_pages


def test_load_pdf_pages_preserves_empty_pages_and_metadata(tmp_path):
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"placeholder")

    pages = [Mock(), Mock(), Mock()]
    pages[0].extract_text.return_value = "Page one"
    pages[1].extract_text.return_value = None
    pages[2].extract_text.return_value = "Page three"

    with patch("llmrag.ingestion.PdfReader") as reader:
        reader.return_value.pages = pages

        documents = load_pdf_pages(
            pdf_path,
            paper_id="example-paper",
            paper_title="Example Paper",
            source="https://example.org/paper",
        )

    assert len(documents) == 3

    assert documents[0].content == "Page one"
    assert documents[1].content == ""
    assert documents[2].content == "Page three"

    assert documents[0].meta["page_number"] == 1
    assert documents[1].meta["page_number"] == 2
    assert documents[2].meta["page_number"] == 3

    for doc in documents:
        assert doc.meta["paper_id"] == "example-paper"
        assert doc.meta["paper_title"] == "Example Paper"
        assert doc.meta["source"] == "https://example.org/paper"


def test_load_pdf_pages_reads_real_blank_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    documents = load_pdf_pages(
        pdf_path,
        paper_id="blank-paper",
        paper_title="Blank Paper",
        source="local-test",
    )

    assert len(documents) == 1

    assert documents[0].content == ""

    assert documents[0].meta["paper_id"] == "blank-paper"
    assert documents[0].meta["paper_title"] == "Blank Paper"
    assert documents[0].meta["source"] == "local-test"
    assert documents[0].meta["page_number"] == 1


def test_chunk_pages_preserves_page_numbers_and_metadata():
    pages = [
        Document(
            content=text,
            meta={
                "paper_id": "example-paper",
                "paper_title": "Example Paper",
                "page_number": page_number,
                "source": "local-test",
            },
        )
        for page_number, text in enumerate(["abcdef", "", "xyz"], start=1)
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    actual = [
        (
            chunk.content,
            chunk.meta["page_number"],
            chunk.meta["chunk_number"],
        )
        for chunk in chunks
    ]

    assert actual == [
        ("abcd", 1, 1),
        ("def", 1, 2),
        ("xyz", 3, 1),
    ]

    for chunk in chunks:
        assert chunk.meta["paper_id"] == "example-paper"
        assert chunk.meta["paper_title"] == "Example Paper"
        assert chunk.meta["source"] == "local-test"
        assert chunk.meta["page_number"] in {1, 3}

    for page in pages:
        assert "chunk_number" not in page.meta


def test_chunk_pages_handles_empty_pages():
    pages = [
        Document(
            content="",
            meta={
                "paper_id": "example-paper",
                "paper_title": "Example Paper",
                "page_number": 1,
                "source": "local-test",
            },
        )
    ]
    chunks = chunk_pages(pages, chunk_size=4, overlap=1)
    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (4, -1),
        (4, 4),
        (4, 5),
    ],
)
def test_chunk_pages_validates_parameters_without_pages(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)


def test_chunk_pages_handles_empty_list():
    assert chunk_pages([], chunk_size=4, overlap=1) == []
