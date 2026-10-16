from unittest.mock import patch

import pytest

from llmrag.__main__ import main
from llmrag.rag import RagResult


def test_cli_connects_pdf_retriever(capsys):
    argv = [
        "llmrag",
        "What are identity shortcuts?",
        "--pdf",
        "example.pdf",
        "--paper-id",
        "example-paper",
        "--paper-title",
        "Example Paper",
        "--source",
        "local-test",
        "--top-k",
        "3",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "llmrag.__main__.build_pdf_retriever",
            autospec=True,
        ) as build,
        patch("llmrag.__main__.genai.Client") as client_class,
        patch("llmrag.__main__.answer_question") as answer,
    ):
        client = client_class.return_value.__enter__.return_value
        answer.return_value = RagResult(answer="An answer.", documents=[])

        main()

        build.assert_called_once_with(
            path="example.pdf",
            paper_id="example-paper",
            paper_title="Example Paper",
            source="local-test",
        )

        answer.assert_called_once_with(
            "What are identity shortcuts?",
            client=client,
            retriever=build.return_value,
            top_k=3,
        )

    assert "Answer: An answer." in capsys.readouterr().out


def test_cli_rejects_pdf_without_metadata(capsys):
    with (
        patch("sys.argv", ["llmrag", "A question", "--pdf", "example.pdf"]),
        patch("llmrag.__main__.build_pdf_retriever") as build,
        patch("llmrag.__main__.genai.Client") as client_class,
    ):
        with pytest.raises(SystemExit) as error:
            main()

        assert error.value.code == 2
        build.assert_not_called()
        client_class.assert_not_called()

    assert "Missing required arguments" in capsys.readouterr().err
