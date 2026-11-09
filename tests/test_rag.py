from unittest.mock import Mock, patch

import pytest
from haystack import Document

from llmrag.rag import answer_question, answer_question_with_pgvector


def test_answer_question_connects_retrieval_prompt_and_generation():
    client = Mock()
    client.models.generate_content.return_value.text = "A semantic label [1]."

    result = answer_question(
        "What does image segmentation assign to each pixel?",
        client=client,
        model="test-model",
    )

    assert result.answer == "A semantic label [1]."
    assert result.documents[0].meta["paper_id"] == "example-segmentation"

    client.models.generate_content.assert_called_once()
    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "test-model"

    sent_prompt = call.kwargs["contents"]
    assert "What does image segmentation assign to each pixel?" in sent_prompt
    assert "Synthetic Segmentation Paper" in sent_prompt
    assert result.documents[0].content in sent_prompt


def test_answer_question_empty_question_raises_value_error():
    client = Mock()

    with pytest.raises(ValueError, match="question must not be empty"):
        answer_question(" \n ", client=client)

    client.models.generate_content.assert_not_called()


def test_answer_question_abstains_when_retrieval_is_empty():
    client = Mock()

    with patch("llmrag.rag.retrieve_examples", return_value=[]) as retrieve:
        result = answer_question(
            "A question without evidence",
            client=client,
            top_k=3,
        )

    retrieve.assert_called_once_with(
        "A question without evidence",
        top_k=3,
    )
    assert result.answer == "Insufficient evidence was found to answer the question."
    assert result.documents == []
    client.models.generate_content.assert_not_called()


def test_answer_question_uses_supplied_retriever():
    client = Mock()
    client.models.generate_content.return_value.text = "An answer [1]."

    document = Document(
        content="Evidence from a supplied collection.",
        meta={
            "paper_id": "custom-paper",
            "paper_title": "Custom Paper",
            "page_number": 3,
            "source": "local-test",
        },
    )
    retriever = Mock()
    retriever.run.return_value = {"documents": [document]}

    result = answer_question(
        "A question",
        client=client,
        retriever=retriever,
        top_k=3,
    )

    retriever.run.assert_called_once_with(query="A question", top_k=3)
    assert result.answer == "An answer [1]."
    assert result.documents == [document]
    assert (
        document.content in client.models.generate_content.call_args.kwargs["contents"]
    )


def test_pgvector_answer_connects_embedding_retrieval_and_generation():
    client = Mock()
    client.models.generate_content.return_value.text = "An answer [1]."

    query_vector = [0.1] * 768
    document = Document(
        content="Identity shortcuts pass the input unchanged.",
        meta={
            "paper_id": "example-paper",
            "paper_title": "Example Paper",
            "page_number": 2,
            "chunk_number": 1,
            "source": "local-test",
        },
    )
    retriever = Mock(spec=["run"])
    retriever.run.return_value = {"documents": [document]}

    with patch(
        "llmrag.rag.embed_texts",
        autospec=True,
        return_value=[query_vector],
    ) as embed:
        result = answer_question_with_pgvector(
            "  What are identity shortcuts?  ",
            client=client,
            retriever=retriever,
            top_k=2,
            model="test-model",
        )

        embed.assert_called_once_with(
            ["What are identity shortcuts?"], client=client, task_type="RETRIEVAL_QUERY"
        )

    retriever.run.assert_called_once_with(query_embedding=query_vector, top_k=2)

    assert result.answer == "An answer [1]."
    assert result.documents == [document]

    client.models.generate_content.assert_called_once()
    arguments = client.models.generate_content.call_args.kwargs

    assert arguments["model"] == "test-model"
    assert "What are identity shortcuts?" in arguments["contents"]
    assert document.content in arguments["contents"]


def test_pgvector_answer_skips_generation_when_retrieval_is_empty():
    client = Mock()
    query_vector = [0.1] * 768

    retriever = Mock(spec=["run"])
    retriever.run.return_value = {"documents": []}

    with patch(
        "llmrag.rag.embed_texts",
        autospec=True,
        return_value=[query_vector],
    ) as embed:
        result = answer_question_with_pgvector(
            "A question without evidence",
            client=client,
            retriever=retriever,
        )

        embed.assert_called_once()

    retriever.run.assert_called_once_with(query_embedding=query_vector, top_k=3)

    assert result.answer == "Insufficient evidence was found to answer the question."

    assert not result.documents

    client.models.generate_content.assert_not_called()


@pytest.mark.parametrize(
    ("question", "top_k", "message"),
    [
        (" \n\t", 3, "question must not be empty"),
        ("A valid question", 0, "top_k must be positive"),
        ("A valid question", -1, "top_k must be positive"),
    ],
)
def test_pgvector_answer_rejects_invalid_input_before_external_calls(
    question, top_k, message
):
    client = Mock()
    retriever = Mock(spec=["run"])

    with patch("llmrag.rag.embed_texts", autospec=True) as embed:
        with pytest.raises(ValueError, match=message):
            answer_question_with_pgvector(
                question, client=client, retriever=retriever, top_k=top_k
            )

        embed.assert_not_called()

    retriever.run.assert_not_called()
    client.models.generate_content.assert_not_called()
