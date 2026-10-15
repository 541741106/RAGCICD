from unittest.mock import Mock, patch

import pytest

from llmrag.rag import answer_question


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
