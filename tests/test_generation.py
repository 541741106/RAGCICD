from unittest.mock import Mock

import pytest

from llmrag.generation import generate_answer


def test_generate_answer_returns_stripped_text():
    client = Mock()
    client.models.generate_content.return_value.text = "  Answer [1].\n"

    answer = generate_answer(
        "Evidence and question",
        client=client,
        model="test-model",
    )

    assert answer == "Answer [1]."
    client.models.generate_content.assert_called_once_with(
        model="test-model",
        contents="Evidence and question",
    )


def test_empty_prompt():
    client = Mock()

    with pytest.raises(ValueError):
        generate_answer(
            " \n",
            client=client,
            model="test-model",
        )

    client.models.generate_content.assert_not_called()


@pytest.mark.parametrize("response_text", [None, "", " \n "])
def test_generate_answer_rejects_empty_response(response_text):
    client = Mock()
    client.models.generate_content.return_value.text = response_text

    with pytest.raises(RuntimeError, match="model returned no answer text"):
        generate_answer("A valid prompt", client=client)

    client.models.generate_content.assert_called_once()


def test_generate_answer_error_api_response():
    client = Mock()
    client.models.generate_content.side_effect = ConnectionError("network unavailable")

    with pytest.raises(ConnectionError, match="network unavailable"):
        generate_answer("A valid prompt", client=client)

    client.models.generate_content.assert_called_once()
