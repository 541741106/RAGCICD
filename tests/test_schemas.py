import pytest
from pydantic import ValidationError

from llmrag.schemas import AskRequest


def test_request_normalizes_question_and_uses_default_top_k():
    request = AskRequest(question="  What is ResNet?  ")

    assert request.question == "What is ResNet?"
    assert request.top_k == 3


@pytest.mark.parametrize(
    ("question", "top_k"),
    [
        ("", 3),
        ("   ", 3),
        ("\n\t", 3),
        ("What is ResNet?", 0),
        ("What is ResNet?", 11),
    ],
)
def test_error_questions_and_top_k(question, top_k):
    with pytest.raises(ValidationError):
        AskRequest(question=question, top_k=top_k)
