from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class AskRequest(BaseModel):
    """Validated input for a question-answering request."""

    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
        ),
    ]

    top_k: int = Field(default=3, ge=1, le=10)


class EvidenceResponse(BaseModel):
    """One retrieved evidence item exposed by the API."""

    evidence_id: int = Field(ge=1)
    paper_id: str
    paper_title: str
    page_number: int = Field(ge=1)
    content: str


class AskResponse(BaseModel):
    """Answer and retrieved evidence returned to the caller."""

    answer: str
    evidence: list[EvidenceResponse]
