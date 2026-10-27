from unittest.mock import Mock

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from llmrag.pgvector_storage import write_embedded_chunks


def test_write_embedded_chunks_preserves_sources_and_originals():
    chunk = Document(
        id="chunk-001",
        content="Paper text.",
        meta={"paper_id": "example-paper", "page_number": 2},
    )
    vector = [0.1] * 768

    store = Mock(spec=["write_documents"])
    store.write_documents.return_value = 1

    written = write_embedded_chunks(
        [chunk],
        [vector],
        store=store,
    )

    assert written == 1
    store.write_documents.assert_called_once()

    arguments = store.write_documents.call_args
    saved_documents = arguments.args[0]
    saved = saved_documents[0]

    assert len(saved_documents) == 1
    assert saved.id == chunk.id
    assert saved.content == chunk.content
    assert saved.meta == chunk.meta
    assert saved.embedding == vector
    assert not saved is chunk
    assert not saved.meta is chunk.meta
    assert chunk.embedding is None
    assert arguments.kwargs["policy"] == DuplicatePolicy.OVERWRITE


def test_write_embedded_chunks_rejects_mismatched_counts():
    chunks = [Document(content="Paper text.")]
    store = Mock(spec=["write_documents"])

    with pytest.raises(
        ValueError,
        match="chunk count must match vector count",
    ):
        write_embedded_chunks(chunks=chunks, vectors=[], store=store)

    store.write_documents.assert_not_called()
