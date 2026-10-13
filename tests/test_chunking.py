import pytest

from llmrag.chunking import chunk_text


def test_chunk_text_with_overlap():
    result = chunk_text("abcdefghij", chunk_size=4, overlap=1)

    assert result == ["abcd", "defg", "ghij"]


def test_chunk_text_empty_input():
    result = chunk_text("", chunk_size=4, overlap=1)

    assert result == []


def test_chunk_text_short_input():
    result = chunk_text("abc", chunk_size=4, overlap=1)

    assert result == ["abc"]


def test_chunk_text_trailing_chunk():
    result = chunk_text("abcdef", chunk_size=4, overlap=0)

    assert result == ["abcd", "ef"]


def test_chunk_text_rejects_zero_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)


def test_negative_chunk_size():
    """负数 chunk_size 应抛出 ValueError。"""
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=-1)


def test_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, overlap=-1)


def test_overlap_equals_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, overlap=4)


def test_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, overlap=5)


def test_invalid_chunk_size_with_empty_text():
    with pytest.raises(ValueError):
        chunk_text("", chunk_size=0)


def test_exact_chunk_size():
    result = chunk_text(
        "abcdefgh",
        chunk_size=4,
        overlap=0,
    )

    assert result == ["abcd", "efgh"]


def test_preserve_whitespace():
    result = chunk_text(
        "a b\nc ",
        chunk_size=4,
        overlap=0,
    )

    assert result == ["a b\n", "c "]
