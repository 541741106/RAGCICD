def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[str]:
    """Split text into character chunks while preserving whitespace."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break

    return chunks
