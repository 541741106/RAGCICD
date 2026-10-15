from google import genai


def generate_answer(
    prompt: str,
    *,
    client: genai.Client,
    model: str = "gemini-3.5-flash-lite",
) -> str:
    """Generate answer text using a supplied Gemini client."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    text = response.text
    if text is None or not text.strip():
        raise RuntimeError("model returned no answer text")

    return text.strip()
