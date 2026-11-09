FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./

COPY src ./src

RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "llmrag.api:app", "--host", "0.0.0.0", "--port", "8000"]