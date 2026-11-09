# Containerized ResNet API Smoke Test

## Configuration

- Local Docker Desktop Linux engine; API built as `papertrace-api:dev`.
- Base image: `python:3.11-slim-bookworm`; runtime user: UID 10001.
- Database image: `pgvector/pgvector:0.8.7-pg17-bookworm`.
- Compose API port: `127.0.0.1:8001` to container port 8000.
- API database address: `db:5432`; existing volume: `papertrace_postgres_data`.
- Table: `resnet_gemini_768_v1`, with 75 previously imported chunks.
- Query embeddings: `gemini-embedding-001`, 768 dimensions.
- Generation model: configured default `gemini-3.5-flash-lite`.

## Manual Observations

1. The image built successfully, including installation of runtime dependencies.
2. A standalone temporary API container started without database or model
   configuration. `GET /health` returned `{"status":"OK"}`. The temporary
   container was stopped and automatically removed.
3. The first Compose database query failed because the API connection used
   `127.0.0.1:5433`. Changing the API environment to `db:5432` resolved the
   connection issue. A query executed inside the API container returned 75
   stored documents from the existing database.
4. On 2026-11-09, a request through the container API's interactive docs returned
   HTTP 200 for "What are identity shortcut connections?" with `top_k=3`.
   It returned three ResNet evidence items on PDF pages 2, 2, and 6.

The answer cited [2] for identity mapping, addition to stacked-layer outputs,
and no extra parameters or computational complexity. It cited [1, 3] for the
parameter-free statement. Manual inspection of the supplied evidence supports
these claims. This was a real query embedding and generation request; token
usage, billing cost, and latency were not recorded.

## Limits

- Single-question manual integration check, not a quality benchmark.
- The database already contained imported data; this was not an empty-database
  bootstrap test. The API image does not contain or import the corpus.
- Health checks do not establish database query or provider availability.
- Provider failures, concurrency, recovery, and automated deployment were not tested.
- Local containers are running; image publishing and CD remain future work.
