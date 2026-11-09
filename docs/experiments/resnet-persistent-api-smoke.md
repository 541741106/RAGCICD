# Persistent ResNet HTTP Question-Answering Smoke Test

## Scope

On 2026-11-09, one manual `POST /ask` request was exercised through the local
interactive API documentation after importing 75 ResNet chunks into PostgreSQL.
The request used the question "What are identity shortcut connections?" and
`top_k=3`.

The path uses `gemini-embedding-001` with 768-dimensional query vectors,
cosine exact-neighbor retrieval from `resnet_gemini_768_v1`, and the configured
default generation model, `gemini-3.5-flash-lite`.

## Observed Response

The response contained an answer and three evidence items from
`arxiv:1512.03385v1`, on PDF pages 2, 2, and 6. The answer described identity
mapping and addition to the stacked-layer output with citation [2], followed
by the parameter-free and computational-cost statements with citations [1, 2].

Manual review found that evidence [2] supports identity mapping, addition,
and the absence of extra shortcut parameters or computational complexity.
Evidence [1] supports the parameter-free statement and comparison with gated
shortcuts, but does not independently establish the computational-cost claim.
The combined citations support the sentence; more precise claim-level citation
placement would make the attribution clearer. Evidence [3] was not cited.

The response was inspected manually; this record does not contain a captured
HTTP status code, latency, token usage, or billing cost. No additional model
request was made to write this record.

## Limits and Related Artifacts

This is a single end-to-end connectivity and evidence-support check, not an
answer-accuracy benchmark. Refusal behavior, concurrency, database outages,
and provider network failures were not evaluated. Citation support was checked
by a human, not automatically.

- [Earlier direct-function QA smoke result](resnet-pgvector-qa-smoke-01.json)
- [Storage and retrieval consistency checks](resnet-pgvector-storage.md)
- [API setup](../../README.md#run-the-http-api)
