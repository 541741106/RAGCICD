# ResNet pgvector Storage Verification

## Scope

- Corpus: `arxiv:1512.03385v1`, 12 PDF pages and 75 chunks.
- Chunking: 1000 Python characters, overlap 150.
- Existing embeddings: `gemini-embedding-001`, 768 dimensions.
- Task types: `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`.
- Storage: PostgreSQL 17, pgvector 0.8.7, `pgvector-haystack==8.1.2`.
- Table: `resnet_gemini_768_v1`.
- Search: cosine similarity, exact nearest neighbor, top-k 3.
- Dataset: five development questions across four question groups.
- No embedding or answer-generation API calls were made during this check.

## Observed Checks

A temporary document was written, read through a new document-store object,
and checked for matching ID, text, metadata, and embedding before deletion.

The existing 75 chunks and cached embeddings were imported. Repeating the
import reported 75 written documents and 75 total stored documents. A new
Python session loaded configuration from `.env` and read the same count.
These observations do not constitute a machine-reboot test or a backup test.

All five questions were queried using their cached query embeddings.
Compared with the in-memory dense baseline, the ordered top-3 evidence
matched for all five questions. Comparison included rank, paper ID, PDF page
index, chunk number, and content. Scores were excluded from equality checks.

## Repeatable Import

Follow the database setup and local cache prerequisites in the
[README](../../README.md#persistent-storage-with-postgresql-and-pgvector), then run:

```bat
python -m llmrag.pgvector_ingestion
```

The import validates cached inputs before database access and overwrites
matching IDs. The query and baseline comparison were performed manually;
there is currently no dedicated command to repeat that comparison.

## Limitations

- One paper and a small development set; questions 001 and 004 share a group.
- Equality with the baseline measures migration consistency, not answer accuracy
  or general retrieval quality. No generation or refusal behavior was evaluated.
- The fixed table and cache configuration do not support general index versioning.
- Reimporting does not delete stale IDs after corpus or chunking changes.
- Persistent retrieval is not yet connected to the CLI or HTTP QA paths.
- Unit tests use temporary cache files and mocks, not a live PostgreSQL service.

## Artifacts

- [Dataset](../../data/evaluation/resnet-dev.json)
- [In-memory dense baseline](resnet-dense-dev-run-01.json)
- [pgvector results and comparison flags](resnet-pgvector-dev-run-01.json)
