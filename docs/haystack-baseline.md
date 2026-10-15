# Haystack Baseline

## Selected Version

- Package: haystack-ai
- Version: 3.3.0
- Python environment: 3.11.17
- Verification date: 2026-10-13

The version is declared in pyproject.toml.
Package import and pip check passed locally.
Pipeline behavior has not yet been verified.

## Official References

- Package metadata:
  https://pypi.org/project/haystack-ai/3.3.0/
- Getting started:
  https://haystack.deepset.ai/overview/quick-start
- First RAG pipeline tutorial:
  https://haystack.deepset.ai/tutorials/27_first_rag_pipeline
- Version 3.0 breaking changes:
  https://haystack.deepset.ai/release-notes/3.0.0

These web pages may change. The links do not imply that every example
has been verified with the selected version.

## Planned Adaptations

- Begin with three short, manually written example documents.
- Use an in-memory document store and BM25 retrieval first.
- Add prompt construction and cloud generation after retrieval works.
- Introduce embedding retrieval later and compare it against BM25.
- Verify component imports and behavior with haystack-ai 3.3.0
  before adopting tutorial code.

The BM25 baseline is a teaching adaptation of the RAG workflow,
not a verbatim reproduction of the embedding-based tutorial.

## Verified BM25 Baseline (2026-10-15)

- Component imports succeeded with haystack-ai 3.3.0.
- Three synthetic documents were written to an in-memory store.
- The query "image segmentation" ranked the segmentation document first.
- The query "bounding boxes" ranked the detection document first.
- Retrieved source metadata was preserved.
- All 15 project tests, Ruff lint, and formatting checks passed locally.
- No LLM API calls or model-weight downloads were required.

These checks validate the synthetic examples only.
They do not establish retrieval quality on real papers.
An end-to-end generation pipeline has not yet been verified.

### Run the retrieval example

From the repository root with the project installed:

```powershell
python -c "from llmrag.bm25_baseline import retrieve_examples; docs = retrieve_examples('image segmentation', top_k=2); print([(d.meta, d.score, d.content) for d in docs])"
```