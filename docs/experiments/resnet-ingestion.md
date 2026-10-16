# ResNet PDF Ingestion Baseline

- Date: 2026-10-16
- Paper: Deep Residual Learning for Image Recognition
- Paper ID: arxiv:1512.03385v1
- Source: https://arxiv.org/abs/1512.03385v1
- Parser: pypdf==6.20.1
- Font support: fonttools==4.67.0

## Observed Results

- PDF pages: 12
- Chunks: 75
- Chunk size: 1000 Python characters
- Overlap: 150 Python characters
- The first two chunks share the expected 150-character overlap.
- All chunks are non-empty and contain at most 1000 characters.
- The first two chunks preserve the paper metadata and PDF page index.

## Quality Observations

- Manual inspection found the page 2 body text order broadly correct.
- Ligatures, line-break hyphenation, missing spaces, and figure labels
  remain in the extracted text.
- Character splitting can cut through words and sentences.
- Systematic retrieval and answer quality evaluation has not been performed.

## Manual RAG Check

- Date: 2026-10-16
- Question: What are identity shortcut connections?
- Parser: pypdf==6.20.1
- Model: gemini-3.5-flash-lite
- Retrieval: BM25, top_k=3
- Retrieved locations: (page 2, chunk 2), (page 2, chunk 6),
  (page 4, chunk 3)
- Observation: The response described identity shortcuts and cited
  supporting evidence without generalizing to all shortcut types.
- Limitations: One manual case; no overall quality metrics.
  Token usage and latency were not recorded for this run.

### Reproduction Code

```python
from google import genai

from llmrag.ingestion import build_pdf_retriever
from llmrag.rag import answer_question

retriever = build_pdf_retriever(
    "data/raw/resnet-1512.03385v1.pdf",
    paper_id="arxiv:1512.03385v1",
    paper_title="Deep Residual Learning for Image Recognition",
    source="https://arxiv.org/abs/1512.03385v1",
)

with genai.Client() as client:
    result = answer_question(
        "What are identity shortcut connections?",
        client=client,
        retriever=retriever,
        top_k=3,
    )
    print(result.answer)

for number, document in enumerate(result.documents, start=1):
    print(
        f"[{number}] PDF page {document.meta['page_number']}, "
        f"chunk {document.meta['chunk_number']}"
    )
```

### Actual Model Response

```text
Based on the provided evidence, identity shortcut connections are shortcut connections that simply perform identity mapping, where their outputs are added to the outputs of the stacked layers [1]. They add neither extra parameters nor computational complexity [1]. When the input and output are of the same dimensions, identity shortcuts can be directly used [3]. When dimensions increase, identity shortcuts can still perform identity mapping with extra zero entries padded for the increasing dimensions, introducing no extra parameters [3]. Additionally, identity shortcuts are parameter-free, which contrasts with gated shortcuts in highway networks that have parameters [2].
```

### Retrieved Evidence


| Evidence ID | PDF page index | Chunk number |
| ----------- | -------------: | -----------: |
| [1]         |              2 |            2 |
| [2]         |              2 |            6 |
| [3]         |              4 |            3 |
