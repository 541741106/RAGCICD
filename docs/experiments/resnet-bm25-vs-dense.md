# BM25 vs Dense Retrieval: ResNet Pilot

## Experimental Setup

Both methods used the same ResNet PDF, 75 page-preserving chunks,
five development questions, and top_k=3.

- Chunk size: 1000 Python characters
- Overlap: 150 characters
- BM25: Haystack 3.3.0 in-memory retriever
- Dense model: gemini-embedding-001
- Embedding dimensions: 768
- Document task: RETRIEVAL_DOCUMENT
- Query task: RETRIEVAL_QUERY
- Similarity: cosine

## Human-Reviewed Evidence Sufficiency


| Question | BM25           | Dense          |
| -------- | -------------- | -------------- |
| 001      | sufficient     | sufficient     |
| 002      | insufficient   | sufficient     |
| 003      | partial        | sufficient     |
| 004      | sufficient     | sufficient     |
| 005      | not applicable | not applicable |

Among four answerable questions, top-3 evidence sufficiency increased
from 2/4 (50%) to 4/4 (100%), a difference of 50 percentage points.

Question 002 retrieved the degradation characteristics rather than
only solution discussions. Question 003 retrieved explicit ImageNet
depth statements and architecture information.

## Interpretation and Limitations

The dense run met the predefined pilot target. This does not establish
general superiority over BM25.

- One paper and only four answerable development questions.
- Questions 001 and 004 share a knowledge group.
- Manual review without independent reviewer agreement measurement.
- No held-out test-set evaluation.
- No answer generation or refusal evaluation.
- Latency, actual API token usage, and billing cost were not recorded.
- Existing PDF extraction noise was retained.

## Artifacts

- [Dataset](../../data/evaluation/resnet-dev.json)
- [BM25 results](resnet-bm25-dev-run-01.json)
- [BM25 review](resnet-bm25-dev-review-01.json)
- [Dense results](resnet-dense-dev-run-01.json)
- [Dense review](resnet-dense-dev-review-01.json)
