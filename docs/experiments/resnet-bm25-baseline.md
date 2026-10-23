# ResNet BM25 Pilot Evaluation

## Scope and Configuration

- Corpus: arxiv:1512.03385v1, 12 PDF pages, 75 chunks.
- Retriever: Haystack 3.3.0 in-memory BM25.
- Chunk size: 1000 Python characters; overlap: 150.
- Retrieval depth: top_k=3.
- Dataset: 5 development questions across 4 question groups.
- No LLM generation was performed.

## Results


| Question | Evidence sufficiency | Main observation                                                           |
| -------- | -------------------- | -------------------------------------------------------------------------- |
| 001      | sufficient           | Direct definition retrieved at rank 1.                                     |
| 002      | insufficient         | Solution discussion retrieved instead of problem characteristics.          |
| 003      | partial              | 152-layer result retrieved, but maximum-depth scope was not established.   |
| 004      | sufficient           | Residual formula and variable definitions support the requested mechanism. |
| 005      | not applicable       | Unanswerable case; generation refusal was not evaluated.                   |

For the four answerable questions, human-reviewed evidence sufficiency
at top-3 was 2/4 (50%). This is not answer accuracy or Recall@3.

## Limitations

- Very small development set from one paper.
- Questions 001 and 004 share a knowledge group and are not independent.
- One reviewer; no inter-reviewer agreement measurement.
- Raw PDF extraction retains formatting noise.
- The reference answer for 004 was narrowed during review; the change
  is recorded in the review file.
- Results do not establish performance on unseen questions or papers.

## Next Experiment

Compare dense retrieval against BM25 while keeping the PDF extraction,
chunk size, overlap, questions, and top_k unchanged.

Hypothesis: Dense retrieval may improve retrieval of passages describing
the degradation problem, rather than passages merely mentioning its
solution.

Success criteria:

- Question 002 retrieves evidence sufficient to explain the degradation
  problem within the top 3 results.
- Overall evidence sufficiency improves from 2/4 to at least 3/4 on this
  pilot development set.
- Questions 001 and 004 remain sufficient.
- Evidence is judged by content, not by matching a particular page alone.

These are experimental targets, not established results. Findings on this
small development set require validation on additional question groups.

## Artifacts

- [Dataset](../../data/evaluation/resnet-dev.json)
- [Raw retrieval results](resnet-bm25-dev-run-01.json)
- [Manual review](resnet-bm25-dev-review-01.json)
