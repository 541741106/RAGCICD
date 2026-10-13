# CV Paper Research Assistant

## Project Goals

This project aims to develop a Computer Vision (CV) Paper Research Assistant based on Retrieval-Augmented Generation (RAG), helping students and researchers efficiently read, understand, and compare academic papers in the field of computer vision.

When reviewing multiple research papers, researchers often need to identify key contributions, examine methodologies, analyze experimental results, and compare different approaches. Manually locating and organizing this information can be time-consuming.

To address these challenges, this project aims to combine document processing, semantic retrieval, and Large Language Models (LLMs) to enable evidence-grounded question answering over research papers.

A key objective is to provide traceable references to the original papers, allowing users to verify generated answers and reducing the risk of unsupported or hallucinated information.

## Inputs and Outputs

### 1. Paper Ingestion

**Input:**

- CV research papers in PDF format.
- Single or multiple PDF uploads (planned functionality).
- Source information for each paper, such as its URL or arXiv identifier.

**Output:**

- Paper identifiers and titles for successfully ingested papers.
- Document processing status, including successful ingestion or error information.

**Planned Internal Processing:**

- Extract paper text and preserve source information and PDF page indices.
- Split extracted text into chunks with associated metadata.
- Generate embeddings and store chunks in a searchable repository.

### 2. Paper Question Answering

**Input:**

Users submit natural-language questions about papers stored in the knowledge base. Users may optionally restrict retrieval to selected papers. Without a restriction, retrieval searches the entire ingested collection.

Example questions include:

- What are the main contributions of this paper?
- What neural network architecture does the proposed method use?
- Which datasets are used in the experiments?
- What are the key differences between the methods proposed in Paper A and Paper B?

**Output:**

The system is designed to generate answers based on retrieved evidence from the papers, including:

- **Answer:** A clear explanation or comparison addressing the user's question.
- **Paper Title:** The name of the paper supporting the answer.
- **Page Number:** The 1-based PDF page index containing the evidence, which may differ from the page number printed in the paper.
- **Evidence Snippet:** Relevant excerpts from the original paper that allow users to verify the answer.

If the retrieved information is insufficient to support an answer, the system should explicitly state:

"Insufficient evidence was found in the current paper collection to answer this question."

The system should avoid fabricating research findings, experimental results, or conclusions that are not supported by the source documents.

## Current Progress

### Completed and Verified

- [x] Set up the Python and Conda development environment.
- [x] Created the initial project directory and Python package structure.
- [x] Verified that the package structure and basic imports work correctly.

### Planned / Not Yet Implemented

- [ ] PDF parsing and text extraction.
- [ ] Document chunking and metadata management.
- [ ] Text embedding generation and vector storage.
- [ ] Relevant document retrieval based on user queries.
- [ ] Evidence-grounded answer generation using an LLM.
- [ ] Source citations with paper titles, page numbers, and evidence snippets.
- [ ] Cross-paper comparison and question answering.
- [ ] Evaluation of retrieval and answer generation quality.

The project is currently in its initial development stage. The development environment and basic project structure have been verified, but the end-to-end paper ingestion and question-answering pipelines have not yet been implemented.

Future development will focus on implementing and integrating the core RAG components, followed by evaluating the system's reliability and accuracy using real computer vision research papers.

## Development Setup

Run the following commands from the repository root:

```powershell
conda env create -f environment.yml
conda activate llmrag
python -m pip install -e ".[dev]"
python -c "import llmrag; print(llmrag.__file__)"
```

If the `llmrag` environment already exists, skip the first command.

Currently, only the development environment and package import have been
verified. RAG functionality has not yet been implemented.