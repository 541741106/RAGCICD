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

- [x] Character chunking with behavior tests.
- [x] In-memory BM25 retrieval over synthetic documents.
- [x] Evidence prompt construction and Gemini answer generation.
- [x] Command-line demo and automated tests using mocked model calls.
- [x] GitHub Actions lint, formatting, and test checks.

### Planned / Not Yet Implemented

- [ ] Integrate chunking and metadata management with real PDF ingestion.
- [ ] Evaluate retrieval and generation on real papers.
- [ ] Validate citations against source evidence.

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

A command-line RAG demo over three synthetic CV texts is implemented.
Real PDF ingestion, persistent storage, and citation validation are
not yet implemented.

## Run the Synthetic RAG Demo

The demo retrieves from three synthetic CV texts, builds an evidence
prompt, and calls Gemini. It does not ingest real papers yet.

Install the project using the Development Setup instructions above.

Set the API key in the current CMD / Anaconda Prompt session:

```bat
set "GEMINI_API_KEY=YOUR_API_KEY"
```

Run a question:

```bat
python -m llmrag "What does image segmentation assign to each pixel?" --top-k 2
```

The command makes a real API request and consumes model quota.
It prints the answer followed by retrieved evidence and source metadata.
Evidence numbers correspond to their order in the retrieved results;
citations are not automatically validated.

Show command options without calling the model:

```bat
python -m llmrag --help
```

Run automated checks without calling the model:

```bat
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```
