# Semantic Search RAG (Offline)

This project implements a Retrieval-Augmented Generation (RAG) system that performs semantic search and generates answers using a locally hosted LLM.

## Features

* Semantic search using sentence-transformers
* Cosine similarity based retrieval
* Top-k document selection
* Offline LLM using Ollama (Llama3)
* Streamlit Web UI
* No API key required
* Fully local pipeline

## Project Structure

```
semantic-search-rag/
│
├── data/
│   ├── documents.txt
│   └── embeddings.npy
│
└── src/
    ├── build_index.py
    ├── search.py
    ├── rag.py
    └── app.py
```

## Installation

```bash
pip install -r requirements.txt
```

Install Ollama and pull model:

```bash
ollama pull llama3
```

## Run (CLI Version)

```bash
python src/rag.py
```

## Run (Web UI)

```bash
streamlit run src/app.py
```

Then open the browser and ask questions interactively.

## Example

Query:

```
Explain semantic search
```

Output:

```
Semantic search improves retrieval by understanding meaning...
```

## Tech Stack

* Python
* Sentence Transformers
* NumPy
* Scikit-learn
* Ollama
* Llama3
* Streamlit

## Author

Sourajeet Raha
