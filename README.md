# PGX Chatbot — Pharmacogenomics RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that assists healthcare professionals with pharmacogenomics (PGX) dosing recommendations, built with LangChain, OpenAI, and FAISS.

## Overview

This application provides clinical decision support for TPMT (thiopurine methyltransferase) genotype-based dosing guidance. It retrieves relevant passages from curated pharmacogenomics documents and uses GPT to generate context-aware clinical answers.

## How It Works

1. **Document Loading** — Loads the TPMT pharmacogenomics guideline document
2. **Chunking** — Splits text into overlapping chunks for precise retrieval
3. **Embedding** — Converts chunks to vector embeddings via OpenAI Embeddings API
4. **Indexing** — Stores and queries embeddings using FAISS
5. **Generation** — Uses GPT-3.5 Turbo to answer questions with retrieved context

## Features

- Context-aware Q&A grounded in real clinical documents
- Overlap-aware text chunking for retrieval accuracy
- Streamlit interface for easy clinical use
- Modular LangChain pipeline

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| LLM | OpenAI GPT-3.5 Turbo |
| Embeddings | OpenAI Embeddings |
| Vector Store | FAISS |
| Pipeline | LangChain |
| UI | Streamlit |

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
git clone https://github.com/DAleid/OpenAI-API-.git
cd OpenAI-API-
pip install openai langchain faiss-cpu streamlit tiktoken
```

### Configuration

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Run

```bash
streamlit run Openai_langchain_chatbot.py
```

Open `http://localhost:8501`.

## Use Case

Healthcare professionals can query the chatbot with questions like:
- *"What dose adjustment is recommended for patients with TPMT poor metabolizer status?"*
- *"How should I adjust thiopurine dosing for intermediate metabolizers?"*

## License

MIT License