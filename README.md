# Local RAG Application

A self-hosted RAG application using FastAPI, ChromaDB, Sentence Transformers, and Ollama.

Everything runs in containers using Podman Compose. No Python or Ollama installation is required on the host machine.

## Requirements

- Podman
- Podman Compose provider
- Internet access for initial downloads
- Sufficient RAM for the embedding and LLM models


## Installation

Clone the repository:

    git clone git@github.com:mathiashultman/rag.git
    cd rag

Build the API image:

    podman compose build

Start the containers:

    podman compose up -d

Download the language model:

    podman compose exec ollama ollama pull qwen3:4b

The embedding model (`BAAI/bge-m3`) is downloaded during the API image build.

## Usage

Check API health:

    curl http://localhost:8000/health

Point your browser to:

    http://localhost:8000/?query=vem är vd

Or ask a question via curl:

    curl --get \
      --data-urlencode "query=Vem är VD?" \
      http://localhost:8000/