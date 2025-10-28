# RAG Service

Document processing and semantic search service using ChromaDB and sentence-transformers.

## Features

- Document ingestion with automatic embedding generation
- Semantic search using vector similarity
- ChromaDB v2 API integration
- FastAPI-based REST API
- Docker containerized

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/ingest` - Ingest a document
- `POST /api/v1/search` - Search for similar documents
- `GET /api/v1/collections` - List all collections
- `DELETE /api/v1/collections/{name}` - Delete a collection

## Quick Start

### Using Docker

```bash
# Build image
docker build -t rag-service:latest .

# Run container
docker run -d -p 8002:8002 \
  -e CHROMADB_HOST=maestro-chromadb-dev \
  -e CHROMADB_PORT=8000 \
  --name rag-service \
  rag-service:latest
```

### Local Development

```bash
# Install dependencies
poetry install

# Run service
poetry run python -m uvicorn rag_service.main:app --host 0.0.0.0 --port 8002
```

## Configuration

Environment variables:
- `PORT` - Service port (default: 8002)
- `CHROMADB_HOST` - ChromaDB host (default: maestro-chromadb-dev)
- `CHROMADB_PORT` - ChromaDB port (default: 8000)

## Dependencies

- **FastAPI**: Web framework
- **ChromaDB**: Vector database
- **sentence-transformers**: Embedding generation
- **Pydantic**: Data validation

## Architecture

```
rag-service/
├── src/rag_service/
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic models
│   │   └── rag_models.py
│   └── rag/                 # RAG core logic
│       ├── vector_rag_manager.py
│       └── persona_rag_manager.py
├── Dockerfile
└── pyproject.toml
```
