FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml ./

# Configure Poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (this will download sentence-transformers models)
RUN poetry install --no-interaction --no-ansi --no-root --no-dev

# Copy application code
COPY src/ ./src/

# Add src to PYTHONPATH so imports work
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expose port
EXPOSE 8002

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8002
ENV CHROMADB_HOST=maestro-chromadb-dev
ENV CHROMADB_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
