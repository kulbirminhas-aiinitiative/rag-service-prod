"""
RAG Service - FastAPI Application
Handles document ingestion, embeddings, and semantic search using ChromaDB
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    HealthResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    CollectionsResponse,
    CollectionInfo,
)
from .rag import get_rag_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG manager
rag_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_manager

    # Startup
    logger.info("🚀 Starting RAG Service...")

    # Get ChromaDB connection details from environment
    chromadb_host = os.getenv("CHROMADB_HOST", "maestro-chromadb-dev")
    chromadb_port = int(os.getenv("CHROMADB_PORT", "8000"))

    try:
        # Initialize RAG manager with ChromaDB connection
        rag_manager = get_rag_manager(
            chromadb_host=chromadb_host,
            chromadb_port=chromadb_port
        )
        logger.info(f"✅ Connected to ChromaDB at {chromadb_host}:{chromadb_port}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG manager: {e}")
        raise

    yield

    # Shutdown
    logger.info("👋 Shutting down RAG Service...")


# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    description="Document processing and semantic search service using ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    chromadb_connected = False

    try:
        if rag_manager:
            # Test connection by listing collections
            rag_manager.list_collections()
            chromadb_connected = True
    except Exception as e:
        logger.error(f"Health check failed: {e}")

    return HealthResponse(
        status="healthy" if chromadb_connected else "unhealthy",
        chromadb_connected=chromadb_connected,
    )


@app.post("/api/v1/ingest", response_model=IngestDocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(request: IngestDocumentRequest):
    """
    Ingest a document into a collection

    Creates embeddings and stores the document in ChromaDB for semantic search.
    """
    if not rag_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )

    try:
        logger.info(f"📥 Ingesting document {request.document_id} into {request.collection_name}")

        # Generate embedding
        embedding = rag_manager.embed_text(request.content)

        # Get or create collection
        try:
            collection = rag_manager.client.get_collection(name=request.collection_name)
        except:
            collection = rag_manager.client.create_collection(
                name=request.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"🆕 Created new collection: {request.collection_name}")

        # Add document to collection
        collection.add(
            embeddings=[embedding],
            documents=[request.content],
            metadatas=[request.metadata],
            ids=[request.document_id]
        )

        logger.info(f"✅ Document {request.document_id} ingested successfully")

        return IngestDocumentResponse(
            document_id=request.document_id,
            collection_name=request.collection_name,
            status="indexed",
        )

    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for similar documents in a collection

    Uses semantic search to find documents similar to the query.
    """
    if not rag_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )

    try:
        start_time = time.time()

        logger.info(f"🔍 Searching in {request.collection_name} for: {request.query[:50]}...")

        # Get collection
        try:
            collection = rag_manager.client.get_collection(name=request.collection_name)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection_name}' not found"
            )

        # Generate query embedding
        query_embedding = rag_manager.embed_text(request.query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            where=request.filter_metadata
        )

        # Process results
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1 - distance

                # Filter by minimum similarity
                if similarity >= request.min_similarity:
                    search_results.append(
                        SearchResult(
                            document_id=doc_id,
                            content=results['documents'][0][i],
                            similarity=similarity,
                            metadata=results['metadatas'][0][i]
                        )
                    )

        search_time_ms = (time.time() - start_time) * 1000

        logger.info(f"✅ Found {len(search_results)} results in {search_time_ms:.2f}ms")

        return SearchResponse(
            query=request.query,
            collection_name=request.collection_name,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search documents: {str(e)}"
        )


@app.get("/api/v1/collections", response_model=CollectionsResponse)
async def list_collections():
    """
    List all collections in ChromaDB

    Returns information about all available collections.
    """
    if not rag_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )

    try:
        logger.info("📚 Listing all collections")

        collections = rag_manager.list_collections()

        collection_infos = [
            CollectionInfo(
                name=col["name"],
                count=col["count"],
                metadata=col["metadata"]
            )
            for col in collections
        ]

        logger.info(f"✅ Found {len(collection_infos)} collections")

        return CollectionsResponse(
            collections=collection_infos,
            total_collections=len(collection_infos),
        )

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )


@app.delete("/api/v1/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a collection

    Removes a collection and all its documents from ChromaDB.
    """
    if not rag_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )

    try:
        logger.info(f"🗑️  Deleting collection: {collection_name}")

        rag_manager.client.delete_collection(name=collection_name)

        logger.info(f"✅ Collection {collection_name} deleted")

        return {"status": "success", "message": f"Collection '{collection_name}' deleted"}

    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}"
        )


@app.get("/api/v1/context/{persona}/{session_id}")
async def get_context(persona: str, session_id: str, iteration: int = 1):
    """
    Get conversation context for a persona in a session

    Returns conversation history and team context for multi-agent orchestration.
    MVP implementation returns empty structure - will be enhanced with actual context storage.
    """
    logger.info(f"📖 Getting context for persona={persona}, session={session_id}, iteration={iteration}")

    return {
        "conversation_history": [],
        "team_context": {},
        "persona": persona,
        "session_id": session_id,
        "iteration": iteration
    }


@app.post("/api/v1/interactions/store")
async def store_interaction(request: dict):
    """
    Store interaction for future context retrieval

    Stores conversation interactions for context building in multi-agent sessions.
    MVP implementation logs and returns success - will be enhanced with actual storage.
    """
    persona = request.get("persona", "unknown")
    session_id = request.get("session_id", "unknown")

    logger.info(f"💾 Storing interaction: persona={persona}, session={session_id}")
    logger.debug(f"Interaction data: {request}")

    return {
        "status": "stored",
        "session_id": session_id,
        "persona": persona
    }


@app.post("/api/v1/personas/{persona}/query")
async def query_persona_knowledge(persona: str, request: dict):
    """
    Query persona's knowledge base

    Searches for relevant context from persona-specific knowledge.
    MVP implementation returns empty results - will be enhanced with actual persona knowledge storage.
    """
    query = request.get("query", "")
    top_k = request.get("top_k", 3)

    logger.info(f"🔎 Querying knowledge for persona={persona}, query={query[:50]}...")

    return {
        "results": [],
        "total": 0,
        "query": query,
        "persona": persona,
        "top_k": top_k
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(
        "rag_service.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
