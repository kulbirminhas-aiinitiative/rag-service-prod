"""RAG Service Pydantic Models"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "rag-service"
    version: str = "1.0.0"
    chromadb_connected: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class IngestDocumentRequest(BaseModel):
    """Request to ingest a document"""
    collection_name: str = Field(..., description="Target collection name")
    document_id: str = Field(..., description="Unique document ID")
    content: str = Field(..., min_length=1, description="Document content to index")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class IngestDocumentResponse(BaseModel):
    """Response from document ingestion"""
    document_id: str
    collection_name: str
    status: str
    indexed_at: datetime = Field(default_factory=datetime.now)


class SearchRequest(BaseModel):
    """Request to search documents"""
    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of results to return")
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class SearchResult(BaseModel):
    """Individual search result"""
    document_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response from search"""
    query: str
    collection_name: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


class CollectionInfo(BaseModel):
    """Collection information"""
    name: str
    count: int
    metadata: Dict[str, Any]


class CollectionsResponse(BaseModel):
    """Response listing all collections"""
    collections: List[CollectionInfo]
    total_collections: int
