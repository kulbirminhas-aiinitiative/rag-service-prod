"""
Persona RAG Manager
Manages per-persona knowledge bases with ChromaDB collections
Each persona has their own SME docs, execution history, and patterns
Updated for ChromaDB v2 API
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

try:
    from .vector_rag_manager import VectorRAGManager
    import chromadb
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    logger.warning("RAG dependencies not available")


class PersonaRAGManager:
    """
    Manages RAG knowledge bases for individual personas
    Each persona gets 3 collections: sme_docs, execution_history, patterns
    """

    # Persona collection naming convention
    COLLECTION_TEMPLATE = "{persona}_{knowledge_type}"

    # Knowledge types
    KNOWLEDGE_TYPES = ["sme_docs", "execution_history", "patterns"]

    def __init__(self, config: Dict[str, Any] = None, chromadb_host: str = "maestro-chromadb-dev", chromadb_port: int = 8000):
        if not DEPS_AVAILABLE:
            raise ImportError("RAG dependencies not available")

        # Use base VectorRAGManager for config and embeddings
        self.base_rag = VectorRAGManager(config=config, chromadb_host=chromadb_host, chromadb_port=chromadb_port)
        self.client = self.base_rag.client
        self.embedding_function = self.base_rag.embedding_function

        # Cache for persona collections
        self.persona_collections: Dict[str, Dict[str, Any]] = {}

        # Query cache (persona -> query_hash -> results)
        self.query_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("✅ PersonaRAGManager initialized")

    def _get_collection_name(self, persona: str, knowledge_type: str) -> str:
        """Generate collection name for persona and knowledge type"""
        return self.COLLECTION_TEMPLATE.format(
            persona=persona.lower().replace(" ", "_"),
            knowledge_type=knowledge_type
        )

    def _ensure_persona_collections(self, persona: str):
        """Ensure all collections exist for a persona"""
        if persona in self.persona_collections:
            return

        collections = {}
        for knowledge_type in self.KNOWLEDGE_TYPES:
            collection_name = self._get_collection_name(persona, knowledge_type)

            try:
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"📚 Loaded collection: {collection_name}")
            except:
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "persona": persona,
                        "knowledge_type": knowledge_type,
                        "hnsw:space": "cosine"
                    }
                )
                logger.info(f"🆕 Created collection: {collection_name}")

            collections[knowledge_type] = collection

        self.persona_collections[persona] = collections

    async def add_sme_document(
        self,
        persona: str,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ):
        """Add SME document to persona's knowledge base"""
        self._ensure_persona_collections(persona)

        embedding = self.base_rag.embed_text(content)

        collection = self.persona_collections[persona]["sme_docs"]
        collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "persona": persona,
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }],
            ids=[doc_id]
        )

        logger.info(f"📄 Added SME doc for {persona}: {doc_id}")

    async def add_execution_history(
        self,
        persona: str,
        execution_id: str,
        query: str,
        response: str,
        success: bool,
        metadata: Dict[str, Any]
    ):
        """Store execution interaction for persona learning"""
        self._ensure_persona_collections(persona)

        # Create document from interaction
        document = f"""
Query: {query}
Response: {response}
Success: {success}
Context: {metadata.get('context', 'N/A')}
"""

        embedding = self.base_rag.embed_text(document)

        collection = self.persona_collections[persona]["execution_history"]
        collection.add(
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "persona": persona,
                "execution_id": execution_id,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }],
            ids=[execution_id]
        )

        logger.info(f"📝 Added execution history for {persona}: {execution_id}")

    async def add_pattern(
        self,
        persona: str,
        pattern_id: str,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Store successful pattern for persona"""
        self._ensure_persona_collections(persona)

        # Serialize pattern data
        import json
        document = json.dumps(pattern_data, indent=2)

        embedding = self.base_rag.embed_text(document)

        collection = self.persona_collections[persona]["patterns"]
        collection.add(
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "persona": persona,
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }],
            ids=[pattern_id]
        )

        logger.info(f"🎯 Added pattern for {persona}: {pattern_id} ({pattern_type})")

    async def query_persona_knowledge(
        self,
        persona: str,
        query: str,
        collections: Optional[List[str]] = None,
        top_k: int = 3,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query persona's knowledge bases
        Returns relevant documents from specified collections
        """
        self._ensure_persona_collections(persona)

        # Default to all knowledge types
        if collections is None:
            collections = self.KNOWLEDGE_TYPES

        # Check cache
        cache_key = self._get_cache_key(persona, query, collections, top_k)
        if cache_key in self.query_cache.get(persona, {}):
            logger.info(f"💨 Cache hit for {persona} query")
            return self.query_cache[persona][cache_key]

        # Generate query embedding
        query_embedding = self.base_rag.embed_text(query)

        results = []
        sources_used = []

        # Query each requested collection
        for knowledge_type in collections:
            if knowledge_type not in self.persona_collections[persona]:
                continue

            collection = self.persona_collections[persona][knowledge_type]

            try:
                query_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )

                # Process results
                if query_results['ids'] and len(query_results['ids'][0]) > 0:
                    for i, doc_id in enumerate(query_results['ids'][0]):
                        distance = query_results['distances'][0][i] if 'distances' in query_results else 0
                        relevance = 1 - distance

                        if relevance >= min_confidence:
                            results.append({
                                "doc_id": doc_id,
                                "content": query_results['documents'][0][i],
                                "relevance": relevance,
                                "source_type": knowledge_type,
                                "metadata": query_results['metadatas'][0][i]
                            })
                            sources_used.append(knowledge_type)

            except Exception as e:
                logger.error(f"Error querying {knowledge_type} for {persona}: {e}")

        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)

        # Calculate average confidence
        avg_confidence = sum(r['relevance'] for r in results) / len(results) if results else 0.0

        response = {
            "query": query,
            "results": results[:top_k],  # Limit to top_k total
            "avg_confidence": avg_confidence,
            "sources_used": list(set(sources_used))
        }

        # Cache results
        if persona not in self.query_cache:
            self.query_cache[persona] = {}
        self.query_cache[persona][cache_key] = response

        logger.info(f"🔍 RAG query for {persona}: {len(results)} results (avg conf: {avg_confidence:.2f})")

        return response

    def _get_cache_key(self, persona: str, query: str, collections: List[str], top_k: int) -> str:
        """Generate cache key for query"""
        key_str = f"{persona}|{query}|{'_'.join(sorted(collections))}|{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_persona_stats(self, persona: str) -> Dict[str, Any]:
        """Get statistics for persona's knowledge bases"""
        self._ensure_persona_collections(persona)

        stats = {
            "persona": persona,
            "collections": {},
            "total_documents": 0
        }

        for knowledge_type, collection in self.persona_collections[persona].items():
            count = collection.count()
            stats["collections"][knowledge_type] = count
            stats["total_documents"] += count

        # Cache stats
        cache_persona = self.query_cache.get(persona, {})
        stats["cache_hits"] = sum(1 for v in cache_persona.values() if v)
        stats["cache_misses"] = 0  # Would need tracking

        return stats

    def clear_persona_cache(self, persona: Optional[str] = None):
        """Clear query cache for persona or all"""
        if persona:
            self.query_cache.pop(persona, None)
            logger.info(f"🧹 Cleared cache for {persona}")
        else:
            self.query_cache.clear()
            logger.info("🧹 Cleared all persona caches")

    def list_persona_knowledge(self, persona: str, knowledge_type: str = None) -> List[Dict[str, Any]]:
        """List all documents in persona's knowledge base"""
        self._ensure_persona_collections(persona)

        if knowledge_type and knowledge_type not in self.KNOWLEDGE_TYPES:
            raise ValueError(f"Invalid knowledge_type: {knowledge_type}")

        types_to_query = [knowledge_type] if knowledge_type else self.KNOWLEDGE_TYPES

        documents = []
        for ktype in types_to_query:
            collection = self.persona_collections[persona][ktype]
            all_docs = collection.get()

            for i, doc_id in enumerate(all_docs['ids']):
                documents.append({
                    "doc_id": doc_id,
                    "knowledge_type": ktype,
                    "metadata": all_docs['metadatas'][i],
                    "content_preview": all_docs['documents'][i][:200] + "..."
                })

        return documents


# Global persona RAG manager instance
_global_persona_rag = None


def get_persona_rag_manager(config: Dict[str, Any] = None, chromadb_host: str = "maestro-chromadb-dev", chromadb_port: int = 8000) -> PersonaRAGManager:
    """Get global persona RAG manager instance"""
    global _global_persona_rag

    if _global_persona_rag is None:
        _global_persona_rag = PersonaRAGManager(config=config, chromadb_host=chromadb_host, chromadb_port=chromadb_port)

    return _global_persona_rag
