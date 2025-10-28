#!/usr/bin/env python3
"""
Vector RAG Manager
Core vector database management with ChromaDB and in-memory caching
Updated for ChromaDB v2 API with HttpClient
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    logger.warning("ChromaDB or sentence-transformers not installed. Install with: pip install chromadb sentence-transformers")


class VectorRAGManager:
    """
    Vector database manager for Maestro RAG workflow
    Uses ChromaDB with sentence-transformers for semantic search
    """

    def __init__(self, config: Dict[str, Any] = None, chromadb_host: str = "maestro-chromadb-dev", chromadb_port: int = 8000):
        if not DEPS_AVAILABLE:
            raise ImportError("Required dependencies not available. Install: pip install chromadb sentence-transformers")

        self.config = config or self._default_config()

        # Use HttpClient for ChromaDB v2 API
        self.client = chromadb.HttpClient(
            host=chromadb_host,
            port=chromadb_port
        )

        embedding_model = self.config['embedding']['model']
        self.embedding_function = SentenceTransformer(embedding_model)

        self.collections = {
            'executions': self._get_or_create_collection('executions'),
            'collaterals': self._get_or_create_collection('collaterals'),
            'patterns': self._get_or_create_collection('patterns')
        }

        self.in_memory_cache = {}

        logger.info(f"✅ VectorRAGManager initialized")
        logger.info(f"🌐 ChromaDB: {chromadb_host}:{chromadb_port}")
        logger.info(f"🤖 Embedding model: {embedding_model}")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "chromadb": {
                "collection_names": {
                    "executions": "maestro_execution_history",
                    "collaterals": "reusable_collaterals",
                    "patterns": "successful_patterns"
                },
                "distance_function": "cosine"
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu"
            },
            "rag": {
                "top_k_similar": 3,
                "similarity_threshold": 0.25,
                "context_window_tokens": 2000,
                "max_history_days": 90
            }
        }

    def _get_or_create_collection(self, collection_type: str):
        """Get or create a ChromaDB collection"""
        collection_name = self.config['chromadb']['collection_names'][collection_type]
        distance_function = self.config['chromadb'].get('distance_function', 'cosine')

        try:
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"📚 Loaded existing collection: {collection_name}")
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "type": collection_type,
                    "hnsw:space": distance_function
                }
            )
            logger.info(f"🆕 Created new collection: {collection_name} (distance: {distance_function})")

        return collection

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        cache_key = hash(text)

        if cache_key in self.in_memory_cache:
            return self.in_memory_cache[cache_key]

        embedding = self.embedding_function.encode(text).tolist()
        self.in_memory_cache[cache_key] = embedding

        return embedding

    def add_execution(self, execution_id: str, requirement: str,
                     metadata: Dict[str, Any], collaterals: Dict[str, Any] = None):
        """Add execution to vector database"""

        document = f"""
Requirement: {requirement}
Team: {', '.join(metadata.get('team_members', []))}
Files Generated: {metadata.get('total_files', 0)}
Success: {metadata.get('success', False)}
Execution Time: {metadata.get('execution_time', 0)}s
"""

        if collaterals:
            document += f"\nDeliverables: {json.dumps(collaterals, indent=2)}"

        embedding = self.embed_text(document)

        self.collections['executions'].add(
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                'execution_id': execution_id,
                'requirement': requirement,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }],
            ids=[execution_id]
        )

        logger.info(f"📝 Added execution: {execution_id}")

    def add_collateral(self, collateral_id: str, collateral_type: str,
                      content: str, metadata: Dict[str, Any]):
        """Add reusable collateral to vector database"""

        embedding = self.embed_text(content)

        self.collections['collaterals'].add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                'collateral_id': collateral_id,
                'type': collateral_type,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }],
            ids=[collateral_id]
        )

        logger.info(f"📄 Added collateral: {collateral_id} ({collateral_type})")

    def add_pattern(self, pattern_id: str, pattern_type: str,
                   pattern_data: Dict[str, Any], metadata: Dict[str, Any]):
        """Add successful pattern to vector database"""

        document = json.dumps(pattern_data, indent=2)
        embedding = self.embed_text(document)

        self.collections['patterns'].add(
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                'pattern_id': pattern_id,
                'type': pattern_type,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }],
            ids=[pattern_id]
        )

        logger.info(f"🎯 Added pattern: {pattern_id} ({pattern_type})")

    def search_similar_executions(self, requirement: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar executions using vector similarity"""

        if top_k is None:
            top_k = self.config['rag']['top_k_similar']

        query_embedding = self.embed_text(requirement)

        results = self.collections['executions'].query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        similar_executions = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, exec_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1 - distance

                if similarity >= self.config['rag']['similarity_threshold']:
                    similar_executions.append({
                        'execution_id': exec_id,
                        'similarity': similarity,
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i]
                    })

        logger.info(f"🔍 Found {len(similar_executions)} similar executions for: {requirement[:50]}...")

        return similar_executions

    def search_similar_collaterals(self, query: str, collateral_type: str = None,
                                   top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar collaterals"""

        query_embedding = self.embed_text(query)

        where_filter = {"type": collateral_type} if collateral_type else None

        results = self.collections['collaterals'].query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        similar_collaterals = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, coll_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1 - distance

                similar_collaterals.append({
                    'collateral_id': coll_id,
                    'similarity': similarity,
                    'metadata': results['metadatas'][0][i],
                    'content': results['documents'][0][i]
                })

        return similar_collaterals

    def search_patterns(self, query: str, pattern_type: str = None,
                       top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for successful patterns"""

        query_embedding = self.embed_text(query)

        where_filter = {"type": pattern_type} if pattern_type else None

        results = self.collections['patterns'].query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        patterns = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, pattern_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1 - distance

                patterns.append({
                    'pattern_id': pattern_id,
                    'similarity': similarity,
                    'metadata': results['metadatas'][0][i],
                    'pattern_data': json.loads(results['documents'][0][i])
                })

        return patterns

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""

        stats = {}
        for name, collection in self.collections.items():
            count = collection.count()
            stats[name] = {
                'count': count,
                'collection_name': self.config['chromadb']['collection_names'][name]
            }

        stats['cache_size'] = len(self.in_memory_cache)
        stats['config'] = self.config

        return stats

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in ChromaDB"""
        try:
            collections = self.client.list_collections()
            return [
                {
                    "name": col.name,
                    "count": col.count(),
                    "metadata": col.metadata
                }
                for col in collections
            ]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def clear_cache(self):
        """Clear in-memory embedding cache"""
        self.in_memory_cache.clear()
        logger.info("🧹 Cleared in-memory cache")

    def cleanup_old_entries(self, days: int = None):
        """Remove old entries based on age"""

        if days is None:
            days = self.config['rag'].get('max_history_days', 90)

        cutoff_date = datetime.now() - timedelta(days=days)

        for collection_name, collection in self.collections.items():
            all_items = collection.get()

            ids_to_delete = []
            for i, metadata in enumerate(all_items['metadatas']):
                timestamp_str = metadata.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp < cutoff_date:
                            ids_to_delete.append(all_items['ids'][i])
                    except:
                        pass

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"🗑️  Deleted {len(ids_to_delete)} old entries from {collection_name}")


_global_rag_manager = None

def get_rag_manager(config: Dict[str, Any] = None, chromadb_host: str = "maestro-chromadb-dev", chromadb_port: int = 8000) -> VectorRAGManager:
    """Get global RAG manager instance"""
    global _global_rag_manager

    if _global_rag_manager is None:
        _global_rag_manager = VectorRAGManager(config=config, chromadb_host=chromadb_host, chromadb_port=chromadb_port)

    return _global_rag_manager
