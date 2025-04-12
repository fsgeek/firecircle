"""
Memory Store for Fire Circle

This module provides storage and retrieval capabilities for Fire Circle dialogues,
enabling persistence, semantic search, and context management across conversations.
"""

from firecircle.memory.base import MemoryStore
from firecircle.memory.vector_store import VectorMemoryStore
from firecircle.memory.inmemory import InMemoryStore

# Import ArangoDB store conditionally to handle missing dependencies
try:
    from firecircle.memory.arangodb_store import ArangoDBStore
    __all__ = ["MemoryStore", "VectorMemoryStore", "InMemoryStore", "ArangoDBStore"]
except ImportError:
    # ArangoDB dependencies not available
    __all__ = ["MemoryStore", "VectorMemoryStore", "InMemoryStore"]