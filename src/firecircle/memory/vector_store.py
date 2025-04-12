"""
Vector-based Memory Store for Fire Circle.

This module provides a memory store implementation that uses vector embeddings
for semantic search of messages and dialogues.
"""

import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
import numpy as np

from firecircle.memory.base import MemoryStore
from firecircle.protocol.message import Dialogue, Message


class VectorMemoryStore(MemoryStore):
    """
    Vector-based implementation of the Memory Store.
    
    This implementation stores semantic vector embeddings of messages and dialogues
    to enable efficient semantic search and retrieval.
    """
    
    def __init__(self, embedding_model: Any = None, vector_dimension: int = 1536):
        """
        Initialize the vector memory store.
        
        Args:
            embedding_model: The model to use for generating embeddings
            vector_dimension: Dimension of the embedding vectors
        """
        self._messages: Dict[UUID, Message] = {}
        self._dialogues: Dict[UUID, Dialogue] = {}
        self._dialogue_messages: Dict[UUID, List[UUID]] = {}
        
        # Vector storage
        self._message_vectors: Dict[UUID, np.ndarray] = {}
        self._dialogue_vectors: Dict[UUID, np.ndarray] = {}
        
        # Embedding model
        self._embedding_model = embedding_model
        self._vector_dimension = vector_dimension
        
        # Default embedding function if no model is provided
        if self._embedding_model is None:
            # Simple random embeddings for testing/development
            self._get_embedding = lambda text: np.random.randn(self._vector_dimension).astype(np.float32)
        else:
            # Set up real embedding function
            self._setup_embedding_function()
    
    def _setup_embedding_function(self):
        """Set up the embedding function based on the provided model."""
        # This would be implemented based on the embedding model
        # For now, we'll use a placeholder
        self._get_embedding = lambda text: np.random.randn(self._vector_dimension).astype(np.float32)
    
    async def store_message(self, message: Message) -> bool:
        """Store a message and its vector embedding."""
        try:
            # Store a deep copy to prevent modifications
            self._messages[message.id] = deepcopy(message)
            
            # Add to dialogue-message mapping
            dialogue_id = message.metadata.dialogue_id
            if dialogue_id not in self._dialogue_messages:
                self._dialogue_messages[dialogue_id] = []
            self._dialogue_messages[dialogue_id].append(message.id)
            
            # Generate and store embedding
            embedding = self._get_embedding(message.content.text)
            self._message_vectors[message.id] = embedding
            
            return True
        except Exception as e:
            print(f"Error storing message: {e}")
            return False
    
    async def store_dialogue(self, dialogue: Dialogue) -> bool:
        """Store a dialogue and all its messages."""
        try:
            # Store a deep copy of the dialogue
            self._dialogues[dialogue.id] = deepcopy(dialogue)
            
            # Store all messages in the dialogue
            for message in dialogue.messages:
                await self.store_message(message)
            
            # Generate and store embedding for the entire dialogue
            all_text = " ".join([msg.content.text for msg in dialogue.messages])
            embedding = self._get_embedding(all_text)
            self._dialogue_vectors[dialogue.id] = embedding
            
            return True
        except Exception as e:
            print(f"Error storing dialogue: {e}")
            return False
    
    async def get_message(self, message_id: UUID) -> Optional[Message]:
        """Retrieve a message by its ID."""
        # Return a deep copy to prevent modifications
        return deepcopy(self._messages.get(message_id))
    
    async def get_dialogue(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """Retrieve a dialogue by its ID."""
        # Return a deep copy to prevent modifications
        return deepcopy(self._dialogues.get(dialogue_id))
    
    async def get_dialogue_messages(self, 
                                  dialogue_id: UUID, 
                                  limit: Optional[int] = None,
                                  offset: Optional[int] = None) -> List[Message]:
        """Retrieve messages from a specific dialogue."""
        if dialogue_id not in self._dialogue_messages:
            return []
        
        # Get all message IDs for this dialogue
        message_ids = self._dialogue_messages[dialogue_id]
        
        # Get the actual messages
        messages = [self._messages[mid] for mid in message_ids if mid in self._messages]
        
        # Sort by sequence number
        messages.sort(key=lambda m: m.metadata.sequence_number)
        
        # Apply offset and limit
        if offset is not None:
            messages = messages[offset:]
        if limit is not None:
            messages = messages[:limit]
        
        # Return deep copies to prevent modifications
        return [deepcopy(m) for m in messages]
    
    async def search_messages(self, 
                            query: str, 
                            dialogue_id: Optional[UUID] = None,
                            limit: int = 10) -> List[Message]:
        """Search for messages semantically related to the query."""
        # Generate embedding for the query
        query_embedding = self._get_embedding(query)
        
        # Calculate cosine similarity for each message
        similarities = []
        
        for mid, embedding in self._message_vectors.items():
            # Skip messages not in the specified dialogue if provided
            if dialogue_id is not None:
                if dialogue_id not in self._dialogue_messages or mid not in self._dialogue_messages[dialogue_id]:
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((mid, similarity))
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_message_ids = [mid for mid, _ in similarities[:limit]]
        
        # Get the actual messages
        result = [deepcopy(self._messages[mid]) for mid in top_message_ids if mid in self._messages]
        
        return result
    
    async def search_dialogues(self, 
                             query: str, 
                             limit: int = 10) -> List[Dialogue]:
        """Search for dialogues semantically related to the query."""
        # Generate embedding for the query
        query_embedding = self._get_embedding(query)
        
        # Calculate cosine similarity for each dialogue
        similarities = []
        
        for did, embedding in self._dialogue_vectors.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((did, similarity))
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_dialogue_ids = [did for did, _ in similarities[:limit]]
        
        # Get the actual dialogues
        result = [deepcopy(self._dialogues[did]) for did in top_dialogue_ids if did in self._dialogues]
        
        return result
    
    async def update_message(self, message: Message) -> bool:
        """Update an existing message in the memory store."""
        if message.id not in self._messages:
            return False
        
        try:
            # Store a deep copy to prevent modifications
            self._messages[message.id] = deepcopy(message)
            
            # Update embedding
            embedding = self._get_embedding(message.content.text)
            self._message_vectors[message.id] = embedding
            
            # Update dialogue embedding if needed
            dialogue_id = message.metadata.dialogue_id
            if dialogue_id in self._dialogues:
                all_text = " ".join([msg.content.text for msg in self._dialogues[dialogue_id].messages])
                embedding = self._get_embedding(all_text)
                self._dialogue_vectors[dialogue_id] = embedding
            
            return True
        except Exception:
            return False
    
    async def delete_message(self, message_id: UUID) -> bool:
        """Delete a message from the memory store."""
        if message_id not in self._messages:
            return False
        
        try:
            # Get the dialogue ID before removing the message
            dialogue_id = self._messages[message_id].metadata.dialogue_id
            
            # Remove from messages dictionary
            del self._messages[message_id]
            
            # Remove from message vectors
            if message_id in self._message_vectors:
                del self._message_vectors[message_id]
            
            # Remove from dialogue-message mapping
            if dialogue_id in self._dialogue_messages:
                self._dialogue_messages[dialogue_id] = [
                    mid for mid in self._dialogue_messages[dialogue_id] if mid != message_id
                ]
            
            # Update dialogue embedding if needed
            if dialogue_id in self._dialogues:
                all_text = " ".join([msg.content.text for msg in self._dialogues[dialogue_id].messages])
                embedding = self._get_embedding(all_text)
                self._dialogue_vectors[dialogue_id] = embedding
            
            return True
        except Exception:
            return False
    
    async def delete_dialogue(self, dialogue_id: UUID) -> bool:
        """Delete a dialogue and all its messages from the memory store."""
        if dialogue_id not in self._dialogues:
            return False
        
        try:
            # Delete all messages in the dialogue
            if dialogue_id in self._dialogue_messages:
                message_ids = list(self._dialogue_messages[dialogue_id])
                for message_id in message_ids:
                    await self.delete_message(message_id)
                
                # Remove the dialogue-message mapping
                del self._dialogue_messages[dialogue_id]
            
            # Remove from dialogues dictionary
            del self._dialogues[dialogue_id]
            
            # Remove from dialogue vectors
            if dialogue_id in self._dialogue_vectors:
                del self._dialogue_vectors[dialogue_id]
            
            return True
        except Exception:
            return False
    
    async def get_related_contexts(self, 
                                 message: Message, 
                                 max_contexts: int = 5) -> List[Message]:
        """Find messages that provide relevant context for a given message."""
        # Generate embedding for the message
        message_embedding = self._get_embedding(message.content.text)
        
        # Get the dialogue ID to exclude messages from the same dialogue
        dialogue_id = message.metadata.dialogue_id
        
        # Calculate cosine similarity for each message
        similarities = []
        
        for mid, embedding in self._message_vectors.items():
            # Skip the message itself
            if mid == message.id:
                continue
            
            # Skip messages from the same dialogue
            msg_dialogue_id = self._messages[mid].metadata.dialogue_id if mid in self._messages else None
            if msg_dialogue_id == dialogue_id:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(message_embedding, embedding)
            similarities.append((mid, similarity))
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_message_ids = [mid for mid, _ in similarities[:max_contexts]]
        
        # Get the actual messages
        return [deepcopy(self._messages[mid]) for mid in top_message_ids if mid in self._messages]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-9)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-9)
        
        # Calculate dot product
        return float(np.dot(vec1_norm, vec2_norm))
    
    async def set_embedding_model(self, model: Any) -> None:
        """
        Set or update the embedding model.
        
        Args:
            model: The new embedding model to use
        """
        self._embedding_model = model
        self._setup_embedding_function()
        
        # Optionally regenerate all embeddings with the new model
        # This would be an expensive operation for large datasets
    
    async def add_real_embedding_provider(self, provider_name: str, api_key: str) -> bool:
        """
        Configure a real embedding provider.
        
        Args:
            provider_name: Name of the provider (e.g., "openai")
            api_key: API key for the provider
            
        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            if provider_name.lower() == "openai":
                # Set up OpenAI embedding function
                import openai
                openai.api_key = api_key
                
                async def get_openai_embedding(text):
                    response = await openai.Embedding.acreate(
                        input=text,
                        model="text-embedding-3-small"
                    )
                    return np.array(response.data[0].embedding, dtype=np.float32)
                
                self._get_embedding = get_openai_embedding
                return True
                
            elif provider_name.lower() == "cohere":
                # Set up Cohere embedding function
                import cohere
                co_client = cohere.Client(api_key)
                
                def get_cohere_embedding(text):
                    response = co_client.embed(
                        texts=[text],
                        model="embed-english-v2.0"
                    )
                    return np.array(response.embeddings[0], dtype=np.float32)
                
                self._get_embedding = get_cohere_embedding
                return True
                
            else:
                print(f"Unsupported embedding provider: {provider_name}")
                return False
                
        except Exception as e:
            print(f"Error configuring embedding provider: {e}")
            return False