"""
In-memory implementation of the Memory Store for Fire Circle.

This module provides a simple in-memory implementation of the MemoryStore interface,
useful for testing and development purposes.
"""

import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set
from uuid import UUID
import re

from firecircle.memory.base import MemoryStore
from firecircle.protocol.message import Dialogue, Message


class InMemoryStore(MemoryStore):
    """
    In-memory implementation of the Memory Store.
    
    This implementation stores all data in memory, making it fast but non-persistent.
    It's primarily intended for testing, development, and small-scale deployments.
    """
    
    def __init__(self):
        """Initialize the in-memory store."""
        self._messages: Dict[UUID, Message] = {}
        self._dialogues: Dict[UUID, Dialogue] = {}
        self._dialogue_messages: Dict[UUID, Set[UUID]] = {}
        self._dialogue_index: Dict[UUID, Dict[str, float]] = {}  # Simple term frequency index
        self._message_index: Dict[UUID, Dict[str, float]] = {}   # Simple term frequency index
        
    async def store_message(self, message: Message) -> bool:
        """Store a message in memory."""
        try:
            # Store a deep copy to prevent modifications
            self._messages[message.id] = deepcopy(message)
            
            # Add to dialogue-message mapping
            dialogue_id = message.metadata.dialogue_id
            if dialogue_id not in self._dialogue_messages:
                self._dialogue_messages[dialogue_id] = set()
            self._dialogue_messages[dialogue_id].add(message.id)
            
            # Index the message text for search
            self._index_message(message)
            
            return True
        except Exception:
            return False
    
    async def store_dialogue(self, dialogue: Dialogue) -> bool:
        """Store a dialogue and all its messages in memory."""
        try:
            # Store a deep copy of the dialogue
            self._dialogues[dialogue.id] = deepcopy(dialogue)
            
            # Store all messages in the dialogue
            for message in dialogue.messages:
                await self.store_message(message)
            
            # Index the dialogue for search
            self._index_dialogue(dialogue)
            
            return True
        except Exception:
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
        # Tokenize the query
        query_terms = self._tokenize(query)
        
        # Calculate scores for each message
        scores = []
        
        for mid, terms in self._message_index.items():
            # Skip messages not in the specified dialogue if provided
            if dialogue_id is not None:
                if dialogue_id not in self._dialogue_messages or mid not in self._dialogue_messages[dialogue_id]:
                    continue
            
            # Calculate a simple similarity score (term frequency dot product)
            score = 0
            for term in query_terms:
                if term in terms:
                    score += terms[term]
            
            if score > 0:
                scores.append((mid, score))
        
        # Sort by score and take top results
        scores.sort(key=lambda x: x[1], reverse=True)
        top_message_ids = [mid for mid, _ in scores[:limit]]
        
        # Get the actual messages
        result = [deepcopy(self._messages[mid]) for mid in top_message_ids if mid in self._messages]
        
        return result
    
    async def search_dialogues(self, 
                             query: str, 
                             limit: int = 10) -> List[Dialogue]:
        """Search for dialogues semantically related to the query."""
        # Tokenize the query
        query_terms = self._tokenize(query)
        
        # Calculate scores for each dialogue
        scores = []
        
        for did, terms in self._dialogue_index.items():
            # Calculate a simple similarity score (term frequency dot product)
            score = 0
            for term in query_terms:
                if term in terms:
                    score += terms[term]
            
            if score > 0:
                scores.append((did, score))
        
        # Sort by score and take top results
        scores.sort(key=lambda x: x[1], reverse=True)
        top_dialogue_ids = [did for did, _ in scores[:limit]]
        
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
            
            # Re-index the message
            self._index_message(message)
            
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
            
            # Remove from dialogue-message mapping
            if dialogue_id in self._dialogue_messages:
                self._dialogue_messages[dialogue_id].discard(message_id)
            
            # Remove from message index
            if message_id in self._message_index:
                del self._message_index[message_id]
            
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
            
            # Remove from dialogue index
            if dialogue_id in self._dialogue_index:
                del self._dialogue_index[dialogue_id]
            
            return True
        except Exception:
            return False
    
    async def get_related_contexts(self, 
                                 message: Message, 
                                 max_contexts: int = 5) -> List[Message]:
        """Find messages that provide relevant context for a given message."""
        # Use the message text as a query for semantic search
        query = message.content.text
        
        # Get the dialogue ID to exclude messages from the same dialogue
        dialogue_id = message.metadata.dialogue_id
        
        # Perform the search, excluding messages from the same dialogue
        results = []
        
        for mid, terms in self._message_index.items():
            # Skip the message itself
            if mid == message.id:
                continue
            
            # Skip messages from the same dialogue
            msg_dialogue_id = self._messages[mid].metadata.dialogue_id if mid in self._messages else None
            if msg_dialogue_id == dialogue_id:
                continue
            
            # Calculate similarity
            query_terms = self._tokenize(query)
            score = 0
            for term in query_terms:
                if term in terms:
                    score += terms[term]
            
            if score > 0:
                results.append((mid, score))
        
        # Sort by relevance and take top results
        results.sort(key=lambda x: x[1], reverse=True)
        top_message_ids = [mid for mid, _ in results[:max_contexts]]
        
        # Get the actual messages
        return [deepcopy(self._messages[mid]) for mid in top_message_ids if mid in self._messages]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words for indexing and search.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Simple tokenization - split on non-alphanumeric chars and convert to lowercase
        return [word.lower() for word in re.findall(r'\w+', text)]
    
    def _index_message(self, message: Message) -> None:
        """
        Index a message for search.
        
        Args:
            message: The message to index
        """
        # Tokenize the message text
        tokens = self._tokenize(message.content.text)
        
        # Calculate term frequencies
        term_freq = {}
        for token in tokens:
            if token in term_freq:
                term_freq[token] += 1
            else:
                term_freq[token] = 1
        
        # Normalize by document length
        doc_length = len(tokens) or 1  # Avoid division by zero
        for term in term_freq:
            term_freq[term] = term_freq[term] / doc_length
        
        # Store the index
        self._message_index[message.id] = term_freq
    
    def _index_dialogue(self, dialogue: Dialogue) -> None:
        """
        Index a dialogue for search.
        
        Args:
            dialogue: The dialogue to index
        """
        # Combine all message texts
        all_text = " ".join([msg.content.text for msg in dialogue.messages])
        
        # Tokenize the combined text
        tokens = self._tokenize(all_text)
        
        # Calculate term frequencies
        term_freq = {}
        for token in tokens:
            if token in term_freq:
                term_freq[token] += 1
            else:
                term_freq[token] = 1
        
        # Normalize by document length
        doc_length = len(tokens) or 1  # Avoid division by zero
        for term in term_freq:
            term_freq[term] = term_freq[term] / doc_length
        
        # Store the index
        self._dialogue_index[dialogue.id] = term_freq