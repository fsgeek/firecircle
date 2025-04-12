"""
Base Memory Store interface for Fire Circle.

This module defines the abstract base class that all memory store implementations
must adhere to, ensuring consistent behavior across different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from firecircle.protocol.message import Dialogue, Message


class MemoryStore(ABC):
    """
    Abstract base class for memory storage implementations.
    
    The MemoryStore provides an interface for storing, retrieving, and
    searching messages and dialogues in the Fire Circle system.
    """
    
    @abstractmethod
    async def store_message(self, message: Message) -> bool:
        """
        Store a message in the memory store.
        
        Args:
            message: The message to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def store_dialogue(self, dialogue: Dialogue) -> bool:
        """
        Store a complete dialogue in the memory store.
        
        Args:
            dialogue: The dialogue to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_message(self, message_id: UUID) -> Optional[Message]:
        """
        Retrieve a message by its ID.
        
        Args:
            message_id: The ID of the message to retrieve
            
        Returns:
            The Message object if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_dialogue(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """
        Retrieve a dialogue by its ID.
        
        Args:
            dialogue_id: The ID of the dialogue to retrieve
            
        Returns:
            The Dialogue object if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_dialogue_messages(self, 
                                  dialogue_id: UUID, 
                                  limit: Optional[int] = None,
                                  offset: Optional[int] = None) -> List[Message]:
        """
        Retrieve messages from a specific dialogue.
        
        Args:
            dialogue_id: The ID of the dialogue
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip
            
        Returns:
            List of Message objects in the dialogue
        """
        pass
    
    @abstractmethod
    async def search_messages(self, 
                            query: str, 
                            dialogue_id: Optional[UUID] = None,
                            limit: int = 10) -> List[Message]:
        """
        Search for messages semantically related to the query.
        
        Args:
            query: The search query
            dialogue_id: Optional dialogue ID to limit the search scope
            limit: Maximum number of results to return
            
        Returns:
            List of Message objects matching the search criteria
        """
        pass
    
    @abstractmethod
    async def search_dialogues(self, 
                             query: str, 
                             limit: int = 10) -> List[Dialogue]:
        """
        Search for dialogues semantically related to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of Dialogue objects matching the search criteria
        """
        pass
    
    @abstractmethod
    async def update_message(self, message: Message) -> bool:
        """
        Update an existing message in the memory store.
        
        Args:
            message: The updated message
            
        Returns:
            True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_message(self, message_id: UUID) -> bool:
        """
        Delete a message from the memory store.
        
        Args:
            message_id: The ID of the message to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_dialogue(self, dialogue_id: UUID) -> bool:
        """
        Delete a dialogue and all its messages from the memory store.
        
        Args:
            dialogue_id: The ID of the dialogue to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_related_contexts(self, 
                                 message: Message, 
                                 max_contexts: int = 5) -> List[Message]:
        """
        Find messages that provide relevant context for a given message.
        
        This method uses semantic search to find messages that are conceptually
        related to the given message, which can serve as additional context
        for generating responses.
        
        Args:
            message: The message to find related contexts for
            max_contexts: Maximum number of contextual messages to return
            
        Returns:
            List of contextually relevant messages
        """
        pass