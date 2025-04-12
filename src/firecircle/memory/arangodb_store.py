"""
ArangoDB implementation of the Memory Store for Fire Circle.

This module provides a memory store implementation using ArangoDB for both
document storage and vector search capabilities.
"""

import asyncio
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

import numpy as np
from arango import ArangoClient
from arango.collection import StandardCollection
from arango.database import Database
from arango.exceptions import DocumentGetError, DocumentInsertError, CollectionCreateError

from firecircle.memory.base import MemoryStore
from firecircle.protocol.message import Dialogue, Message, MessageContent


class ArangoDBStore(MemoryStore):
    """
    ArangoDB-based implementation of the Memory Store.
    
    This implementation stores messages and dialogues in ArangoDB collections and
    uses ArangoDB's vector search capabilities for semantic retrieval.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
        db_name: str = "firecircle",
        collection_prefix: str = "firecircle_",
        vector_dimension: int = 1536,
        embedding_model: Any = None,
        create_if_not_exists: bool = True
    ):
        """
        Initialize the ArangoDB memory store.
        
        Args:
            host: ArangoDB server URL
            username: Database username
            password: Database password
            db_name: Name of the database to use
            collection_prefix: Prefix for all collections
            vector_dimension: Dimension of embedding vectors
            embedding_model: Model for generating embeddings
            create_if_not_exists: Whether to create database and collections if they don't exist
        """
        self.host = host
        self.username = username
        self.password = password
        self.db_name = db_name
        self.collection_prefix = collection_prefix
        self._vector_dimension = vector_dimension
        self._embedding_model = embedding_model
        self._create_if_not_exists = create_if_not_exists
        
        # Collection names
        self.message_collection_name = f"{collection_prefix}messages"
        self.dialogue_collection_name = f"{collection_prefix}dialogues"
        self.dialogue_message_collection_name = f"{collection_prefix}dialogue_messages"
        self.message_vector_collection_name = f"{collection_prefix}message_vectors"
        self.dialogue_vector_collection_name = f"{collection_prefix}dialogue_vectors"
        
        # Initialize connection
        self.client = None
        self.db = None
        self.message_collection = None
        self.dialogue_collection = None
        self.dialogue_message_collection = None
        self.message_vector_collection = None
        self.dialogue_vector_collection = None
        
        # Setup embedding function
        if self._embedding_model is None:
            # Simple random embeddings for testing
            self._get_embedding = lambda text: np.random.randn(self._vector_dimension).astype(np.float32)
        else:
            # Set up actual embedding function
            self._setup_embedding_function()
    
    def _setup_embedding_function(self):
        """Set up the embedding function based on the provided model."""
        # This would be implemented based on the embedding model
        # For now, use a placeholder
        self._get_embedding = lambda text: np.random.randn(self._vector_dimension).astype(np.float32)
    
    def connect(self) -> None:
        """
        Connect to ArangoDB and initialize collections.
        
        This method establishes a connection to ArangoDB, creates the database
        and collections if they don't exist, and sets up the necessary indexes.
        """
        try:
            # Connect to ArangoDB
            self.client = ArangoClient(hosts=self.host)
            sys_db = self.client.db("_system", username=self.username, password=self.password)
            
            # Create database if it doesn't exist
            if self._create_if_not_exists and not sys_db.has_database(self.db_name):
                sys_db.create_database(
                    self.db_name,
                    users=[{"username": self.username, "password": self.password, "active": True}]
                )
            
            # Connect to the database
            self.db = self.client.db(self.db_name, username=self.username, password=self.password)
            
            # Create collections if they don't exist
            self._create_collections()
            
            # Create indexes
            self._create_indexes()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ArangoDB: {e}")
    
    def _create_collections(self) -> None:
        """Create the necessary collections if they don't exist."""
        if not self._create_if_not_exists:
            # Just get existing collections
            self.message_collection = self.db.collection(self.message_collection_name)
            self.dialogue_collection = self.db.collection(self.dialogue_collection_name)
            self.dialogue_message_collection = self.db.collection(self.dialogue_message_collection_name)
            self.message_vector_collection = self.db.collection(self.message_vector_collection_name)
            self.dialogue_vector_collection = self.db.collection(self.dialogue_vector_collection_name)
            return
        
        # Create or get document collections
        for name in [
            self.message_collection_name,
            self.dialogue_collection_name
        ]:
            if not self.db.has_collection(name):
                self.db.create_collection(name)
        
        # Create or get edge collections
        for name in [self.dialogue_message_collection_name]:
            if not self.db.has_collection(name):
                self.db.create_collection(name, edge=True)
        
        # Create or get vector collections
        for name in [
            self.message_vector_collection_name,
            self.dialogue_vector_collection_name
        ]:
            if not self.db.has_collection(name):
                # Create collection with vector capabilities
                self.db.create_collection(name)
        
        # Assign collections to instance variables
        self.message_collection = self.db.collection(self.message_collection_name)
        self.dialogue_collection = self.db.collection(self.dialogue_collection_name)
        self.dialogue_message_collection = self.db.collection(self.dialogue_message_collection_name)
        self.message_vector_collection = self.db.collection(self.message_vector_collection_name)
        self.dialogue_vector_collection = self.db.collection(self.dialogue_vector_collection_name)
    
    def _create_indexes(self) -> None:
        """Create necessary indexes on the collections."""
        # Create index on dialogue_id in messages
        if not any(idx["type"] == "hash" and "dialogue_id" in idx["fields"] 
                 for idx in self.message_collection.indexes()):
            self.message_collection.add_hash_index(["metadata.dialogue_id"], unique=False)
        
        # Create indexes on the edge collection
        if not any(idx["type"] == "hash" and "_from" in idx["fields"] 
                 for idx in self.dialogue_message_collection.indexes()):
            self.dialogue_message_collection.add_hash_index(["_from"], unique=False)
        
        if not any(idx["type"] == "hash" and "_to" in idx["fields"] 
                 for idx in self.dialogue_message_collection.indexes()):
            self.dialogue_message_collection.add_hash_index(["_to"], unique=False)
        
        # Create vector indexes for semantic search
        # Note: This requires ArangoDB Enterprise or a version that supports vector indexes
        try:
            # For message vectors
            if not any(idx.get("type") == "vector" for idx in self.message_vector_collection.indexes()):
                self.message_vector_collection.add_persistent_index(
                    fields=["embedding"],
                    name="vector_idx_message",
                    unique=False,
                    sparse=False,
                    deduplicate=False,
                    estimates=True,
                    cleanup_interval_step=0,
                    cleanup_interval_step_unit="millisecond",
                    in_background=False,
                    weights={"embedding": 1.0}
                )
            
            # For dialogue vectors
            if not any(idx.get("type") == "vector" for idx in self.dialogue_vector_collection.indexes()):
                self.dialogue_vector_collection.add_persistent_index(
                    fields=["embedding"],
                    name="vector_idx_dialogue",
                    unique=False,
                    sparse=False,
                    deduplicate=False,
                    estimates=True,
                    cleanup_interval_step=0,
                    cleanup_interval_step_unit="millisecond",
                    in_background=False,
                    weights={"embedding": 1.0}
                )
        except Exception as e:
            print(f"Warning: Could not create vector indexes. Vector search may not be available: {e}")
    
    def _message_to_doc(self, message: Message) -> Dict[str, Any]:
        """
        Convert a Message object to an ArangoDB document.
        
        Args:
            message: The message to convert
            
        Returns:
            Dictionary representing the ArangoDB document
        """
        # Convert UUID to string
        message_dict = message.dict()
        message_dict["_key"] = str(message.id)
        
        # Convert any UUID fields to strings
        message_dict["id"] = str(message_dict["id"])
        message_dict["sender"] = str(message_dict["sender"])
        
        if message_dict.get("metadata", {}).get("dialogue_id"):
            message_dict["metadata"]["dialogue_id"] = str(message_dict["metadata"]["dialogue_id"])
        
        if message_dict.get("metadata", {}).get("in_response_to"):
            message_dict["metadata"]["in_response_to"] = str(message_dict["metadata"]["in_response_to"])
        
        if message_dict.get("metadata", {}).get("references"):
            message_dict["metadata"]["references"] = [str(r) for r in message_dict["metadata"]["references"]]
        
        if message_dict.get("metadata", {}).get("recipients"):
            message_dict["metadata"]["recipients"] = [str(r) for r in message_dict["metadata"]["recipients"]]
        
        return message_dict
    
    def _doc_to_message(self, doc: Dict[str, Any]) -> Message:
        """
        Convert an ArangoDB document to a Message object.
        
        Args:
            doc: The document to convert
            
        Returns:
            Message object
        """
        # Remove ArangoDB specific fields
        message_dict = doc.copy()
        for key in ["_key", "_id", "_rev"]:
            message_dict.pop(key, None)
        
        # Convert string IDs back to UUIDs
        message_dict["id"] = UUID(message_dict["id"])
        message_dict["sender"] = UUID(message_dict["sender"])
        
        if message_dict.get("metadata", {}).get("dialogue_id"):
            message_dict["metadata"]["dialogue_id"] = UUID(message_dict["metadata"]["dialogue_id"])
        
        if message_dict.get("metadata", {}).get("in_response_to"):
            message_dict["metadata"]["in_response_to"] = UUID(message_dict["metadata"]["in_response_to"])
        
        if message_dict.get("metadata", {}).get("references"):
            message_dict["metadata"]["references"] = [UUID(r) for r in message_dict["metadata"]["references"]]
        
        if message_dict.get("metadata", {}).get("recipients"):
            message_dict["metadata"]["recipients"] = [UUID(r) for r in message_dict["metadata"]["recipients"]]
        
        # Create Message object
        return Message(**message_dict)
    
    def _dialogue_to_doc(self, dialogue: Dialogue) -> Dict[str, Any]:
        """
        Convert a Dialogue object to an ArangoDB document.
        
        Args:
            dialogue: The dialogue to convert
            
        Returns:
            Dictionary representing the ArangoDB document
        """
        # Convert to dict and handle nested UUIDs
        dialogue_dict = dialogue.dict()
        dialogue_dict["_key"] = str(dialogue.id)
        
        # Convert ID fields to strings
        dialogue_dict["id"] = str(dialogue_dict["id"])
        
        # Convert participant IDs
        for i, participant in enumerate(dialogue_dict.get("participants", [])):
            participant["id"] = str(participant["id"])
            dialogue_dict["participants"][i] = participant
        
        # Remove messages - they're stored separately
        dialogue_dict.pop("messages", None)
        
        return dialogue_dict
    
    def _doc_to_dialogue(self, doc: Dict[str, Any], messages: List[Message] = None) -> Dialogue:
        """
        Convert an ArangoDB document to a Dialogue object.
        
        Args:
            doc: The document to convert
            messages: Optional list of messages for this dialogue
            
        Returns:
            Dialogue object
        """
        # Remove ArangoDB specific fields
        dialogue_dict = doc.copy()
        for key in ["_key", "_id", "_rev"]:
            dialogue_dict.pop(key, None)
        
        # Convert string IDs back to UUIDs
        dialogue_dict["id"] = UUID(dialogue_dict["id"])
        
        # Convert participant IDs
        for i, participant in enumerate(dialogue_dict.get("participants", [])):
            participant["id"] = UUID(participant["id"])
            dialogue_dict["participants"][i] = participant
        
        # Create Dialogue object
        dialogue = Dialogue(**dialogue_dict)
        
        # Add messages if provided
        if messages:
            dialogue.messages = messages
        
        return dialogue
    
    async def store_message(self, message: Message) -> bool:
        """
        Store a message in the ArangoDB store.
        
        Args:
            message: The message to store
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Convert to document
            message_doc = self._message_to_doc(message)
            
            # Store in message collection
            self.message_collection.insert(message_doc, overwrite=True)
            
            # Store message-dialogue relationship
            dialogue_id = message.metadata.dialogue_id
            if dialogue_id:
                edge = {
                    "_from": f"{self.dialogue_collection_name}/{str(dialogue_id)}",
                    "_to": f"{self.message_collection_name}/{str(message.id)}",
                    "sequence_number": message.metadata.sequence_number,
                    "turn_number": message.metadata.turn_number,
                    "timestamp": message.metadata.timestamp.isoformat() if message.metadata.timestamp else None
                }
                
                try:
                    self.dialogue_message_collection.insert(edge, overwrite=True)
                except Exception as e:
                    print(f"Warning: Failed to create dialogue-message edge: {e}")
            
            # Generate and store vector embedding
            text = message.content.text
            embedding = self._get_embedding(text)
            
            vector_doc = {
                "_key": str(message.id),
                "message_id": str(message.id),
                "dialogue_id": str(dialogue_id) if dialogue_id else None,
                "embedding": embedding.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.message_vector_collection.insert(vector_doc, overwrite=True)
            
            return True
            
        except Exception as e:
            print(f"Error storing message: {e}")
            return False
    
    async def store_dialogue(self, dialogue: Dialogue) -> bool:
        """
        Store a dialogue and all its messages in ArangoDB.
        
        Args:
            dialogue: The dialogue to store
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Convert to document
            dialogue_doc = self._dialogue_to_doc(dialogue)
            
            # Store in dialogue collection
            self.dialogue_collection.insert(dialogue_doc, overwrite=True)
            
            # Store all messages in the dialogue
            for message in dialogue.messages:
                await self.store_message(message)
            
            # Generate and store dialogue vector embedding
            all_text = " ".join([msg.content.text for msg in dialogue.messages])
            embedding = self._get_embedding(all_text)
            
            vector_doc = {
                "_key": str(dialogue.id),
                "dialogue_id": str(dialogue.id),
                "embedding": embedding.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.dialogue_vector_collection.insert(vector_doc, overwrite=True)
            
            return True
            
        except Exception as e:
            print(f"Error storing dialogue: {e}")
            return False
    
    async def get_message(self, message_id: UUID) -> Optional[Message]:
        """
        Retrieve a message by its ID.
        
        Args:
            message_id: The ID of the message to retrieve
            
        Returns:
            The Message object if found, None otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Retrieve from message collection
            doc = self.message_collection.get(str(message_id))
            if doc:
                return self._doc_to_message(doc)
            return None
            
        except DocumentGetError:
            return None
        except Exception as e:
            print(f"Error retrieving message: {e}")
            return None
    
    async def get_dialogue(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """
        Retrieve a dialogue by its ID.
        
        Args:
            dialogue_id: The ID of the dialogue to retrieve
            
        Returns:
            The Dialogue object if found, None otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Retrieve from dialogue collection
            doc = self.dialogue_collection.get(str(dialogue_id))
            if not doc:
                return None
            
            # Get messages for this dialogue
            messages = await self.get_dialogue_messages(dialogue_id)
            
            # Convert to Dialogue object
            return self._doc_to_dialogue(doc, messages)
            
        except DocumentGetError:
            return None
        except Exception as e:
            print(f"Error retrieving dialogue: {e}")
            return None
    
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
        if self.client is None:
            self.connect()
        
        try:
            # Build AQL query to get messages with proper sorting
            query = """
            FOR edge IN @@dialogue_message_collection
                FILTER edge._from == @dialogue_key
                LET message = DOCUMENT(edge._to)
                SORT edge.sequence_number ASC
                LIMIT @offset, @limit
                RETURN message
            """
            
            # Set up bind variables
            bind_vars = {
                "@dialogue_message_collection": self.dialogue_message_collection_name,
                "dialogue_key": f"{self.dialogue_collection_name}/{str(dialogue_id)}",
                "offset": offset or 0,
                "limit": limit or 1000
            }
            
            # Execute query
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            
            # Convert documents to Message objects
            messages = [self._doc_to_message(doc) for doc in cursor]
            
            return messages
            
        except Exception as e:
            print(f"Error retrieving dialogue messages: {e}")
            return []
    
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
        if self.client is None:
            self.connect()
        
        try:
            # Generate embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Build AQL query for vector search
            aql_query = """
            FOR vector IN @@vector_collection
            """
            
            # Add dialogue filter if provided
            if dialogue_id:
                aql_query += """
                FILTER vector.dialogue_id == @dialogue_id
                """
            
            # Continue with vector search
            aql_query += """
            LET distance = VECTOR_DISTANCE(vector.embedding, @query_embedding)
            SORT distance ASC
            LIMIT @limit
            LET message = DOCUMENT(CONCAT(@message_collection, '/', vector.message_id))
            RETURN message
            """
            
            # Set up bind variables
            bind_vars = {
                "@vector_collection": self.message_vector_collection_name,
                "@message_collection": self.message_collection_name,
                "query_embedding": query_embedding.tolist(),
                "limit": limit
            }
            
            if dialogue_id:
                bind_vars["dialogue_id"] = str(dialogue_id)
            
            # Execute query
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Convert documents to Message objects
            messages = [self._doc_to_message(doc) for doc in cursor]
            
            return messages
            
        except Exception as e:
            print(f"Error searching messages: {e}")
            print("Falling back to basic text search...")
            
            # Fallback to basic text search
            try:
                aql_query = """
                FOR message IN @@message_collection
                """
                
                if dialogue_id:
                    aql_query += """
                    FILTER message.metadata.dialogue_id == @dialogue_id
                    """
                
                aql_query += """
                FILTER CONTAINS(LOWER(message.content.text), LOWER(@query_text))
                SORT message.metadata.timestamp DESC
                LIMIT @limit
                RETURN message
                """
                
                bind_vars = {
                    "@message_collection": self.message_collection_name,
                    "query_text": query,
                    "limit": limit
                }
                
                if dialogue_id:
                    bind_vars["dialogue_id"] = str(dialogue_id)
                
                cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
                messages = [self._doc_to_message(doc) for doc in cursor]
                return messages
                
            except Exception as fallback_error:
                print(f"Fallback search also failed: {fallback_error}")
                return []
    
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
        if self.client is None:
            self.connect()
        
        try:
            # Generate embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Build AQL query for vector search
            aql_query = """
            FOR vector IN @@vector_collection
            LET distance = VECTOR_DISTANCE(vector.embedding, @query_embedding)
            SORT distance ASC
            LIMIT @limit
            LET dialogue = DOCUMENT(CONCAT(@dialogue_collection, '/', vector.dialogue_id))
            RETURN dialogue
            """
            
            # Set up bind variables
            bind_vars = {
                "@vector_collection": self.dialogue_vector_collection_name,
                "@dialogue_collection": self.dialogue_collection_name,
                "query_embedding": query_embedding.tolist(),
                "limit": limit
            }
            
            # Execute query
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Convert documents to Dialogue objects (without messages for performance)
            dialogues = []
            for doc in cursor:
                dialogue = self._doc_to_dialogue(doc)
                # Get just enough messages for context
                messages = await self.get_dialogue_messages(UUID(doc["id"]), limit=5)
                dialogue.messages = messages
                dialogues.append(dialogue)
            
            return dialogues
            
        except Exception as e:
            print(f"Error searching dialogues: {e}")
            print("Falling back to basic text search...")
            
            # Fallback to basic title search
            try:
                aql_query = """
                FOR dialogue IN @@dialogue_collection
                FILTER CONTAINS(LOWER(dialogue.title), LOWER(@query_text))
                SORT dialogue.updated_at DESC
                LIMIT @limit
                RETURN dialogue
                """
                
                bind_vars = {
                    "@dialogue_collection": self.dialogue_collection_name,
                    "query_text": query,
                    "limit": limit
                }
                
                cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
                
                dialogues = []
                for doc in cursor:
                    dialogue = self._doc_to_dialogue(doc)
                    # Get just enough messages for context
                    messages = await self.get_dialogue_messages(UUID(doc["id"]), limit=5)
                    dialogue.messages = messages
                    dialogues.append(dialogue)
                
                return dialogues
                
            except Exception as fallback_error:
                print(f"Fallback search also failed: {fallback_error}")
                return []
    
    async def update_message(self, message: Message) -> bool:
        """
        Update an existing message in the memory store.
        
        Args:
            message: The updated message
            
        Returns:
            True if update was successful, False otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Check if message exists
            if not self.message_collection.has(str(message.id)):
                return False
            
            # Convert to document
            message_doc = self._message_to_doc(message)
            
            # Update in message collection
            self.message_collection.replace(message_doc)
            
            # Update vector embedding
            text = message.content.text
            embedding = self._get_embedding(text)
            
            vector_doc = {
                "_key": str(message.id),
                "message_id": str(message.id),
                "dialogue_id": str(message.metadata.dialogue_id) if message.metadata.dialogue_id else None,
                "embedding": embedding.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.message_vector_collection.replace(vector_doc)
            
            # Check if we need to update dialogue embedding
            if message.metadata.dialogue_id:
                # Get all messages for this dialogue
                dialogue_messages = await self.get_dialogue_messages(message.metadata.dialogue_id)
                
                # Update dialogue embedding
                all_text = " ".join([msg.content.text for msg in dialogue_messages])
                dialogue_embedding = self._get_embedding(all_text)
                
                dialogue_vector_doc = {
                    "_key": str(message.metadata.dialogue_id),
                    "dialogue_id": str(message.metadata.dialogue_id),
                    "embedding": dialogue_embedding.tolist(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                self.dialogue_vector_collection.replace(dialogue_vector_doc)
            
            return True
            
        except Exception as e:
            print(f"Error updating message: {e}")
            return False
    
    async def delete_message(self, message_id: UUID) -> bool:
        """
        Delete a message from the memory store.
        
        Args:
            message_id: The ID of the message to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Check if message exists
            message_key = str(message_id)
            if not self.message_collection.has(message_key):
                return False
            
            # Get the message to find its dialogue
            message = await self.get_message(message_id)
            dialogue_id = message.metadata.dialogue_id if message else None
            
            # Delete message-dialogue edges
            if dialogue_id:
                aql_query = """
                FOR edge IN @@dialogue_message_collection
                    FILTER edge._to == @message_key
                    REMOVE edge IN @@dialogue_message_collection
                """
                
                bind_vars = {
                    "@dialogue_message_collection": self.dialogue_message_collection_name,
                    "message_key": f"{self.message_collection_name}/{message_key}"
                }
                
                self.db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Delete message vector
            if self.message_vector_collection.has(message_key):
                self.message_vector_collection.delete(message_key)
            
            # Delete the message
            self.message_collection.delete(message_key)
            
            # Update dialogue embedding if needed
            if dialogue_id:
                dialogue_messages = await self.get_dialogue_messages(dialogue_id)
                if dialogue_messages:
                    all_text = " ".join([msg.content.text for msg in dialogue_messages])
                    dialogue_embedding = self._get_embedding(all_text)
                    
                    dialogue_vector_doc = {
                        "_key": str(dialogue_id),
                        "dialogue_id": str(dialogue_id),
                        "embedding": dialogue_embedding.tolist(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    self.dialogue_vector_collection.replace(dialogue_vector_doc)
            
            return True
            
        except Exception as e:
            print(f"Error deleting message: {e}")
            return False
    
    async def delete_dialogue(self, dialogue_id: UUID) -> bool:
        """
        Delete a dialogue and all its messages from the memory store.
        
        Args:
            dialogue_id: The ID of the dialogue to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self.client is None:
            self.connect()
        
        try:
            # Check if dialogue exists
            dialogue_key = str(dialogue_id)
            if not self.dialogue_collection.has(dialogue_key):
                return False
            
            # Get all messages for this dialogue
            dialogue_messages = await self.get_dialogue_messages(dialogue_id)
            
            # Delete all messages
            for message in dialogue_messages:
                await self.delete_message(message.id)
            
            # Delete dialogue vector
            if self.dialogue_vector_collection.has(dialogue_key):
                self.dialogue_vector_collection.delete(dialogue_key)
            
            # Delete the dialogue
            self.dialogue_collection.delete(dialogue_key)
            
            return True
            
        except Exception as e:
            print(f"Error deleting dialogue: {e}")
            return False
    
    async def get_related_contexts(self, 
                                 message: Message, 
                                 max_contexts: int = 5) -> List[Message]:
        """
        Find messages that provide relevant context for a given message.
        
        Args:
            message: The message to find related contexts for
            max_contexts: Maximum number of contextual messages to return
            
        Returns:
            List of contextually relevant messages
        """
        if self.client is None:
            self.connect()
        
        try:
            # Generate embedding for the message
            message_embedding = self._get_embedding(message.content.text)
            
            # Build AQL query for vector search
            aql_query = """
            FOR vector IN @@vector_collection
            FILTER vector.message_id != @message_id
            """
            
            # Exclude messages from the same dialogue
            if message.metadata.dialogue_id:
                aql_query += """
                FILTER vector.dialogue_id != @dialogue_id
                """
            
            # Find closest vectors
            aql_query += """
            LET distance = VECTOR_DISTANCE(vector.embedding, @message_embedding)
            SORT distance ASC
            LIMIT @limit
            LET message = DOCUMENT(CONCAT(@message_collection, '/', vector.message_id))
            RETURN message
            """
            
            # Set up bind variables
            bind_vars = {
                "@vector_collection": self.message_vector_collection_name,
                "@message_collection": self.message_collection_name,
                "message_id": str(message.id),
                "message_embedding": message_embedding.tolist(),
                "limit": max_contexts
            }
            
            if message.metadata.dialogue_id:
                bind_vars["dialogue_id"] = str(message.metadata.dialogue_id)
            
            # Execute query
            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Convert documents to Message objects
            related_messages = [self._doc_to_message(doc) for doc in cursor]
            
            return related_messages
            
        except Exception as e:
            print(f"Error finding related contexts: {e}")
            
            # Fallback to text-based matching
            try:
                # Get some keywords from the message
                text = message.content.text
                words = text.split()
                query_words = [word for word in words if len(word) > 4][:5]  # Take up to 5 significant words
                
                if not query_words:
                    return []
                
                query_text = " ".join(query_words)
                
                # Do a simple text search
                return await self.search_messages(
                    query=query_text,
                    dialogue_id=None,  # Don't filter by dialogue
                    limit=max_contexts
                )
                
            except Exception as fallback_error:
                print(f"Fallback related contexts also failed: {fallback_error}")
                return []
    
    async def set_embedding_model(self, model: Any) -> None:
        """
        Set or update the embedding model.
        
        Args:
            model: The new embedding model to use
        """
        self._embedding_model = model
        self._setup_embedding_function()
    
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