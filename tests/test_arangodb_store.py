"""
Tests for the ArangoDB Memory Store.

This module contains tests for the ArangoDB implementation of the Memory Store,
which can be run when an ArangoDB instance is available.
"""

import asyncio
import os
import pytest
import uuid
from datetime import datetime

from firecircle.memory.arangodb_store import ArangoDBStore
from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    MessageType, 
    MessageRole, 
    MessageMetadata,
    Participant,
    Dialogue
)


# Skip all tests if ArangoDB is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_ARANGODB_HOST"),
    reason="ArangoDB host not specified, skipping ArangoDB tests"
)


def create_test_message(dialogue_id, sequence_number=1, text="Test message"):
    """Helper function to create a test message."""
    return Message(
        id=uuid.uuid4(),
        type=MessageType.MESSAGE,
        role=MessageRole.ASSISTANT,
        sender=uuid.uuid4(),
        content=MessageContent(text=text),
        metadata=MessageMetadata(
            dialogue_id=dialogue_id,
            sequence_number=sequence_number,
            turn_number=sequence_number,
            timestamp=datetime.utcnow()
        )
    )


def create_test_dialogue(num_messages=3):
    """Helper function to create a test dialogue with messages."""
    dialogue_id = uuid.uuid4()
    
    # Create participants
    participants = [
        Participant(id=uuid.uuid4(), name="User", type="human"),
        Participant(id=uuid.uuid4(), name="Assistant", type="ai_model", provider="openai")
    ]
    
    # Create messages
    messages = [
        create_test_message(
            dialogue_id=dialogue_id, 
            sequence_number=i+1, 
            text=f"Message {i+1} in dialogue"
        ) for i in range(num_messages)
    ]
    
    # Create dialogue
    dialogue = Dialogue(
        id=dialogue_id,
        title="Test Dialogue",
        participants=participants,
        messages=messages,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    return dialogue


@pytest.fixture
async def arango_store():
    """Fixture to provide an ArangoDB memory store for testing."""
    # Get connection details from environment variables
    host = os.getenv("TEST_ARANGODB_HOST", "http://localhost:8529")
    username = os.getenv("TEST_ARANGODB_USERNAME", "root")
    password = os.getenv("TEST_ARANGODB_PASSWORD", "")
    db_name = os.getenv("TEST_ARANGODB_DBNAME", "firecircle_test")
    
    # Create a test-specific collection prefix to avoid conflicts
    test_prefix = f"test_{uuid.uuid4().hex[:8]}_"
    
    # Initialize store
    store = ArangoDBStore(
        host=host,
        username=username,
        password=password,
        db_name=db_name,
        collection_prefix=test_prefix,
        create_if_not_exists=True
    )
    
    # Connect to ensure setup is ready
    store.connect()
    
    # Return the store for testing
    yield store
    
    # Clean up after tests
    try:
        # Delete test collections
        for collection_name in [
            store.message_collection_name,
            store.dialogue_collection_name,
            store.dialogue_message_collection_name,
            store.message_vector_collection_name,
            store.dialogue_vector_collection_name
        ]:
            if store.db.has_collection(collection_name):
                store.db.delete_collection(collection_name)
    except Exception as e:
        print(f"Error cleaning up test collections: {e}")


@pytest.mark.asyncio
async def test_store_retrieve_message(arango_store):
    """Test storing and retrieving a message."""
    # Create a test message
    dialogue_id = uuid.uuid4()
    message = create_test_message(dialogue_id)
    
    # Store the message
    result = await arango_store.store_message(message)
    assert result is True, "Failed to store message"
    
    # Retrieve the message
    retrieved = await arango_store.get_message(message.id)
    assert retrieved is not None, "Failed to retrieve message"
    assert retrieved.id == message.id, "Retrieved message has incorrect ID"
    assert retrieved.content.text == message.content.text, "Retrieved message has incorrect content"


@pytest.mark.asyncio
async def test_store_retrieve_dialogue(arango_store):
    """Test storing and retrieving a dialogue."""
    # Create a test dialogue
    dialogue = create_test_dialogue()
    
    # Store the dialogue
    result = await arango_store.store_dialogue(dialogue)
    assert result is True, "Failed to store dialogue"
    
    # Retrieve the dialogue
    retrieved = await arango_store.get_dialogue(dialogue.id)
    assert retrieved is not None, "Failed to retrieve dialogue"
    assert retrieved.id == dialogue.id, "Retrieved dialogue has incorrect ID"
    assert len(retrieved.messages) == len(dialogue.messages), "Retrieved dialogue has incorrect number of messages"


@pytest.mark.asyncio
async def test_get_dialogue_messages(arango_store):
    """Test retrieving messages from a dialogue."""
    # Create a test dialogue
    dialogue = create_test_dialogue(num_messages=5)
    
    # Store the dialogue
    await arango_store.store_dialogue(dialogue)
    
    # Retrieve all messages
    messages = await arango_store.get_dialogue_messages(dialogue.id)
    assert len(messages) == 5, "Incorrect number of messages retrieved"
    
    # Test limit
    limited = await arango_store.get_dialogue_messages(dialogue.id, limit=3)
    assert len(limited) == 3, "Limit not applied correctly"
    
    # Test offset
    offset = await arango_store.get_dialogue_messages(dialogue.id, offset=2)
    assert len(offset) == 3, "Offset not applied correctly"
    
    # Test limit and offset
    limited_offset = await arango_store.get_dialogue_messages(dialogue.id, limit=2, offset=2)
    assert len(limited_offset) == 2, "Limit and offset not applied correctly"


@pytest.mark.asyncio
async def test_search_messages(arango_store):
    """Test searching for messages."""
    # Create messages with different content
    dialogue_id = uuid.uuid4()
    messages = [
        create_test_message(dialogue_id, 1, "Python is a programming language"),
        create_test_message(dialogue_id, 2, "Fire Circle enables meaningful dialogue between AI models"),
        create_test_message(dialogue_id, 3, "The concept of Ayni represents reciprocity"),
        create_test_message(dialogue_id, 4, "Programming in Python is fun and productive")
    ]
    
    # Store the messages
    for message in messages:
        await arango_store.store_message(message)
    
    # Search for messages
    results = await arango_store.search_messages("Python programming")
    assert len(results) > 0, "No search results found"
    
    # Search within a specific dialogue
    dialogue_results = await arango_store.search_messages("Python", dialogue_id=dialogue_id)
    assert len(dialogue_results) > 0, "No dialogue-specific search results found"


@pytest.mark.asyncio
async def test_update_message(arango_store):
    """Test updating a message."""
    # Create and store a message
    dialogue_id = uuid.uuid4()
    message = create_test_message(dialogue_id, text="Original content")
    await arango_store.store_message(message)
    
    # Update the message
    message.content.text = "Updated content"
    result = await arango_store.update_message(message)
    assert result is True, "Failed to update message"
    
    # Verify the update
    retrieved = await arango_store.get_message(message.id)
    assert retrieved.content.text == "Updated content", "Message update not applied"


@pytest.mark.asyncio
async def test_delete_message(arango_store):
    """Test deleting a message."""
    # Create and store a message
    dialogue_id = uuid.uuid4()
    message = create_test_message(dialogue_id)
    await arango_store.store_message(message)
    
    # Delete the message
    result = await arango_store.delete_message(message.id)
    assert result is True, "Failed to delete message"
    
    # Verify the deletion
    retrieved = await arango_store.get_message(message.id)
    assert retrieved is None, "Message not deleted"


@pytest.mark.asyncio
async def test_delete_dialogue(arango_store):
    """Test deleting a dialogue."""
    # Create and store a dialogue
    dialogue = create_test_dialogue()
    await arango_store.store_dialogue(dialogue)
    
    # Delete the dialogue
    result = await arango_store.delete_dialogue(dialogue.id)
    assert result is True, "Failed to delete dialogue"
    
    # Verify the dialogue deletion
    retrieved_dialogue = await arango_store.get_dialogue(dialogue.id)
    assert retrieved_dialogue is None, "Dialogue not deleted"
    
    # Verify message deletion
    for message in dialogue.messages:
        retrieved_message = await arango_store.get_message(message.id)
        assert retrieved_message is None, "Message not deleted with dialogue"


@pytest.mark.asyncio
async def test_related_contexts(arango_store):
    """Test getting related contexts for a message."""
    # Create dialogues with different content
    dialogue1 = create_test_dialogue(num_messages=2)
    dialogue1.messages[0].content.text = "Python is a versatile programming language"
    dialogue1.messages[1].content.text = "Fire Circle enables dialogue between AI models"
    
    dialogue2 = create_test_dialogue(num_messages=2)
    dialogue2.messages[0].content.text = "JavaScript is used for web development"
    dialogue2.messages[1].content.text = "Python and JavaScript are both popular languages"
    
    # Store the dialogues
    await arango_store.store_dialogue(dialogue1)
    await arango_store.store_dialogue(dialogue2)
    
    # Create a test message
    query_message = create_test_message(
        dialogue_id=uuid.uuid4(),
        text="I need help with Python programming"
    )
    
    # Get related contexts
    related = await arango_store.get_related_contexts(query_message)
    
    # Should find at least one related message
    assert len(related) > 0, "No related contexts found"