"""
Tests for the Memory Store implementations.

This module contains tests for the various Memory Store implementations
to ensure they correctly store, retrieve, and search messages and dialogues.
"""

import asyncio
import pytest
import uuid
from datetime import datetime

from firecircle.memory import InMemoryStore, VectorMemoryStore, MemoryStore
from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    MessageType, 
    MessageRole, 
    MessageMetadata,
    Participant,
    Dialogue
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
def memory_stores():
    """Fixture to provide all memory store implementations for testing."""
    return [
        InMemoryStore(),
        VectorMemoryStore()
    ]


@pytest.mark.asyncio
async def test_store_retrieve_message(memory_stores):
    """Test storing and retrieving a message."""
    for store in memory_stores:
        # Create a test message
        dialogue_id = uuid.uuid4()
        message = create_test_message(dialogue_id)
        
        # Store the message
        result = await store.store_message(message)
        assert result is True, f"Failed to store message in {store.__class__.__name__}"
        
        # Retrieve the message
        retrieved = await store.get_message(message.id)
        assert retrieved is not None, f"Failed to retrieve message in {store.__class__.__name__}"
        assert retrieved.id == message.id, f"Retrieved message has incorrect ID in {store.__class__.__name__}"
        assert retrieved.content.text == message.content.text, f"Retrieved message has incorrect content in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_store_retrieve_dialogue(memory_stores):
    """Test storing and retrieving a dialogue."""
    for store in memory_stores:
        # Create a test dialogue
        dialogue = create_test_dialogue()
        
        # Store the dialogue
        result = await store.store_dialogue(dialogue)
        assert result is True, f"Failed to store dialogue in {store.__class__.__name__}"
        
        # Retrieve the dialogue
        retrieved = await store.get_dialogue(dialogue.id)
        assert retrieved is not None, f"Failed to retrieve dialogue in {store.__class__.__name__}"
        assert retrieved.id == dialogue.id, f"Retrieved dialogue has incorrect ID in {store.__class__.__name__}"
        assert len(retrieved.messages) == len(dialogue.messages), f"Retrieved dialogue has incorrect number of messages in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_get_dialogue_messages(memory_stores):
    """Test retrieving messages from a dialogue."""
    for store in memory_stores:
        # Create a test dialogue
        dialogue = create_test_dialogue(num_messages=5)
        
        # Store the dialogue
        await store.store_dialogue(dialogue)
        
        # Retrieve all messages
        messages = await store.get_dialogue_messages(dialogue.id)
        assert len(messages) == 5, f"Incorrect number of messages retrieved in {store.__class__.__name__}"
        
        # Test limit
        limited = await store.get_dialogue_messages(dialogue.id, limit=3)
        assert len(limited) == 3, f"Limit not applied correctly in {store.__class__.__name__}"
        
        # Test offset
        offset = await store.get_dialogue_messages(dialogue.id, offset=2)
        assert len(offset) == 3, f"Offset not applied correctly in {store.__class__.__name__}"
        
        # Test limit and offset
        limited_offset = await store.get_dialogue_messages(dialogue.id, limit=2, offset=2)
        assert len(limited_offset) == 2, f"Limit and offset not applied correctly in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_search_messages(memory_stores):
    """Test searching for messages."""
    for store in memory_stores:
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
            await store.store_message(message)
        
        # Search for messages about Python
        results = await store.search_messages("Python programming")
        assert len(results) > 0, f"No search results found in {store.__class__.__name__}"
        
        # Search within a specific dialogue
        dialogue_results = await store.search_messages("Python", dialogue_id=dialogue_id)
        assert len(dialogue_results) > 0, f"No dialogue-specific search results found in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_search_dialogues(memory_stores):
    """Test searching for dialogues."""
    for store in memory_stores:
        # Create dialogues with different content
        dialogues = [
            create_test_dialogue(),  # Default messages
            create_test_dialogue()   # Default messages
        ]
        
        # Customize dialogue messages
        dialogues[0].messages[0].content.text = "Discussion about artificial intelligence and machine learning"
        dialogues[1].messages[0].content.text = "Conversation about programming languages and software development"
        
        # Store the dialogues
        for dialogue in dialogues:
            await store.store_dialogue(dialogue)
        
        # Search for dialogues about AI
        results = await store.search_dialogues("artificial intelligence")
        assert len(results) > 0, f"No search results found in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_update_message(memory_stores):
    """Test updating a message."""
    for store in memory_stores:
        # Create and store a message
        dialogue_id = uuid.uuid4()
        message = create_test_message(dialogue_id, text="Original content")
        await store.store_message(message)
        
        # Update the message
        message.content.text = "Updated content"
        result = await store.update_message(message)
        assert result is True, f"Failed to update message in {store.__class__.__name__}"
        
        # Verify the update
        retrieved = await store.get_message(message.id)
        assert retrieved.content.text == "Updated content", f"Message update not applied in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_delete_message(memory_stores):
    """Test deleting a message."""
    for store in memory_stores:
        # Create and store a message
        dialogue_id = uuid.uuid4()
        message = create_test_message(dialogue_id)
        await store.store_message(message)
        
        # Delete the message
        result = await store.delete_message(message.id)
        assert result is True, f"Failed to delete message in {store.__class__.__name__}"
        
        # Verify the deletion
        retrieved = await store.get_message(message.id)
        assert retrieved is None, f"Message not deleted in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_delete_dialogue(memory_stores):
    """Test deleting a dialogue."""
    for store in memory_stores:
        # Create and store a dialogue
        dialogue = create_test_dialogue()
        await store.store_dialogue(dialogue)
        
        # Delete the dialogue
        result = await store.delete_dialogue(dialogue.id)
        assert result is True, f"Failed to delete dialogue in {store.__class__.__name__}"
        
        # Verify the dialogue deletion
        retrieved_dialogue = await store.get_dialogue(dialogue.id)
        assert retrieved_dialogue is None, f"Dialogue not deleted in {store.__class__.__name__}"
        
        # Verify message deletion
        for message in dialogue.messages:
            retrieved_message = await store.get_message(message.id)
            assert retrieved_message is None, f"Message not deleted with dialogue in {store.__class__.__name__}"


@pytest.mark.asyncio
async def test_get_related_contexts(memory_stores):
    """Test getting related contexts for a message."""
    for store in memory_stores:
        # Create messages in different dialogues
        dialogue1_id = uuid.uuid4()
        dialogue2_id = uuid.uuid4()
        
        # Messages in first dialogue
        messages1 = [
            create_test_message(dialogue1_id, 1, "Python is a versatile programming language"),
            create_test_message(dialogue1_id, 2, "Fire Circle enables dialogue between AI models")
        ]
        
        # Messages in second dialogue
        messages2 = [
            create_test_message(dialogue2_id, 1, "JavaScript is used for web development"),
            create_test_message(dialogue2_id, 2, "Python and JavaScript are both popular languages")
        ]
        
        # Store all messages
        for message in messages1 + messages2:
            await store.store_message(message)
        
        # Get related contexts for a message about Python
        query = create_test_message(uuid.uuid4(), text="I need help with Python programming")
        contexts = await store.get_related_contexts(query)
        
        assert len(contexts) > 0, f"No related contexts found in {store.__class__.__name__}"


# Add more tests as needed:
# - Test error handling
# - Test edge cases (empty messages, etc.)
# - Test performance with larger datasets
# - Test concurrent access