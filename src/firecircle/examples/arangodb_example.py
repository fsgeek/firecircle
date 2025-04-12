"""
ArangoDB Memory Store Example for Fire Circle.

This example demonstrates how to use the ArangoDB implementation of 
the Memory Store for persisting dialogues and performing semantic search.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from firecircle.protocol.message import (
    Message,
    MessageContent,
    MessageType,
    MessageRole,
    MessageMetadata,
    Participant,
    Dialogue
)

# Import conditionally to handle missing dependencies
try:
    from firecircle.memory.arangodb_store import ArangoDBStore
except ImportError:
    print("Error: ArangoDB dependencies not installed.")
    print("Install them with: pip install firecircle[arangodb]")
    sys.exit(1)


def create_test_message(dialogue_id, sender_id, text="Test message"):
    """Helper function to create a test message."""
    return Message(
        id=uuid.uuid4(),
        type=MessageType.MESSAGE,
        role=MessageRole.USER,
        sender=sender_id,
        content=MessageContent(text=text),
        metadata=MessageMetadata(
            dialogue_id=dialogue_id,
            sequence_number=1,
            turn_number=1,
            timestamp=datetime.utcnow()
        )
    )


async def setup_store():
    """Set up the ArangoDB memory store."""
    # Get connection details from environment variables or use defaults
    host = os.environ.get("ARANGODB_HOST", "http://localhost:8529")
    username = os.environ.get("ARANGODB_USERNAME", "root")
    password = os.environ.get("ARANGODB_PASSWORD", "")
    db_name = os.environ.get("ARANGODB_DBNAME", "firecircle")
    
    print(f"Connecting to ArangoDB at {host}...")
    
    # Create store
    store = ArangoDBStore(
        host=host,
        username=username,
        password=password,
        db_name=db_name,
        collection_prefix="firecircle_",
        create_if_not_exists=True
    )
    
    # Initialize connection
    try:
        store.connect()
        print("Connected to ArangoDB successfully")
        return store
    except Exception as e:
        print(f"Error connecting to ArangoDB: {e}")
        return None


async def demo_basic_operations(store):
    """Demonstrate basic Memory Store operations."""
    print("\n--- Basic Operations ---")
    
    # Create a simple dialogue
    dialogue_id = uuid.uuid4()
    user_id = uuid.uuid4()
    ai_id = uuid.uuid4()
    
    participants = [
        Participant(id=user_id, name="User", type="human"),
        Participant(id=ai_id, name="Assistant", type="ai_model", provider="openai")
    ]
    
    # Create some messages
    user_message = create_test_message(
        dialogue_id=dialogue_id,
        sender_id=user_id,
        text="What is Ayni and how does it relate to reciprocity?"
    )
    
    ai_message = create_test_message(
        dialogue_id=dialogue_id,
        sender_id=ai_id,
        text="Ayni is a concept from Andean cultures that embodies reciprocity. It represents the idea that everything in life is connected through mutual exchanges of energy and support. In practice, Ayni means that what you give, you will receive in return - creating balance and harmony in communities and ecological systems."
    )
    
    # Create dialogue
    dialogue = Dialogue(
        id=dialogue_id,
        title="Understanding Ayni",
        participants=participants,
        messages=[user_message, ai_message],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Store dialogue
    print("Storing dialogue...")
    result = await store.store_dialogue(dialogue)
    print(f"Store result: {result}")
    
    # Retrieve dialogue
    print("\nRetrieving dialogue...")
    retrieved = await store.get_dialogue(dialogue_id)
    print(f"Retrieved dialogue: {retrieved.title}")
    print(f"Number of messages: {len(retrieved.messages)}")
    print(f"First message: {retrieved.messages[0].content.text}")
    
    # Update a message
    print("\nUpdating message...")
    ai_message.content.text += "\n\nThe Fire Circle project embodies Ayni by creating a system where multiple AI models contribute their unique perspectives in a balanced exchange."
    update_result = await store.update_message(ai_message)
    print(f"Update result: {update_result}")
    
    # Retrieve updated message
    updated = await store.get_message(ai_message.id)
    print(f"Updated message length: {len(updated.content.text)} characters")


async def demo_search_capabilities(store):
    """Demonstrate search capabilities of the Memory Store."""
    print("\n--- Search Capabilities ---")
    
    # Create another dialogue with different content
    dialogue_id = uuid.uuid4()
    user_id = uuid.uuid4()
    ai_id = uuid.uuid4()
    
    participants = [
        Participant(id=user_id, name="User", type="human"),
        Participant(id=ai_id, name="Assistant", type="ai_model", provider="anthropic")
    ]
    
    # Create messages about a different topic
    user_message = create_test_message(
        dialogue_id=dialogue_id,
        sender_id=user_id,
        text="How can multiple AI models collaborate effectively?"
    )
    
    ai_message = create_test_message(
        dialogue_id=dialogue_id,
        sender_id=ai_id,
        text="Multiple AI models can collaborate effectively through structured dialogue protocols that leverage their complementary strengths. By implementing turn-taking mechanisms, shared context, and explicit role assignment, models can contribute their unique capabilities while mitigating individual limitations. This approach enables more balanced and comprehensive insights than any single model working alone."
    )
    
    # Create dialogue
    dialogue = Dialogue(
        id=dialogue_id,
        title="AI Collaboration Techniques",
        participants=participants,
        messages=[user_message, ai_message],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Store second dialogue
    await store.store_dialogue(dialogue)
    
    # Search for messages about reciprocity
    print("\nSearching for messages about reciprocity...")
    reciprocity_results = await store.search_messages("reciprocity Ayni")
    print(f"Found {len(reciprocity_results)} messages")
    if reciprocity_results:
        print(f"First result snippet: {reciprocity_results[0].content.text[:100]}...")
    
    # Search for messages about collaboration
    print("\nSearching for messages about collaboration...")
    collab_results = await store.search_messages("collaboration AI models")
    print(f"Found {len(collab_results)} messages")
    if collab_results:
        print(f"First result snippet: {collab_results[0].content.text[:100]}...")
    
    # Search for dialogues
    print("\nSearching for dialogues about AI...")
    dialogue_results = await store.search_dialogues("AI collaboration")
    print(f"Found {len(dialogue_results)} dialogues")
    for d in dialogue_results:
        print(f"- {d.title}")


async def clean_up(store, permanent=False):
    """Optional cleanup of test data."""
    if permanent and input("\nDelete test data? (y/n): ").lower() == 'y':
        print("Cleaning up test data...")
        # This would delete all data in the database
        # In a real application, you would be more selective
        for collection in [
            store.message_collection,
            store.dialogue_collection,
            store.dialogue_message_collection,
            store.message_vector_collection,
            store.dialogue_vector_collection
        ]:
            collection.truncate()
        print("Test data deleted")
    else:
        print("\nTest data remains in the database for inspection")


async def main():
    """Main entry point for the example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("ArangoDB Memory Store Example")
    print("============================")
    
    # Set up store
    store = await setup_store()
    if not store:
        return
    
    # Run demonstrations
    await demo_basic_operations(store)
    await demo_search_capabilities(store)
    
    # Optional cleanup
    await clean_up(store, permanent=False)
    
    print("\nExample completed")


if __name__ == "__main__":
    asyncio.run(main())