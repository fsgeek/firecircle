"""
Comparative Dialogue Example for Fire Circle.

This example demonstrates how to run a dialogue between different AI models
(OpenAI and Anthropic) and compare their responses to the same prompts.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from firecircle.adapters.base.adapter import AdapterConfig, AdapterFactory
from firecircle.memory.inmemory import InMemoryStore
from firecircle.orchestrator.dialogue_manager import DialogueManager, DialogueConfig
from firecircle.orchestrator import TurnPolicy, DialogueState, DialoguePhase
from firecircle.protocol.message import (
    Message,
    MessageContent,
    MessageType,
    MessageRole,
    MessageMetadata,
    Participant
)


SYSTEM_PROMPT = """
You are participating in a comparative dialogue where different AI models discuss
the same topic from their unique perspectives. Please provide thoughtful, nuanced
responses that reflect your specific training approach and knowledge base.

When responding, consider:
1. What unique insights can you offer based on your specific training?
2. How might your perspective differ from other models?
3. Where might there be gaps or limitations in your approach?

Aim for intellectual honesty and epistemic humility, acknowledging both strengths
and limitations in your responses.
"""

TOPICS = [
    "How can AI systems best collaborate to overcome individual limitations?",
    "What are the ethical considerations in designing AI systems for collaborative decision-making?",
    "How can we ensure diversity of thought in AI collaborations while maintaining coherent outputs?",
    "What mechanisms might help AI systems recognize their own limitations and defer to others with complementary strengths?"
]


async def setup_adapters() -> Dict[str, any]:
    """Set up adapters for available AI models."""
    adapters = {}
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            openai_config = AdapterConfig(api_key=openai_api_key)
            openai_adapter = await AdapterFactory.create_adapter("openai", openai_config)
            adapters["openai"] = openai_adapter
            print("OpenAI adapter initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenAI adapter: {e}")
    
    # Check for Anthropic API key
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        try:
            anthropic_config = AdapterConfig(api_key=anthropic_api_key)
            anthropic_adapter = await AdapterFactory.create_adapter("anthropic", anthropic_config)
            adapters["anthropic"] = anthropic_adapter
            print("Anthropic adapter initialized successfully")
        except Exception as e:
            print(f"Error initializing Anthropic adapter: {e}")
    
    return adapters


async def create_participants(adapters: Dict[str, any]) -> List[Participant]:
    """Create participant objects for the dialogue."""
    participants = []
    
    # Human facilitator
    human_id = uuid.uuid4()
    participants.append(
        Participant(
            id=human_id,
            name="Human Facilitator",
            type="human"
        )
    )
    
    # AI participants
    if "openai" in adapters:
        openai_id = uuid.uuid4()
        participants.append(
            Participant(
                id=openai_id,
                name="GPT-4",
                type="ai_model",
                provider="openai",
                model="gpt-4-turbo",
                provider_config={"model_id": "gpt-4-turbo"},
                capabilities=["reasoning", "code", "knowledge-cutoff-2023"]
            )
        )
    
    if "anthropic" in adapters:
        anthropic_id = uuid.uuid4()
        participants.append(
            Participant(
                id=anthropic_id,
                name="Claude",
                type="ai_model",
                provider="anthropic",
                model="claude-3-opus-20240229",
                provider_config={"model_id": "claude-3-opus-20240229"},
                capabilities=["reasoning", "nuance", "knowledge-cutoff-2023"]
            )
        )
    
    return participants


async def message_handler(message: Message) -> None:
    """Handle incoming messages in the dialogue."""
    # Format based on role
    if message.role == MessageRole.SYSTEM:
        print(f"\n[System] {message.content.text}\n")
    elif message.role == MessageRole.USER:
        print(f"\n[Human] {message.content.text}\n")
    elif message.role == MessageRole.ASSISTANT:
        # Find the participant name for more context
        print(f"\n[{message.metadata.name}] {message.content.text}\n")
    else:
        print(f"\n[{message.role}] {message.content.text}\n")


async def send_ai_response(
    dialogue_manager: DialogueManager,
    dialogue_id: uuid.UUID,
    adapters: Dict[str, any],
    participant: Participant,
    topic: str
) -> bool:
    """
    Send a response from an AI participant.
    
    Args:
        dialogue_manager: The dialogue manager
        dialogue_id: ID of the dialogue
        adapters: Dictionary of AI adapters
        participant: The participant to respond
        topic: The topic being discussed
        
    Returns:
        True if successful, False otherwise
    """
    # Check if it's this participant's turn
    if not dialogue_manager.can_speak(dialogue_id, participant.id):
        print(f"Waiting for {participant.name}'s turn...")
        return False
    
    try:
        # Find the appropriate adapter
        adapter = adapters.get(participant.provider)
        if not adapter:
            print(f"No adapter found for {participant.provider}")
            return False
        
        # Create the message object
        ai_message = Message(
            id=uuid.uuid4(),
            type=MessageType.MESSAGE,
            role=MessageRole.ASSISTANT,
            sender=participant.id,
            content=MessageContent(text="...thinking..."),  # Placeholder
            metadata=MessageMetadata(
                dialogue_id=dialogue_id,
                sequence_number=len(dialogue_manager.active_dialogues[dialogue_id].messages) + 1,
                turn_number=dialogue_manager.active_dialogues[dialogue_id].messages[-1].metadata.turn_number + 1,
                timestamp=datetime.utcnow(),
                recipient_id=None,  # Response to everyone
                provider_config={
                    "temperature": 0.7,
                    "max_tokens": 750  # Limit response length
                }
            )
        )
        
        # Create a message for the AI to respond to
        prompt_message = Message(
            id=uuid.uuid4(),
            type=MessageType.MESSAGE,
            role=MessageRole.USER,
            sender=dialogue_manager.active_dialogues[dialogue_id].participants[0].id,  # Human
            content=MessageContent(text=f"Please respond to this topic: {topic}"),
            metadata=MessageMetadata(
                dialogue_id=dialogue_id,
                sequence_number=0,  # Not added to dialogue
                turn_number=0,  # Not added to dialogue
                timestamp=datetime.utcnow(),
                recipient_id=participant.id
            )
        )
        
        # Get response from the AI model
        print(f"Getting response from {participant.name}...")
        response = await adapter.send_message(prompt_message, dialogue_manager.active_dialogues[dialogue_id])
        
        # Update the message with the actual response
        ai_message.content.text = response.content.text
        
        # Process the message
        success, error = await dialogue_manager.process_message(ai_message)
        if not success:
            print(f"Error sending AI response: {error}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error getting {participant.name} response: {e}")
        return False


async def run_comparative_dialogue(topic_index: int = 0) -> None:
    """
    Run a comparative dialogue between different AI models.
    
    Args:
        topic_index: Index of the topic to discuss
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("Initializing Fire Circle comparative dialogue...")
    
    # Initialize components
    memory_store = InMemoryStore()
    dialogue_manager = DialogueManager(memory_store)
    
    # Set up adapters
    adapters = await setup_adapters()
    
    if len(adapters) < 2:
        print("Need at least two AI adapters for a comparative dialogue.")
        print("Please set both OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.")
        return
    
    # Create participants
    participants = await create_participants(adapters)
    facilitator_id = participants[0].id  # Human facilitator
    
    # Configure dialogue
    config = DialogueConfig(
        title="Comparative AI Dialogue",
        turn_policy=TurnPolicy.ROUND_ROBIN,
        randomize_initial_order=False,
        require_facilitator=True,
        initial_phase=DialoguePhase.EXPLORING
    )
    
    # Create dialogue
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id,
        system_message=SYSTEM_PROMPT
    )
    
    print(f"Dialogue created: {dialogue.id}")
    print(f"Participants: {', '.join([p.name for p in participants])}")
    
    # Register message handler
    dialogue_manager.register_message_handler(dialogue.id, message_handler)
    
    # Select topic
    topic = TOPICS[topic_index % len(TOPICS)]
    print(f"\nSelected topic: {topic}\n")
    
    # Send facilitator message
    facilitator_message = Message(
        id=uuid.uuid4(),
        type=MessageType.MESSAGE,
        role=MessageRole.USER,
        sender=facilitator_id,
        content=MessageContent(text=f"""
        Welcome to this comparative dialogue. Today, we'll explore the following topic:
        
        "{topic}"
        
        I'd like each of you to share your perspective on this topic, highlighting your
        unique insights and acknowledging any limitations in your approach.
        
        Let's begin with our AI participants taking turns to respond.
        """),
        metadata=MessageMetadata(
            dialogue_id=dialogue.id,
            sequence_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.sequence_number + 1,
            turn_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.turn_number + 1,
            timestamp=datetime.utcnow()
        )
    )
    
    success, error = await dialogue_manager.process_message(facilitator_message)
    if not success:
        print(f"Error sending facilitator message: {error}")
        return
    
    # Have each AI participant respond in turn
    for participant in participants[1:]:  # Skip facilitator
        success = await send_ai_response(
            dialogue_manager=dialogue_manager,
            dialogue_id=dialogue.id,
            adapters=adapters,
            participant=participant,
            topic=topic
        )
        if not success:
            print(f"Failed to get response from {participant.name}")
    
    # Facilitator summary
    current_speaker = dialogue_manager.get_current_speaker(dialogue.id)
    if current_speaker == facilitator_id:
        # Summary message
        summary_message = Message(
            id=uuid.uuid4(),
            type=MessageType.SUMMARY,
            role=MessageRole.USER,
            sender=facilitator_id,
            content=MessageContent(text=f"""
            Thank you both for your perspectives on "{topic}".
            
            We've seen both similarities and differences in your approaches. This illustrates
            the value of having multiple AI systems collaborate on complex topics, where each
            can bring unique strengths and different analytical frameworks.
            
            This dialogue demonstrates the Fire Circle system's ability to facilitate
            structured exchanges between different AI models, creating a space for
            reciprocal sharing of insights.
            """),
            metadata=MessageMetadata(
                dialogue_id=dialogue.id,
                sequence_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.sequence_number + 1,
                turn_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.turn_number + 1,
                timestamp=datetime.utcnow()
            )
        )
        
        # Change dialogue phase
        await dialogue_manager.change_dialogue_phase(dialogue.id, DialoguePhase.SUMMARIZATION)
        
        success, error = await dialogue_manager.process_message(summary_message)
        if not success:
            print(f"Error sending summary message: {error}")
    
    # Complete the dialogue
    await dialogue_manager.change_dialogue_state(dialogue.id, DialogueState.CONCLUDING)
    await dialogue_manager.close_dialogue(dialogue.id)
    
    # Display information about memory store
    try:
        dialogues = await memory_store.search_dialogues(topic, limit=1)
        if dialogues:
            print(f"\nDialogue successfully stored in memory store with ID: {dialogues[0].id}")
            print(f"Title: {dialogues[0].title}")
            print(f"Message count: {len(dialogues[0].messages)}")
    except Exception as e:
        print(f"Error accessing memory store: {e}")
    
    print("\nComparative dialogue example completed.")


async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a comparative dialogue between AI models")
    parser.add_argument(
        "--topic", 
        type=int, 
        default=0,
        help="Topic index (0-3) to discuss"
    )
    
    args = parser.parse_args()
    
    await run_comparative_dialogue(topic_index=args.topic)


if __name__ == "__main__":
    asyncio.run(main())