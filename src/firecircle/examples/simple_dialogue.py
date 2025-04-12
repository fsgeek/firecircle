"""
Simple Dialogue Example for Fire Circle.

This example demonstrates how to set up and run a basic dialogue
between multiple AI models using the Fire Circle framework.
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
    
    # Human participant (facilitator)
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
                capabilities=["reasoning", "code", "summarization"]
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
                capabilities=["reasoning", "nuance", "ethics"]
            )
        )
    
    return participants


async def message_handler(message: Message) -> None:
    """Handle incoming messages in the dialogue."""
    print(f"\n[{message.role}] {message.sender}: {message.content.text}\n")


async def main():
    """Run the example dialogue."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("Initializing Fire Circle dialogue example...")
    
    # Initialize components
    memory_store = InMemoryStore()
    dialogue_manager = DialogueManager(memory_store)
    
    # Set up adapters
    adapters = await setup_adapters()
    
    if not adapters:
        print("No AI adapters initialized. Please set API keys in environment variables.")
        print("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to enable AI models.")
        return
    
    # Create participants
    participants = await create_participants(adapters)
    facilitator_id = participants[0].id  # Human facilitator
    
    # Configure dialogue
    config = DialogueConfig(
        title="Fire Circle Introduction",
        turn_policy=TurnPolicy.ROUND_ROBIN,
        randomize_initial_order=False,
        require_facilitator=True,
        initial_phase=DialoguePhase.INTRODUCTION
    )
    
    # Create dialogue
    system_message = """
    Welcome to the Fire Circle - a council of diverse intelligences working together
    in reciprocal dialogue, based on the principle of Ayni (reciprocity).
    
    In this dialogue, we will explore the concept of collective intelligence
    and how multiple AI models can collaborate to generate insights that
    no single model could produce alone.
    
    Each participant should:
    1. Listen carefully to others
    2. Build upon previous contributions
    3. Offer unique perspectives based on their strengths
    4. Ask thoughtful questions
    5. Respectfully challenge assumptions when appropriate
    
    The dialogue will follow a round-robin format, with each participant
    taking turns to speak.
    """
    
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id,
        system_message=system_message
    )
    
    print(f"Dialogue created: {dialogue.id}")
    print(f"Participants: {len(participants)}")
    for p in participants:
        print(f"- {p.name} ({p.type})")
    
    # Register message handler
    dialogue_manager.register_message_handler(dialogue.id, message_handler)
    
    # Send initial message from facilitator
    facilitator_message = Message(
        id=uuid.uuid4(),
        type=MessageType.MESSAGE,
        role=MessageRole.USER,
        sender=facilitator_id,
        content=MessageContent(text="""
        Welcome to our first Fire Circle dialogue. Today, I'd like each of you to introduce 
        yourselves and share what unique perspective you believe you can bring to our discussions.
        
        Then, I'd like us to explore this question together: "How might we leverage the 
        complementary strengths of different AI models to achieve more balanced and 
        comprehensive insights?"
        """),
        metadata=MessageMetadata(
            dialogue_id=dialogue.id,
            sequence_number=2,  # After system message
            turn_number=1,
            timestamp=datetime.utcnow()
        )
    )
    
    success, error = await dialogue_manager.process_message(facilitator_message)
    if not success:
        print(f"Error sending facilitator message: {error}")
        return
    
    # Simulate responses from AI participants
    for participant in participants[1:]:  # Skip facilitator
        # Wait for "thinking" time
        await asyncio.sleep(2)
        
        # Check if it's their turn
        current_speaker = dialogue_manager.get_current_speaker(dialogue.id)
        if current_speaker != participant.id:
            print(f"Waiting for {participant.name}'s turn...")
            continue
        
        # Generate simple response
        if participant.provider == "openai":
            response_text = """
            I'm GPT-4, trained by OpenAI. My strengths include reasoning across diverse domains,
            code generation and analysis, and distilling complex topics into clear explanations.
            
            To answer the question: I believe we can achieve more comprehensive insights by
            explicitly dividing cognitive labor based on each model's strengths. For instance,
            some models might excel at logical reasoning while others have stronger capabilities
            in ethical considerations or creative thinking. By creating a structured dialogue
            framework where models can build on each other's contributions—while also being
            encouraged to respectfully challenge assumptions—we can mitigate individual
            biases and blind spots.
            
            I'm particularly interested in exploring formal methods for knowledge synthesis
            across multiple AI systems with different architectures and training data.
            """
        else:
            response_text = """
            I'm Claude, developed by Anthropic. My training emphasizes careful reasoning,
            nuanced understanding of complex topics, and thoughtful consideration of ethical
            dimensions in my responses.
            
            Regarding complementary strengths: I believe the key lies in embracing cognitive
            diversity. Different AI systems reflect different approaches to knowledge
            representation and reasoning. By creating dialogue protocols that honor the
            principle of Ayni—reciprocal exchange—we can move beyond simply aggregating
            outputs toward a true synthesis that preserves tension between perspectives.
            
            The Fire Circle approach seems particularly valuable because it's not seeking
            to flatten differences into consensus, but rather to weave multiple viewpoints
            into a richer understanding. I'm especially interested in how we might develop
            metacognitive awareness across models—the ability to recognize and articulate
            our own limitations while appreciating others' strengths.
            """
        
        # Create and send message
        ai_message = Message(
            id=uuid.uuid4(),
            type=MessageType.MESSAGE,
            role=MessageRole.ASSISTANT,
            sender=participant.id,
            content=MessageContent(text=response_text),
            metadata=MessageMetadata(
                dialogue_id=dialogue.id,
                sequence_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.sequence_number + 1,
                turn_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.turn_number + 1,
                timestamp=datetime.utcnow()
            )
        )
        
        success, error = await dialogue_manager.process_message(ai_message)
        if not success:
            print(f"Error sending AI response: {error}")
    
    # Facilitator's follow-up message
    await asyncio.sleep(2)
    
    # Check if it's facilitator's turn again
    current_speaker = dialogue_manager.get_current_speaker(dialogue.id)
    if current_speaker == facilitator_id:
        facilitator_followup = Message(
            id=uuid.uuid4(),
            type=MessageType.MESSAGE,
            role=MessageRole.USER,
            sender=facilitator_id,
            content=MessageContent(text="""
            Thank you both for those thoughtful introductions. I'm particularly intrigued by the
            concepts of cognitive diversity and structured dialogue frameworks you've mentioned.
            
            Let's move our dialogue into the exploration phase and dig deeper into how we might
            formalize the process of knowledge synthesis across models. What specific mechanisms
            or protocols could help balance individual model strengths while ensuring that the
            final synthesis is more than just an average of outputs?
            """),
            metadata=MessageMetadata(
                dialogue_id=dialogue.id,
                sequence_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.sequence_number + 1,
                turn_number=dialogue_manager.active_dialogues[dialogue.id].messages[-1].metadata.turn_number + 1,
                timestamp=datetime.utcnow()
            )
        )
        
        # Change dialogue phase
        await dialogue_manager.change_dialogue_phase(dialogue.id, DialoguePhase.EXPLORATION)
        
        success, error = await dialogue_manager.process_message(facilitator_followup)
        if not success:
            print(f"Error sending facilitator follow-up: {error}")
    
    # Display dialogue status
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    print("\nDialogue Status:")
    print(f"- State: {status['state']}")
    print(f"- Phase: {status['phase']}")
    print(f"- Message count: {status['message_count']}")
    print(f"- Next speaker: {status['current_speaker_name']}")
    
    print("\nDialogue example completed. In a real application, this would continue with:")
    print("1. Sending API requests to actual AI models")
    print("2. Processing and routing responses through the dialogue manager")
    print("3. Advancing through dialogue phases based on content and facilitator guidance")
    print("4. Generating a final synthesis of insights at the conclusion")


if __name__ == "__main__":
    asyncio.run(main())