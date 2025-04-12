"""
Tests for the Conversation Orchestrator components.

This module contains tests for dialogue management, turn-taking policies,
and state management within the Fire Circle system.
"""

import asyncio
import pytest
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from firecircle.orchestrator import (
    DialogueManager, 
    TurnPolicy, 
    DialogueState, 
    DialoguePhase
)
from firecircle.orchestrator.dialogue_manager import DialogueConfig
from firecircle.orchestrator.turn_policy import TurnManager
from firecircle.memory import InMemoryStore
from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    MessageType, 
    MessageRole, 
    MessageMetadata,
    Participant, 
    Dialogue
)


def create_test_participants(count: int = 3, include_facilitator: bool = True) -> Tuple[List[Participant], Optional[uuid.UUID]]:
    """Helper function to create test participants."""
    participants = []
    facilitator_id = None
    
    # Create human participant (facilitator if requested)
    human_id = uuid.uuid4()
    if include_facilitator:
        facilitator_id = human_id
    
    participants.append(
        Participant(
            id=human_id,
            name="Human",
            type="human"
        )
    )
    
    # Create AI participants
    ai_models = ["openai", "anthropic", "mistral"]
    for i in range(min(count - 1, len(ai_models))):
        participants.append(
            Participant(
                id=uuid.uuid4(),
                name=f"{ai_models[i].capitalize()} Assistant",
                type="ai_model",
                provider=ai_models[i],
                model=f"test-model-{i}"
            )
        )
    
    return participants, facilitator_id


def create_test_message(
    dialogue_id: uuid.UUID,
    sender_id: uuid.UUID,
    sequence_number: int = 1,
    text: str = "Test message"
) -> Message:
    """Helper function to create a test message."""
    return Message(
        id=uuid.uuid4(),
        type=MessageType.MESSAGE,
        role=MessageRole.ASSISTANT,
        sender=sender_id,
        content=MessageContent(text=text),
        metadata=MessageMetadata(
            dialogue_id=dialogue_id,
            sequence_number=sequence_number,
            turn_number=sequence_number,
            timestamp=datetime.utcnow()
        )
    )


@pytest.fixture
def memory_store():
    """Fixture to provide a memory store for testing."""
    return InMemoryStore()


@pytest.mark.asyncio
async def test_create_dialogue(memory_store):
    """Test creating a dialogue with participants."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants()
    
    # Create dialogue config
    config = DialogueConfig(
        title="Test Dialogue",
        turn_policy=TurnPolicy.ROUND_ROBIN,
        max_consecutive_turns=1,
        randomize_initial_order=False,
        require_facilitator=True,
        initial_phase=DialoguePhase.INTRODUCTION
    )
    
    # Create dialogue
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id,
        system_message="This is a test dialogue"
    )
    
    # Check dialogue was created
    assert dialogue is not None
    assert dialogue.id is not None
    assert dialogue.title == "Test Dialogue"
    assert len(dialogue.participants) == len(participants)
    
    # Check dialogue state
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.ACTIVE
    assert status["phase"] == DialoguePhase.INTRODUCTION
    
    # Check system message was added
    messages = await dialogue_manager.get_dialogue_history(dialogue.id)
    assert len(messages) == 1
    assert messages[0].type == MessageType.SYSTEM
    assert messages[0].content.text == "This is a test dialogue"


@pytest.mark.asyncio
async def test_message_processing_round_robin(memory_store):
    """Test message processing with round-robin turn policy."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants(count=3)
    
    # Create dialogue config
    config = DialogueConfig(
        title="Round Robin Test",
        turn_policy=TurnPolicy.ROUND_ROBIN,
        randomize_initial_order=False  # Predictable order for testing
    )
    
    # Create dialogue
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Check current speaker
    current_speaker_id = dialogue_manager.get_current_speaker(dialogue.id)
    assert current_speaker_id == facilitator_id
    
    # Send message from current speaker
    message1 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=current_speaker_id,
        text="Message from facilitator"
    )
    
    success, error = await dialogue_manager.process_message(message1)
    assert success, f"Message processing failed: {error}"
    
    # Check turn advanced to next speaker
    next_speaker_id = dialogue_manager.get_current_speaker(dialogue.id)
    assert next_speaker_id != facilitator_id
    assert next_speaker_id == participants[1].id
    
    # Try to send message from facilitator (out of turn)
    message2 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=facilitator_id,
        text="Out of turn message"
    )
    
    success, error = await dialogue_manager.process_message(message2)
    assert not success, "Out-of-turn message should be rejected"
    
    # Send message from correct speaker
    message3 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=next_speaker_id,
        text="Message from next speaker"
    )
    
    success, error = await dialogue_manager.process_message(message3)
    assert success, f"Message processing failed: {error}"
    
    # Check turn advanced to third speaker
    third_speaker_id = dialogue_manager.get_current_speaker(dialogue.id)
    assert third_speaker_id == participants[2].id


@pytest.mark.asyncio
async def test_message_processing_free_form(memory_store):
    """Test message processing with free-form turn policy."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants(count=3)
    
    # Create dialogue config
    config = DialogueConfig(
        title="Free Form Test",
        turn_policy=TurnPolicy.FREE_FORM
    )
    
    # Create dialogue
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Send multiple messages from the same participant
    sender_id = participants[0].id
    
    # First message
    message1 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=sender_id,
        text="First message"
    )
    
    success1, error1 = await dialogue_manager.process_message(message1)
    assert success1, f"Message processing failed: {error1}"
    
    # Second message from same sender
    message2 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=sender_id,
        text="Second message"
    )
    
    success2, error2 = await dialogue_manager.process_message(message2)
    assert success2, f"Second message processing failed: {error2}"
    
    # Message from different participant
    message3 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=participants[1].id,
        text="Message from another participant"
    )
    
    success3, error3 = await dialogue_manager.process_message(message3)
    assert success3, f"Message from another participant failed: {error3}"
    
    # Check message history
    messages = await dialogue_manager.get_dialogue_history(dialogue.id)
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_message_processing_facilitator(memory_store):
    """Test message processing with facilitator-led turn policy."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants(count=3)
    
    # Create dialogue config
    config = DialogueConfig(
        title="Facilitator Test",
        turn_policy=TurnPolicy.FACILITATOR,
        require_facilitator=True
    )
    
    # Create dialogue
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Check current speaker
    current_speaker_id = dialogue_manager.get_current_speaker(dialogue.id)
    assert current_speaker_id == facilitator_id
    
    # Send message from facilitator
    message1 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=facilitator_id,
        text="Message from facilitator"
    )
    
    success, error = await dialogue_manager.process_message(message1)
    assert success, f"Message processing failed: {error}"
    
    # Set next speaker
    next_speaker_id = participants[1].id
    dialogue_manager.set_next_speaker(dialogue.id, next_speaker_id)
    
    # Check current speaker is updated
    current_speaker_id = dialogue_manager.get_current_speaker(dialogue.id)
    assert current_speaker_id == next_speaker_id
    
    # Send message from designated speaker
    message2 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=next_speaker_id,
        text="Message from designated speaker"
    )
    
    success, error = await dialogue_manager.process_message(message2)
    assert success, f"Message processing failed: {error}"
    
    # Try to send message from non-designated speaker
    message3 = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=participants[2].id,
        text="Message from non-designated speaker"
    )
    
    success, error = await dialogue_manager.process_message(message3)
    assert not success, "Message from non-designated speaker should be rejected"


@pytest.mark.asyncio
async def test_dialogue_state_transitions(memory_store):
    """Test dialogue state transitions."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants()
    
    # Create dialogue
    config = DialogueConfig(title="State Transition Test")
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Check initial state
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.ACTIVE
    
    # Pause the dialogue
    success = await dialogue_manager.change_dialogue_state(dialogue.id, DialogueState.PAUSED)
    assert success, "Failed to pause dialogue"
    
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.PAUSED
    
    # Resume the dialogue
    success = await dialogue_manager.change_dialogue_state(dialogue.id, DialogueState.ACTIVE)
    assert success, "Failed to resume dialogue"
    
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.ACTIVE
    
    # Start concluding
    success = await dialogue_manager.change_dialogue_state(dialogue.id, DialogueState.CONCLUDING)
    assert success, "Failed to start concluding dialogue"
    
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.CONCLUDING
    
    # Complete the dialogue
    success = await dialogue_manager.change_dialogue_state(dialogue.id, DialogueState.COMPLETED)
    assert success, "Failed to complete dialogue"
    
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.COMPLETED
    
    # Try an invalid transition (COMPLETED -> ACTIVE)
    success = await dialogue_manager.change_dialogue_state(dialogue.id, DialogueState.ACTIVE)
    assert not success, "Invalid state transition should be rejected"


@pytest.mark.asyncio
async def test_dialogue_phase_transitions(memory_store):
    """Test dialogue phase transitions."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants()
    
    # Create dialogue
    config = DialogueConfig(
        title="Phase Transition Test",
        initial_phase=DialoguePhase.INTRODUCTION
    )
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Check initial phase
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["phase"] == DialoguePhase.INTRODUCTION
    
    # Move to exploration phase
    success = await dialogue_manager.change_dialogue_phase(dialogue.id, DialoguePhase.EXPLORATION)
    assert success, "Failed to change to exploration phase"
    
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["phase"] == DialoguePhase.EXPLORATION
    
    # Try valid phase transitions
    valid_phases = [DialoguePhase.FOCUSING, DialoguePhase.DELIBERATION, DialoguePhase.REFLECTION]
    for phase in valid_phases:
        success = await dialogue_manager.change_dialogue_phase(dialogue.id, phase)
        assert success, f"Failed to change to {phase} phase"
        
        status = await dialogue_manager.get_dialogue_status(dialogue.id)
        assert status["phase"] == phase
    
    # Try an invalid transition (REFLECTION -> VOTING)
    success = await dialogue_manager.change_dialogue_phase(dialogue.id, DialoguePhase.VOTING)
    assert not success, "Invalid phase transition should be rejected"


@pytest.mark.asyncio
async def test_add_remove_participant(memory_store):
    """Test adding and removing participants from a dialogue."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants(count=2)
    
    # Create dialogue
    config = DialogueConfig(title="Participant Test")
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Check initial participants
    initial_participants = await dialogue_manager.get_dialogue_participants(dialogue.id)
    assert len(initial_participants) == 2
    
    # Add a new participant
    new_participant = Participant(
        id=uuid.uuid4(),
        name="New Participant",
        type="ai_model",
        provider="test"
    )
    
    success = await dialogue_manager.add_participant(dialogue.id, new_participant)
    assert success, "Failed to add participant"
    
    # Check participant was added
    updated_participants = await dialogue_manager.get_dialogue_participants(dialogue.id)
    assert len(updated_participants) == 3
    
    # Remove a participant
    success = await dialogue_manager.remove_participant(dialogue.id, new_participant.id)
    assert success, "Failed to remove participant"
    
    # Check participant was removed
    final_participants = await dialogue_manager.get_dialogue_participants(dialogue.id)
    assert len(final_participants) == 2


@pytest.mark.asyncio
async def test_close_dialogue(memory_store):
    """Test closing a dialogue."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants()
    
    # Create dialogue
    config = DialogueConfig(title="Close Test")
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Add some messages
    for i in range(3):
        message = create_test_message(
            dialogue_id=dialogue.id,
            sender_id=facilitator_id,
            sequence_number=i+1,
            text=f"Message {i+1}"
        )
        
        await dialogue_manager.process_message(message)
    
    # Close the dialogue
    success = await dialogue_manager.close_dialogue(dialogue.id)
    assert success, "Failed to close dialogue"
    
    # Check dialogue status
    status = await dialogue_manager.get_dialogue_status(dialogue.id)
    assert status["state"] == DialogueState.COMPLETED
    
    # Check dialogue is closed in memory store
    stored_dialogue = await memory_store.get_dialogue(dialogue.id)
    assert stored_dialogue.status == "completed"


@pytest.mark.asyncio
async def test_consensus_turn_policy(memory_store):
    """Test consensus turn policy."""
    # Create dialogue manager
    dialogue_manager = DialogueManager(memory_store)
    
    # Create test participants
    participants, facilitator_id = create_test_participants(count=3)
    
    # Create dialogue config
    config = DialogueConfig(
        title="Consensus Test",
        turn_policy=TurnPolicy.CONSENSUS,
        randomize_initial_order=False  # Predictable order for testing
    )
    
    # Create dialogue
    dialogue = await dialogue_manager.create_dialogue(
        config=config,
        participants=participants,
        facilitator_id=facilitator_id
    )
    
    # Get initial turn counts from turn manager
    turn_manager = dialogue_manager.turn_managers[dialogue.id]
    assert all(count == 0 for count in turn_manager.turns_taken.values())
    
    # Each participant should be able to speak once per round
    for participant in participants:
        # Check if participant can speak
        can_speak = dialogue_manager.can_speak(dialogue.id, participant.id)
        assert can_speak, f"Participant {participant.id} should be able to speak"
        
        # Send message
        message = create_test_message(
            dialogue_id=dialogue.id,
            sender_id=participant.id,
            text=f"Message from {participant.name}"
        )
        
        success, error = await dialogue_manager.process_message(message)
        assert success, f"Message processing failed: {error}"
    
    # Check all participants have taken one turn
    for participant in participants:
        assert turn_manager.turns_taken[participant.id] == 1
    
    # Try to send another message from the first participant
    message = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=participants[0].id,
        text="Second message from first participant"
    )
    
    # If everyone has spoken once, the first participant should be able to speak again
    can_speak = dialogue_manager.can_speak(dialogue.id, participants[0].id)
    assert can_speak, "First participant should be able to speak in the second round"
    
    success, error = await dialogue_manager.process_message(message)
    assert success, f"Message processing failed: {error}"
    
    # Check turn count was updated
    assert turn_manager.turns_taken[participants[0].id] == 2