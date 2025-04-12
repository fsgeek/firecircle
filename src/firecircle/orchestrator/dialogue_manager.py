"""
Dialogue Manager for Fire Circle.

This module implements the core orchestration logic for managing dialogues
between multiple participants in the Fire Circle system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, Field

from firecircle.protocol.message import (
    Dialogue,
    Message, 
    MessageContent,
    MessageType,
    MessageRole,
    MessageStatus,
    Participant,
    create_system_message
)
from firecircle.orchestrator.dialogue_state import (
    DialogueState,
    DialoguePhase,
    validate_state_transition,
    validate_phase_transition
)
from firecircle.orchestrator.turn_policy import TurnPolicy, TurnManager
from firecircle.memory.base import MemoryStore


class DialogueConfig(BaseModel):
    """Configuration for a dialogue in the Fire Circle system."""
    
    title: str = Field(..., description="Title or topic of the dialogue")
    turn_policy: TurnPolicy = Field(default=TurnPolicy.ROUND_ROBIN, description="How turn-taking is managed")
    max_consecutive_turns: int = Field(default=1, description="Maximum consecutive turns per participant")
    randomize_initial_order: bool = Field(default=True, description="Whether to randomize the initial speaking order")
    max_turns_per_participant: Optional[int] = Field(None, description="Maximum turns a participant can take")
    max_message_length: Optional[int] = Field(None, description="Maximum characters in a message")
    min_response_time_ms: Optional[int] = Field(None, description="Minimum milliseconds before responding")
    require_facilitator: bool = Field(default=False, description="Whether a facilitator is required")
    allow_empty_chair: bool = Field(default=True, description="Whether Empty Chair messages are allowed")
    auto_advance_turns: bool = Field(default=True, description="Whether to automatically advance turns")
    initial_phase: DialoguePhase = Field(default=DialoguePhase.INTRODUCTION, description="Starting phase of the dialogue")
    custom_rules: Dict[str, Any] = Field(default_factory=dict, description="Custom rule parameters")


class ParticipantStatus(BaseModel):
    """Status and metadata for a participant in a dialogue."""
    
    participant_id: UUID = Field(..., description="ID of the participant")
    name: str = Field(..., description="Display name of the participant")
    is_active: bool = Field(default=True, description="Whether the participant is actively engaged")
    is_facilitator: bool = Field(default=False, description="Whether this participant is the facilitator")
    turns_taken: int = Field(default=0, description="Number of turns taken so far")
    last_message_time: Optional[datetime] = Field(None, description="When they last sent a message")
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time in ms")


class DialogueManager:
    """
    Manages the orchestration of dialogues in the Fire Circle system.
    
    The DialogueManager is responsible for:
    - Creating and initializing dialogues
    - Managing participant assignments
    - Enforcing turn-taking policies
    - Tracking dialogue state transitions
    - Persisting dialogue history
    - Coordinating message routing
    """
    
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        """
        Initialize the dialogue manager.
        
        Args:
            memory_store: Memory store for persisting dialogues and messages
        """
        self.memory_store = memory_store
        self.active_dialogues: Dict[UUID, Dialogue] = {}
        self.dialogue_states: Dict[UUID, DialogueState] = {}
        self.dialogue_phases: Dict[UUID, DialoguePhase] = {}
        self.turn_managers: Dict[UUID, TurnManager] = {}
        self.participant_statuses: Dict[UUID, Dict[UUID, ParticipantStatus]] = {}
        self.message_handlers: Dict[UUID, Callable] = {}
        self.logger = logging.getLogger("firecircle.orchestrator")
    
    async def create_dialogue(
        self,
        config: DialogueConfig,
        participants: List[Participant],
        facilitator_id: Optional[UUID] = None,
        system_message: Optional[str] = None
    ) -> Dialogue:
        """
        Create a new dialogue with the given configuration and participants.
        
        Args:
            config: Configuration for the dialogue
            participants: List of participants to include
            facilitator_id: ID of the facilitator (if any)
            system_message: Optional system message to establish context
            
        Returns:
            A new Dialogue object
        """
        # Create the dialogue
        dialogue = Dialogue(
            title=config.title,
            participants=participants,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=DialogueState.INITIALIZING,
            metadata={
                "config": config.dict(),
                "facilitator_id": facilitator_id
            }
        )
        
        # Set up participant statuses
        participant_statuses = {}
        for participant in participants:
            participant_statuses[participant.id] = ParticipantStatus(
                participant_id=participant.id,
                name=participant.name,
                is_facilitator=(participant.id == facilitator_id)
            )
        
        # Set up turn manager
        turn_manager = TurnManager(
            policy=config.turn_policy,
            participant_ids=[p.id for p in participants],
            facilitator_id=facilitator_id,
            max_consecutive_turns=config.max_consecutive_turns,
            randomize_initial_order=config.randomize_initial_order
        )
        
        # Initialize dialogue state
        self.active_dialogues[dialogue.id] = dialogue
        self.dialogue_states[dialogue.id] = DialogueState.INITIALIZING
        self.dialogue_phases[dialogue.id] = config.initial_phase
        self.turn_managers[dialogue.id] = turn_manager
        self.participant_statuses[dialogue.id] = participant_statuses
        
        # Add system message if provided
        if system_message:
            system_msg = create_system_message(
                dialogue_id=dialogue.id,
                content=system_message,
                sequence_number=1
            )
            dialogue.add_message(system_msg)
            
            # Store in memory if available
            if self.memory_store:
                await self.memory_store.store_message(system_msg)
        
        # Store the dialogue in memory if available
        if self.memory_store:
            await self.memory_store.store_dialogue(dialogue)
        
        # Activate the dialogue
        await self.activate_dialogue(dialogue.id)
        
        return dialogue
    
    async def activate_dialogue(self, dialogue_id: UUID) -> bool:
        """
        Activate a dialogue that has been initialized.
        
        Args:
            dialogue_id: ID of the dialogue to activate
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            self.logger.error(f"Dialogue {dialogue_id} not found")
            return False
        
        dialogue = self.active_dialogues[dialogue_id]
        
        # Check current state
        current_state = self.dialogue_states[dialogue_id]
        if current_state != DialogueState.INITIALIZING:
            self.logger.warning(f"Cannot activate dialogue {dialogue_id} in state {current_state}")
            return False
        
        # Change state to ACTIVE
        if not validate_state_transition(current_state, DialogueState.ACTIVE):
            self.logger.error(f"Invalid state transition: {current_state} -> {DialogueState.ACTIVE}")
            return False
            
        self.dialogue_states[dialogue_id] = DialogueState.ACTIVE
        
        # Update dialogue
        dialogue.status = "active"
        dialogue.updated_at = datetime.utcnow()
        
        # Store in memory if available
        if self.memory_store:
            await self.memory_store.store_dialogue(dialogue)
        
        self.logger.info(f"Activated dialogue {dialogue_id}")
        return True
    
    async def change_dialogue_state(
        self,
        dialogue_id: UUID,
        new_state: DialogueState
    ) -> bool:
        """
        Change the state of a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            new_state: New state to set
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            self.logger.error(f"Dialogue {dialogue_id} not found")
            return False
        
        dialogue = self.active_dialogues[dialogue_id]
        current_state = self.dialogue_states[dialogue_id]
        
        # Validate the transition
        if not validate_state_transition(current_state, new_state):
            self.logger.error(f"Invalid state transition: {current_state} -> {new_state}")
            return False
        
        # Update state
        self.dialogue_states[dialogue_id] = new_state
        
        # Update dialogue
        dialogue.status = new_state
        dialogue.updated_at = datetime.utcnow()
        
        # Store in memory if available
        if self.memory_store:
            await self.memory_store.store_dialogue(dialogue)
        
        self.logger.info(f"Changed dialogue {dialogue_id} state from {current_state} to {new_state}")
        return True
    
    async def change_dialogue_phase(
        self,
        dialogue_id: UUID,
        new_phase: DialoguePhase
    ) -> bool:
        """
        Change the phase of a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            new_phase: New phase to set
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            self.logger.error(f"Dialogue {dialogue_id} not found")
            return False
        
        dialogue = self.active_dialogues[dialogue_id]
        current_phase = self.dialogue_phases[dialogue_id]
        
        # Validate the transition
        if not validate_phase_transition(current_phase, new_phase):
            self.logger.error(f"Invalid phase transition: {current_phase} -> {new_phase}")
            return False
        
        # Update phase
        self.dialogue_phases[dialogue_id] = new_phase
        
        # Update dialogue metadata
        dialogue.metadata["current_phase"] = new_phase
        dialogue.updated_at = datetime.utcnow()
        
        # Store in memory if available
        if self.memory_store:
            await self.memory_store.store_dialogue(dialogue)
        
        self.logger.info(f"Changed dialogue {dialogue_id} phase from {current_phase} to {new_phase}")
        return True
    
    def get_current_speaker(self, dialogue_id: UUID) -> Optional[UUID]:
        """
        Get the ID of the participant whose turn it is to speak.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            ID of the current speaker or None
        """
        if dialogue_id not in self.turn_managers:
            return None
        
        return self.turn_managers[dialogue_id].get_current_speaker()
    
    def can_speak(self, dialogue_id: UUID, participant_id: UUID) -> bool:
        """
        Check if a participant is allowed to speak now.
        
        Args:
            dialogue_id: ID of the dialogue
            participant_id: ID of the participant
            
        Returns:
            True if the participant can speak, False otherwise
        """
        if dialogue_id not in self.turn_managers:
            return False
        
        # Check if dialogue is active
        if self.dialogue_states.get(dialogue_id) != DialogueState.ACTIVE:
            return False
        
        # Check if participant is active
        if (dialogue_id in self.participant_statuses and 
            participant_id in self.participant_statuses[dialogue_id]):
            if not self.participant_statuses[dialogue_id][participant_id].is_active:
                return False
        
        # Check turn policy
        return self.turn_managers[dialogue_id].can_speak(participant_id)
    
    def set_next_speaker(self, dialogue_id: UUID, participant_id: UUID) -> bool:
        """
        Set the next speaker for a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            participant_id: ID of the participant to speak next
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.turn_managers:
            return False
        
        return self.turn_managers[dialogue_id].set_next_speaker(participant_id)
    
    async def process_message(self, message: Message) -> Tuple[bool, Optional[str]]:
        """
        Process an incoming message in a dialogue.
        
        Args:
            message: The message to process
            
        Returns:
            Tuple of (success, error_message)
        """
        dialogue_id = message.metadata.dialogue_id
        sender_id = message.sender
        
        # Check if dialogue exists
        if dialogue_id not in self.active_dialogues:
            return False, "Dialogue not found"
        
        dialogue = self.active_dialogues[dialogue_id]
        
        # Check if dialogue is active
        if self.dialogue_states[dialogue_id] != DialogueState.ACTIVE:
            return False, f"Dialogue is not active (current state: {self.dialogue_states[dialogue_id]})"
        
        # Check if sender can speak
        if not self.can_speak(dialogue_id, sender_id):
            return False, "Not your turn to speak"
        
        # Validate message content
        if not message.content or not message.content.text:
            return False, "Message content cannot be empty"
        
        # Check message length constraints
        config = dialogue.metadata.get("config", {})
        max_length = config.get("max_message_length")
        if max_length and len(message.content.text) > max_length:
            return False, f"Message exceeds maximum length of {max_length} characters"
        
        # Add message to dialogue
        dialogue.add_message(message)
        dialogue.updated_at = datetime.utcnow()
        
        # Update participant status
        now = datetime.utcnow()
        if dialogue_id in self.participant_statuses and sender_id in self.participant_statuses[dialogue_id]:
            participant_status = self.participant_statuses[dialogue_id][sender_id]
            participant_status.turns_taken += 1
            participant_status.last_message_time = now
        
        # Record turn in turn manager
        self.turn_managers[dialogue_id].record_turn(sender_id)
        
        # Store in memory if available
        if self.memory_store:
            await self.memory_store.store_message(message)
            await self.memory_store.store_dialogue(dialogue)
        
        # Handle message
        if dialogue_id in self.message_handlers:
            try:
                await self.message_handlers[dialogue_id](message)
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
        
        self.logger.info(f"Processed message {message.id} in dialogue {dialogue_id}")
        return True, None
    
    def register_message_handler(self, dialogue_id: UUID, handler: Callable) -> None:
        """
        Register a handler function for messages in a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            handler: Async function that accepts a Message
        """
        self.message_handlers[dialogue_id] = handler
    
    async def add_participant(
        self,
        dialogue_id: UUID,
        participant: Participant
    ) -> bool:
        """
        Add a new participant to a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            participant: The participant to add
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            return False
        
        dialogue = self.active_dialogues[dialogue_id]
        
        # Check if already in dialogue
        if any(p.id == participant.id for p in dialogue.participants):
            return False
        
        # Add to dialogue
        dialogue.participants.append(participant)
        dialogue.updated_at = datetime.utcnow()
        
        # Add to turn manager
        if dialogue_id in self.turn_managers:
            self.turn_managers[dialogue_id].add_participant(participant.id)
        
        # Add participant status
        if dialogue_id in self.participant_statuses:
            self.participant_statuses[dialogue_id][participant.id] = ParticipantStatus(
                participant_id=participant.id,
                name=participant.name
            )
        
        # Store in memory if available
        if self.memory_store:
            await self.memory_store.store_dialogue(dialogue)
        
        self.logger.info(f"Added participant {participant.id} to dialogue {dialogue_id}")
        return True
    
    async def remove_participant(
        self,
        dialogue_id: UUID,
        participant_id: UUID
    ) -> bool:
        """
        Remove a participant from a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            participant_id: ID of the participant to remove
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            return False
        
        dialogue = self.active_dialogues[dialogue_id]
        
        # Find and remove the participant
        participant_index = None
        for i, p in enumerate(dialogue.participants):
            if p.id == participant_id:
                participant_index = i
                break
        
        if participant_index is None:
            return False
        
        dialogue.participants.pop(participant_index)
        dialogue.updated_at = datetime.utcnow()
        
        # Remove from turn manager
        if dialogue_id in self.turn_managers:
            self.turn_managers[dialogue_id].remove_participant(participant_id)
        
        # Remove participant status
        if dialogue_id in self.participant_statuses and participant_id in self.participant_statuses[dialogue_id]:
            del self.participant_statuses[dialogue_id][participant_id]
        
        # Store in memory if available
        if self.memory_store:
            await self.memory_store.store_dialogue(dialogue)
        
        self.logger.info(f"Removed participant {participant_id} from dialogue {dialogue_id}")
        return True
    
    async def get_dialogue_history(
        self,
        dialogue_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Message]:
        """
        Get the message history for a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            limit: Maximum number of messages to retrieve
            offset: Number of messages to skip
            
        Returns:
            List of messages in the dialogue
        """
        if self.memory_store:
            # Use memory store for retrieval
            return await self.memory_store.get_dialogue_messages(dialogue_id, limit, offset)
        elif dialogue_id in self.active_dialogues:
            # Use in-memory dialogue
            messages = self.active_dialogues[dialogue_id].messages
            
            # Sort by sequence number
            messages.sort(key=lambda m: m.metadata.sequence_number)
            
            # Apply offset and limit
            if offset is not None:
                messages = messages[offset:]
            if limit is not None:
                messages = messages[:limit]
                
            return messages
        else:
            return []
    
    async def get_dialogue_participants(self, dialogue_id: UUID) -> List[Participant]:
        """
        Get the participants in a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            List of participants in the dialogue
        """
        if dialogue_id in self.active_dialogues:
            return self.active_dialogues[dialogue_id].participants
        else:
            return []
    
    async def get_dialogue_status(self, dialogue_id: UUID) -> Dict[str, Any]:
        """
        Get the current status of a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            Dictionary with dialogue status information
        """
        if dialogue_id not in self.active_dialogues:
            return {"error": "Dialogue not found"}
        
        dialogue = self.active_dialogues[dialogue_id]
        
        # Get current speaker
        current_speaker_id = self.get_current_speaker(dialogue_id)
        current_speaker_name = None
        if current_speaker_id:
            for p in dialogue.participants:
                if p.id == current_speaker_id:
                    current_speaker_name = p.name
                    break
        
        # Build status object
        return {
            "dialogue_id": dialogue_id,
            "title": dialogue.title,
            "state": self.dialogue_states.get(dialogue_id, DialogueState.ERROR),
            "phase": self.dialogue_phases.get(dialogue_id, DialoguePhase.INTRODUCTION),
            "current_speaker_id": current_speaker_id,
            "current_speaker_name": current_speaker_name,
            "participant_count": len(dialogue.participants),
            "message_count": len(dialogue.messages),
            "created_at": dialogue.created_at,
            "updated_at": dialogue.updated_at,
            "config": dialogue.metadata.get("config", {})
        }
    
    async def close_dialogue(self, dialogue_id: UUID) -> bool:
        """
        Close a dialogue and mark it as completed.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            return False
        
        # Change state to COMPLETED
        success = await self.change_dialogue_state(dialogue_id, DialogueState.COMPLETED)
        
        if success:
            # Clean up resources
            if dialogue_id in self.message_handlers:
                del self.message_handlers[dialogue_id]
            
            self.logger.info(f"Closed dialogue {dialogue_id}")
            
        return success