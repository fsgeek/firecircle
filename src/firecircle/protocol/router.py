"""
Message Router for Fire Circle

This module implements the message routing and protocol management layer for 
the Fire Circle system. It handles the distribution of messages between 
participants according to dialogue rules and turn-taking protocols.

The router is responsible for:
- Managing the flow of messages between participants
- Enforcing turn-taking and dialogue protocol rules
- Tracking message delivery and read status
- Handling dialogue state transitions
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from firecircle.protocol.message import (
    Dialogue,
    Message,
    MessageStatus,
    MessageType,
    Participant
)


class DeliveryStatus(str, Enum):
    """Status of message delivery attempts."""
    
    SUCCESS = "success"            # Delivered successfully
    PARTICIPANT_UNAVAILABLE = "participant_unavailable"  # Recipient not available
    DELIVERY_TIMEOUT = "delivery_timeout"          # Delivery timed out
    REJECTED = "rejected"          # Recipient rejected the message
    ERROR = "error"                # General error in delivery


class DeliveryReport(BaseModel):
    """Report of a message delivery attempt."""
    
    message_id: UUID = Field(..., description="ID of the message")
    recipient_id: UUID = Field(..., description="ID of the recipient")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time of delivery attempt")
    status: DeliveryStatus = Field(..., description="Status of the delivery")
    error_message: Optional[str] = Field(None, description="Error message if delivery failed")


class DialogueState(str, Enum):
    """States of a dialogue within the system."""
    
    INITIALIZING = "initializing"  # Setting up, gathering participants
    ACTIVE = "active"              # Normal dialogue flow
    PAUSED = "paused"              # Temporarily suspended
    CONCLUDING = "concluding"      # Final statements being made
    COMPLETED = "completed"        # Dialogue has ended normally
    ABANDONED = "abandoned"        # Dialogue ended prematurely
    ERROR = "error"                # Error state


class TurnPolicy(str, Enum):
    """Turn-taking policies for dialogue management."""
    
    ROUND_ROBIN = "round_robin"    # Participants speak in a fixed order
    FACILITATOR = "facilitator"    # Facilitator decides who speaks next
    FREE_FORM = "free_form"        # Anyone can speak at any time
    REACTIVE = "reactive"          # Participants respond to being addressed
    CONSENSUS = "consensus"        # All must contribute before proceeding


class DialogueRules(BaseModel):
    """Rules and constraints for a dialogue."""
    
    turn_policy: TurnPolicy = Field(default=TurnPolicy.ROUND_ROBIN, description="How turn-taking is managed")
    max_turns_per_participant: Optional[int] = Field(None, description="Maximum turns a participant can take")
    max_message_length: Optional[int] = Field(None, description="Maximum characters in a message")
    min_response_time_ms: Optional[int] = Field(None, description="Minimum milliseconds before responding")
    allowed_message_types: Optional[List[MessageType]] = Field(None, description="Message types permitted in this dialogue")
    require_facilitator: bool = Field(default=False, description="Whether a facilitator is required")
    facilitator_id: Optional[UUID] = Field(None, description="ID of the facilitator if required")
    allow_empty_chair: bool = Field(default=True, description="Whether Empty Chair messages are allowed")
    consecutive_turns_allowed: bool = Field(default=False, description="Whether participants can take consecutive turns")
    auto_advance_turns: bool = Field(default=True, description="Whether to automatically advance turns")
    custom_rules: Dict[str, Any] = Field(default_factory=dict, description="Custom rule parameters")


class DialogueStateManager(BaseModel):
    """
    Manages the state of a dialogue and enforces dialogue rules.
    
    This component is responsible for tracking whose turn it is to speak,
    managing the overall dialogue phase, and enforcing any turn-taking or 
    contribution rules.
    """
    
    dialogue_id: UUID = Field(..., description="ID of the managed dialogue")
    rules: DialogueRules = Field(..., description="Rules governing this dialogue")
    state: DialogueState = Field(default=DialogueState.INITIALIZING, description="Current state of the dialogue")
    
    # Turn management
    current_turn: int = Field(default=0, description="Current turn number")
    current_speaker_index: int = Field(default=0, description="Index of current speaker in turn order")
    turn_order: List[UUID] = Field(default_factory=list, description="Order of participant turns")
    turns_taken: Dict[UUID, int] = Field(default_factory=dict, description="Count of turns taken by each participant")
    
    # Tracking
    last_message_time: Optional[datetime] = Field(None, description="When the last message was sent")
    last_state_change: datetime = Field(default_factory=datetime.utcnow, description="When state last changed")
    
    def initialize(self, participants: List[Participant]) -> None:
        """
        Initialize the dialogue state with participants.
        
        Args:
            participants: List of dialogue participants
        """
        # Reset state
        self.state = DialogueState.INITIALIZING
        self.current_turn = 0
        self.current_speaker_index = 0
        self.turns_taken = {}
        
        # Set up turn order based on policy
        participant_ids = [p.id for p in participants]
        
        if self.rules.turn_policy == TurnPolicy.ROUND_ROBIN:
            # Simple round-robin ordering
            self.turn_order = participant_ids.copy()
            
            # If we have a facilitator, put them first
            if self.rules.facilitator_id and self.rules.facilitator_id in self.turn_order:
                self.turn_order.remove(self.rules.facilitator_id)
                self.turn_order.insert(0, self.rules.facilitator_id)
                
        elif self.rules.turn_policy == TurnPolicy.FACILITATOR:
            # Only the facilitator is in the turn order initially
            if not self.rules.facilitator_id:
                raise ValueError("Facilitator policy requires a facilitator_id")
            self.turn_order = [self.rules.facilitator_id]
            
        elif self.rules.turn_policy == TurnPolicy.FREE_FORM:
            # No enforced order, but initialize with all participants
            self.turn_order = participant_ids.copy()
            
        elif self.rules.turn_policy == TurnPolicy.REACTIVE:
            # Start with facilitator or first participant
            if self.rules.facilitator_id:
                self.turn_order = [self.rules.facilitator_id]
            else:
                self.turn_order = participant_ids[:1]
                
        elif self.rules.turn_policy == TurnPolicy.CONSENSUS:
            # All participants must contribute
            self.turn_order = participant_ids.copy()
        
        # Initialize turn counts
        for p_id in participant_ids:
            self.turns_taken[p_id] = 0
            
        # Mark initialization complete
        self.state = DialogueState.ACTIVE
        self.last_state_change = datetime.utcnow()
    
    def get_current_speaker(self) -> Optional[UUID]:
        """
        Get the ID of the participant whose turn it is to speak.
        
        Returns:
            ID of the current speaker or None if no current speaker
        """
        if not self.turn_order or self.state != DialogueState.ACTIVE:
            return None
            
        if 0 <= self.current_speaker_index < len(self.turn_order):
            return self.turn_order[self.current_speaker_index]
            
        return None
    
    def can_speak(self, participant_id: UUID) -> bool:
        """
        Check if a participant is allowed to speak now.
        
        Args:
            participant_id: ID of the participant
            
        Returns:
            True if the participant can speak, False otherwise
        """
        # Check dialogue state
        if self.state != DialogueState.ACTIVE:
            return False
            
        # Apply turn policy rules
        if self.rules.turn_policy == TurnPolicy.FREE_FORM:
            # Anyone can speak in free-form
            return True
            
        elif self.rules.turn_policy == TurnPolicy.FACILITATOR:
            # Either the facilitator or whoever they've given the floor to
            return (
                participant_id == self.rules.facilitator_id or 
                participant_id == self.get_current_speaker()
            )
            
        elif self.rules.turn_policy == TurnPolicy.ROUND_ROBIN:
            # Only the current speaker in the rotation
            return participant_id == self.get_current_speaker()
            
        elif self.rules.turn_policy == TurnPolicy.REACTIVE:
            # Current speaker or anyone addressing them directly
            return participant_id == self.get_current_speaker()
            
        elif self.rules.turn_policy == TurnPolicy.CONSENSUS:
            # Anyone who hasn't spoken yet in this round, or the facilitator
            if participant_id == self.rules.facilitator_id:
                return True
                
            # Check if they've spoken less than others
            if participant_id in self.turns_taken:
                min_turns = min(self.turns_taken.values())
                return self.turns_taken[participant_id] <= min_turns
        
        return False
    
    def record_message(self, message: Message) -> None:
        """
        Record that a message has been sent and update turn state.
        
        Args:
            message: The message that was sent
        """
        # Update timestamps
        self.last_message_time = datetime.utcnow()
        
        # Update turn counts
        if message.sender in self.turns_taken:
            self.turns_taken[message.sender] += 1
        
        # Only advance turns for regular messages, not system or clarifications
        if (message.type not in [MessageType.SYSTEM, MessageType.CLARIFICATION, MessageType.REFLECTION] and
                self.rules.auto_advance_turns):
            self._advance_turn(message.sender)
    
    def _advance_turn(self, last_speaker: UUID) -> None:
        """
        Advance to the next participant's turn.
        
        Args:
            last_speaker: ID of the participant who just spoke
        """
        if self.rules.turn_policy == TurnPolicy.FREE_FORM:
            # No turn advancement in free-form
            return
            
        elif self.rules.turn_policy == TurnPolicy.FACILITATOR:
            # Only the facilitator changes turns
            if last_speaker == self.rules.facilitator_id:
                # Facilitator spoke, so reset to facilitator for next turn
                self.current_speaker_index = 0  # Facilitator is at index 0
                
        elif self.rules.turn_policy == TurnPolicy.ROUND_ROBIN:
            # Move to next participant in the rotation
            if self.rules.consecutive_turns_allowed or last_speaker == self.get_current_speaker():
                self.current_speaker_index = (self.current_speaker_index + 1) % len(self.turn_order)
                self.current_turn += 1
                
        elif self.rules.turn_policy == TurnPolicy.REACTIVE:
            # Next speaker is determined dynamically based on who was addressed
            # This is handled elsewhere based on message content and references
            pass
            
        elif self.rules.turn_policy == TurnPolicy.CONSENSUS:
            # Check if we've completed a round (everyone spoke)
            min_turns = min(self.turns_taken.values())
            max_turns = max(self.turns_taken.values())
            
            if min_turns == max_turns:
                # Everyone has spoken the same number of times, start new round
                self.current_turn += 1
                
                # If there's a facilitator, give them the first turn in the new round
                if self.rules.facilitator_id and self.rules.facilitator_id in self.turn_order:
                    self.current_speaker_index = self.turn_order.index(self.rules.facilitator_id)
                else:
                    # Otherwise start with the first participant
                    self.current_speaker_index = 0
    
    def set_next_speaker(self, participant_id: UUID) -> bool:
        """
        Explicitly set the next speaker (used by facilitator).
        
        Args:
            participant_id: ID of the participant to speak next
            
        Returns:
            True if successful, False if not allowed
        """
        # Check if we have a facilitator policy or the caller is the facilitator
        if self.rules.turn_policy not in [TurnPolicy.FACILITATOR, TurnPolicy.REACTIVE]:
            return False
            
        # Find the participant in the turn order
        if participant_id in self.turn_order:
            self.current_speaker_index = self.turn_order.index(participant_id)
            return True
            
        # Add them to the turn order if not present (for reactive policy)
        if self.rules.turn_policy == TurnPolicy.REACTIVE:
            self.turn_order.append(participant_id)
            self.current_speaker_index = len(self.turn_order) - 1
            return True
            
        return False
    
    def change_state(self, new_state: DialogueState) -> bool:
        """
        Change the dialogue state.
        
        Args:
            new_state: The new state to transition to
            
        Returns:
            True if state was changed, False if invalid transition
        """
        # Check for valid transitions
        valid_transitions = {
            DialogueState.INITIALIZING: [DialogueState.ACTIVE, DialogueState.ABANDONED, DialogueState.ERROR],
            DialogueState.ACTIVE: [DialogueState.PAUSED, DialogueState.CONCLUDING, DialogueState.ABANDONED, DialogueState.ERROR],
            DialogueState.PAUSED: [DialogueState.ACTIVE, DialogueState.CONCLUDING, DialogueState.ABANDONED, DialogueState.ERROR],
            DialogueState.CONCLUDING: [DialogueState.COMPLETED, DialogueState.ACTIVE, DialogueState.ABANDONED, DialogueState.ERROR],
            DialogueState.COMPLETED: [DialogueState.ERROR],  # Very restricted once completed
            DialogueState.ABANDONED: [DialogueState.ACTIVE, DialogueState.ERROR],  # Can resume an abandoned dialogue
            DialogueState.ERROR: [DialogueState.INITIALIZING, DialogueState.ACTIVE],  # Can recover from errors
        }
        
        if new_state not in valid_transitions.get(self.state, []):
            return False
            
        # Update state
        self.state = new_state
        self.last_state_change = datetime.utcnow()
        return True


class MessageRouter:
    """
    Handles message routing and delivery between dialogue participants.
    
    The MessageRouter is responsible for:
    - Maintaining connections to participants
    - Ensuring proper message delivery
    - Tracking message status changes
    - Enforcing dialogue rules via the DialogueStateManager
    """
    
    def __init__(self):
        """Initialize the message router."""
        self.dialogues: Dict[UUID, Dialogue] = {}
        self.state_managers: Dict[UUID, DialogueStateManager] = {}
        self.participant_handlers: Dict[UUID, Callable] = {}
        self.pending_deliveries: Dict[UUID, List[DeliveryReport]] = {}
        self.logger = logging.getLogger("firecircle.protocol.router")
    
    def register_dialogue(self, dialogue: Dialogue, rules: DialogueRules) -> None:
        """
        Register a dialogue with the router.
        
        Args:
            dialogue: The dialogue to register
            rules: Rules governing the dialogue
        """
        # Store the dialogue
        self.dialogues[dialogue.id] = dialogue
        
        # Create state manager
        state_manager = DialogueStateManager(
            dialogue_id=dialogue.id,
            rules=rules
        )
        
        # Initialize with participants
        state_manager.initialize(dialogue.participants)
        
        # Store state manager
        self.state_managers[dialogue.id] = state_manager
        
        # Initialize pending deliveries
        self.pending_deliveries[dialogue.id] = []
        
        self.logger.info(f"Registered dialogue {dialogue.id} with {len(dialogue.participants)} participants")
    
    def register_participant_handler(self, participant_id: UUID, handler: Callable) -> None:
        """
        Register a handler function for a participant.
        
        The handler will be called when messages are routed to this participant.
        
        Args:
            participant_id: ID of the participant
            handler: Async function that accepts a Message and returns a DeliveryStatus
        """
        self.participant_handlers[participant_id] = handler
        self.logger.debug(f"Registered handler for participant {participant_id}")
    
    async def route_message(self, message: Message) -> List[DeliveryReport]:
        """
        Route a message to its intended recipients.
        
        Args:
            message: The message to route
            
        Returns:
            List of delivery reports
        """
        # Get the dialogue
        dialogue_id = message.metadata.dialogue_id
        if dialogue_id not in self.dialogues:
            self.logger.error(f"Dialogue {dialogue_id} not found")
            return [
                DeliveryReport(
                    message_id=message.id,
                    recipient_id=message.sender,  # Send error back to sender
                    status=DeliveryStatus.ERROR,
                    error_message="Dialogue not found"
                )
            ]
            
        dialogue = self.dialogues[dialogue_id]
        state_manager = self.state_managers[dialogue_id]
        
        # Check if sender can speak based on dialogue rules
        if not state_manager.can_speak(message.sender):
            self.logger.warning(f"Participant {message.sender} is not allowed to speak now")
            return [
                DeliveryReport(
                    message_id=message.id,
                    recipient_id=message.sender,
                    status=DeliveryStatus.REJECTED,
                    error_message="Not your turn to speak"
                )
            ]
        
        # Determine recipients based on visibility
        recipients: Set[UUID] = set()
        
        if message.metadata.visibility == "all":
            # Send to all participants
            recipients = {p.id for p in dialogue.participants}
        elif message.metadata.visibility == "facilitator":
            # Send to facilitator and keep a copy for sender
            facilitator_id = state_manager.rules.facilitator_id
            if facilitator_id:
                recipients = {facilitator_id, message.sender}
            else:
                # No facilitator, just send back to sender
                recipients = {message.sender}
        elif message.metadata.visibility == "direct":
            # Send to specific recipients
            recipients = set(message.metadata.recipients)
            recipients.add(message.sender)  # Always include sender
        elif message.metadata.visibility == "private":
            # Only visible to sender
            recipients = {message.sender}
        
        # Update message status
        message.status = MessageStatus.SENT
        
        # Add to dialogue
        dialogue.add_message(message)
        
        # Record in state manager
        state_manager.record_message(message)
        
        # Deliver to recipients
        delivery_reports = []
        delivery_tasks = []
        
        for recipient_id in recipients:
            if recipient_id in self.participant_handlers:
                # Get handler
                handler = self.participant_handlers[recipient_id]
                
                # Create delivery task
                task = asyncio.create_task(self._deliver_message(
                    message, recipient_id, handler
                ))
                delivery_tasks.append((recipient_id, task))
            else:
                # No handler for this recipient
                report = DeliveryReport(
                    message_id=message.id,
                    recipient_id=recipient_id,
                    status=DeliveryStatus.PARTICIPANT_UNAVAILABLE,
                    error_message="No handler registered for participant"
                )
                delivery_reports.append(report)
        
        # Wait for all deliveries to complete
        for recipient_id, task in delivery_tasks:
            try:
                status = await task
                report = DeliveryReport(
                    message_id=message.id,
                    recipient_id=recipient_id,
                    status=status
                )
                delivery_reports.append(report)
            except Exception as e:
                # Handle delivery error
                report = DeliveryReport(
                    message_id=message.id,
                    recipient_id=recipient_id,
                    status=DeliveryStatus.ERROR,
                    error_message=str(e)
                )
                delivery_reports.append(report)
                self.logger.error(f"Error delivering message to {recipient_id}: {e}")
        
        # Store reports
        self.pending_deliveries[dialogue_id].extend(delivery_reports)
        
        return delivery_reports
    
    async def _deliver_message(
        self, 
        message: Message, 
        recipient_id: UUID, 
        handler: Callable
    ) -> DeliveryStatus:
        """
        Deliver a message to a single recipient.
        
        Args:
            message: The message to deliver
            recipient_id: ID of the recipient
            handler: Handler function for the recipient
            
        Returns:
            Delivery status
        """
        try:
            # Call the handler with the message
            status = await handler(message)
            return status
        except asyncio.TimeoutError:
            # Handle timeout
            return DeliveryStatus.DELIVERY_TIMEOUT
        except Exception as e:
            # Handle other errors
            self.logger.error(f"Error in delivery handler: {e}")
            return DeliveryStatus.ERROR
    
    def get_dialogue_state(self, dialogue_id: UUID) -> Optional[DialogueStateManager]:
        """
        Get the current state manager for a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            The dialogue state manager or None if not found
        """
        return self.state_managers.get(dialogue_id)
    
    def update_message_status(self, message_id: UUID, recipient_id: UUID, new_status: MessageStatus) -> bool:
        """
        Update the status of a message for a specific recipient.
        
        Args:
            message_id: ID of the message
            recipient_id: ID of the recipient
            new_status: New status to set
            
        Returns:
            True if successful, False otherwise
        """
        # Find the dialogue containing this message
        for dialogue in self.dialogues.values():
            for message in dialogue.messages:
                if message.id == message_id:
                    # Found the message, update its status
                    if new_status in [MessageStatus.RECEIVED, MessageStatus.READ]:
                        # These statuses are tracked per recipient, would need recipient-specific status tracking
                        # For now, just update the message's overall status if it's a stronger status
                        if MessageStatus[message.status].index < MessageStatus[new_status].index:
                            message.status = new_status
                    else:
                        # Other statuses apply to the whole message
                        message.status = new_status
                    return True
        
        return False
    
    def get_current_speaker(self, dialogue_id: UUID) -> Optional[UUID]:
        """
        Get the ID of the participant whose turn it is to speak.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            ID of the current speaker or None
        """
        state_manager = self.state_managers.get(dialogue_id)
        if state_manager:
            return state_manager.get_current_speaker()
        return None
    
    def set_next_speaker(self, dialogue_id: UUID, participant_id: UUID) -> bool:
        """
        Set the next speaker for a dialogue (facilitator function).
        
        Args:
            dialogue_id: ID of the dialogue
            participant_id: ID of the participant to speak next
            
        Returns:
            True if successful, False otherwise
        """
        state_manager = self.state_managers.get(dialogue_id)
        if state_manager:
            return state_manager.set_next_speaker(participant_id)
        return False
    
    def change_dialogue_state(self, dialogue_id: UUID, new_state: DialogueState) -> bool:
        """
        Change the state of a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            new_state: New state to set
            
        Returns:
            True if successful, False otherwise
        """
        state_manager = self.state_managers.get(dialogue_id)
        if state_manager:
            return state_manager.change_state(new_state)
        return False
    
    def get_pending_deliveries(self, dialogue_id: UUID) -> List[DeliveryReport]:
        """
        Get pending delivery reports for a dialogue.
        
        Args:
            dialogue_id: ID of the dialogue
            
        Returns:
            List of delivery reports
        """
        return self.pending_deliveries.get(dialogue_id, [])