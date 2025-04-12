"""
Turn Policy implementation for Fire Circle.

This module defines the various turn-taking policies that control
the flow of conversation between participants in a Fire Circle dialogue.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID

import random


class TurnPolicy(str, Enum):
    """Turn-taking policies for dialogue management."""
    
    ROUND_ROBIN = "round_robin"    # Participants speak in a fixed order
    FACILITATOR = "facilitator"    # Facilitator decides who speaks next
    FREE_FORM = "free_form"        # Anyone can speak at any time
    REACTIVE = "reactive"          # Participants respond to being addressed
    CONSENSUS = "consensus"        # All must contribute before proceeding


class TurnManager:
    """
    Manages turn-taking between participants according to a specified policy.
    
    The TurnManager is responsible for:
    - Determining whose turn it is to speak
    - Advancing turns based on the chosen policy
    - Tracking turn counts and ensuring fair participation
    """
    
    def __init__(
        self, 
        policy: TurnPolicy,
        participant_ids: List[UUID],
        facilitator_id: Optional[UUID] = None,
        max_consecutive_turns: int = 1,
        randomize_initial_order: bool = True
    ):
        """
        Initialize the turn manager.
        
        Args:
            policy: The turn-taking policy to use
            participant_ids: List of participant IDs
            facilitator_id: ID of the facilitator (required for FACILITATOR policy)
            max_consecutive_turns: Maximum number of consecutive turns allowed
            randomize_initial_order: Whether to randomize the initial order
        """
        self.policy = policy
        self.participant_ids = participant_ids.copy()
        self.facilitator_id = facilitator_id
        self.max_consecutive_turns = max_consecutive_turns
        
        # Track the speaking order
        self.turn_order: List[UUID] = []
        
        # Track the current position in the order
        self.current_position = 0
        
        # Track how many turns each participant has taken
        self.turns_taken: Dict[UUID, int] = {pid: 0 for pid in participant_ids}
        
        # Track the ID of the participant who spoke last
        self.last_speaker: Optional[UUID] = None
        
        # Track consecutive turns by the same speaker
        self.consecutive_turns = 0
        
        # Set up the turn order based on the policy
        self._initialize_turn_order(randomize_initial_order)
    
    def _initialize_turn_order(self, randomize: bool = True) -> None:
        """
        Initialize the turn order based on the policy.
        
        Args:
            randomize: Whether to randomize the initial order
        """
        if self.policy == TurnPolicy.ROUND_ROBIN:
            # Create an ordered list of all participants
            self.turn_order = self.participant_ids.copy()
            
            # Randomize if requested
            if randomize:
                random.shuffle(self.turn_order)
            
            # If there's a facilitator, move them to the front
            if self.facilitator_id in self.turn_order:
                self.turn_order.remove(self.facilitator_id)
                self.turn_order.insert(0, self.facilitator_id)
                
        elif self.policy == TurnPolicy.FACILITATOR:
            # Only the facilitator is in the turn order initially
            if not self.facilitator_id:
                raise ValueError("Facilitator policy requires a facilitator_id")
            self.turn_order = [self.facilitator_id]
            
        elif self.policy == TurnPolicy.FREE_FORM:
            # No enforced order, but initialize with all participants
            self.turn_order = self.participant_ids.copy()
            if randomize:
                random.shuffle(self.turn_order)
            
        elif self.policy == TurnPolicy.REACTIVE:
            # Start with facilitator or first participant
            if self.facilitator_id:
                self.turn_order = [self.facilitator_id]
            else:
                # Take the first participant (possibly after randomizing)
                order = self.participant_ids.copy()
                if randomize:
                    random.shuffle(order)
                self.turn_order = [order[0]]
                
        elif self.policy == TurnPolicy.CONSENSUS:
            # All participants must contribute
            self.turn_order = self.participant_ids.copy()
            if randomize:
                random.shuffle(self.turn_order)
            
            # If there's a facilitator, move them to the front
            if self.facilitator_id in self.turn_order:
                self.turn_order.remove(self.facilitator_id)
                self.turn_order.insert(0, self.facilitator_id)
    
    def get_current_speaker(self) -> Optional[UUID]:
        """
        Get the ID of the participant whose turn it is to speak.
        
        Returns:
            ID of the current speaker or None if no current speaker
        """
        if not self.turn_order:
            return None
            
        if 0 <= self.current_position < len(self.turn_order):
            return self.turn_order[self.current_position]
            
        return None
    
    def can_speak(self, participant_id: UUID) -> bool:
        """
        Check if a participant is allowed to speak now.
        
        Args:
            participant_id: ID of the participant
            
        Returns:
            True if the participant can speak, False otherwise
        """
        # Apply policy-specific rules
        if self.policy == TurnPolicy.FREE_FORM:
            # Anyone can speak in free-form
            return True
            
        elif self.policy == TurnPolicy.FACILITATOR:
            # Either the facilitator or whoever they've given the floor to
            return (
                participant_id == self.facilitator_id or 
                participant_id == self.get_current_speaker()
            )
            
        elif self.policy == TurnPolicy.ROUND_ROBIN:
            # Only the current speaker in the rotation
            current_speaker = self.get_current_speaker()
            
            # Check consecutive turn limit if it's the same speaker
            if participant_id == self.last_speaker:
                return self.consecutive_turns < self.max_consecutive_turns
                
            # Otherwise, check if it's their turn
            return participant_id == current_speaker
            
        elif self.policy == TurnPolicy.REACTIVE:
            # Current speaker or anyone responding to them
            return participant_id == self.get_current_speaker()
            
        elif self.policy == TurnPolicy.CONSENSUS:
            # Anyone who hasn't spoken yet in this round, or the facilitator
            if participant_id == self.facilitator_id:
                return True
                
            # Check if they've spoken less than others
            if participant_id in self.turns_taken:
                min_turns = min(self.turns_taken.values())
                return self.turns_taken[participant_id] <= min_turns
        
        return False
    
    def record_turn(self, participant_id: UUID) -> None:
        """
        Record that a participant has taken their turn.
        
        Args:
            participant_id: ID of the participant who spoke
        """
        # Update turn counts
        if participant_id in self.turns_taken:
            self.turns_taken[participant_id] += 1
        
        # Update consecutive turn counter
        if participant_id == self.last_speaker:
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 1
        
        # Record the last speaker
        self.last_speaker = participant_id
        
        # Advance the turn based on the policy
        self._advance_turn(participant_id)
    
    def _advance_turn(self, speaker_id: UUID) -> None:
        """
        Advance to the next participant's turn.
        
        Args:
            speaker_id: ID of the participant who just spoke
        """
        if self.policy == TurnPolicy.FREE_FORM:
            # No turn advancement in free-form
            pass
            
        elif self.policy == TurnPolicy.FACILITATOR:
            # Only the facilitator changes turns
            if speaker_id == self.facilitator_id:
                # Facilitator spoke, so reset to facilitator for next turn
                self.current_position = 0  # Facilitator is at index 0
                
        elif self.policy == TurnPolicy.ROUND_ROBIN:
            # Move to next participant in the rotation
            if speaker_id == self.get_current_speaker():
                self.current_position = (self.current_position + 1) % len(self.turn_order)
                
        elif self.policy == TurnPolicy.REACTIVE:
            # Next speaker is determined dynamically based on who was addressed
            # This is handled elsewhere based on message content and references
            pass
            
        elif self.policy == TurnPolicy.CONSENSUS:
            # Check if we've completed a round (everyone spoke)
            min_turns = min(self.turns_taken.values())
            max_turns = max(self.turns_taken.values())
            
            if min_turns == max_turns:
                # Everyone has spoken the same number of times, start new round
                self.current_position = 0
    
    def set_next_speaker(self, participant_id: UUID) -> bool:
        """
        Explicitly set the next speaker (used by facilitator).
        
        Args:
            participant_id: ID of the participant to speak next
            
        Returns:
            True if successful, False if not allowed
        """
        # Check if we have a facilitator policy or the caller is the facilitator
        if self.policy not in [TurnPolicy.FACILITATOR, TurnPolicy.REACTIVE]:
            return False
            
        # Find the participant in the turn order
        if participant_id in self.turn_order:
            self.current_position = self.turn_order.index(participant_id)
            return True
            
        # Add them to the turn order if not present (for reactive policy)
        if participant_id in self.participant_ids:
            self.turn_order.append(participant_id)
            self.current_position = len(self.turn_order) - 1
            return True
            
        return False
    
    def reset_turns(self) -> None:
        """Reset the turn counts for all participants."""
        for pid in self.turns_taken:
            self.turns_taken[pid] = 0
        self.current_position = 0
        self.consecutive_turns = 0
        self.last_speaker = None
    
    def add_participant(self, participant_id: UUID) -> None:
        """
        Add a new participant to the dialogue.
        
        Args:
            participant_id: ID of the participant to add
        """
        if participant_id not in self.participant_ids:
            self.participant_ids.append(participant_id)
            self.turns_taken[participant_id] = 0
            
            # Add to turn order based on policy
            if self.policy in [TurnPolicy.ROUND_ROBIN, TurnPolicy.FREE_FORM, TurnPolicy.CONSENSUS]:
                self.turn_order.append(participant_id)
    
    def remove_participant(self, participant_id: UUID) -> None:
        """
        Remove a participant from the dialogue.
        
        Args:
            participant_id: ID of the participant to remove
        """
        if participant_id in self.participant_ids:
            self.participant_ids.remove(participant_id)
            
            if participant_id in self.turns_taken:
                del self.turns_taken[participant_id]
                
            if participant_id in self.turn_order:
                position = self.turn_order.index(participant_id)
                self.turn_order.remove(participant_id)
                
                # Adjust current position if necessary
                if position < self.current_position:
                    self.current_position -= 1
                elif position == self.current_position:
                    self.current_position = self.current_position % len(self.turn_order) if self.turn_order else 0