"""
Dialogue State Management for Fire Circle.

This module defines the various states and phases a dialogue can be in
within the Fire Circle system, along with state transition rules.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple


class DialogueState(str, Enum):
    """
    States of a dialogue within the Fire Circle system.
    
    These represent the operational status of the dialogue.
    """
    
    INITIALIZING = "initializing"  # Setting up, gathering participants
    ACTIVE = "active"              # Normal dialogue flow
    PAUSED = "paused"              # Temporarily suspended
    CONCLUDING = "concluding"      # Final statements being made
    COMPLETED = "completed"        # Dialogue has ended normally
    ABANDONED = "abandoned"        # Dialogue ended prematurely
    ERROR = "error"                # Error state


class DialoguePhase(str, Enum):
    """
    Dialogue phases representing different modes of conversation.
    
    These represent the semantic structure of the dialogue, independent
    of its operational state.
    """
    
    INTRODUCTION = "introduction"        # Initial setup and framing
    EXPLORATION = "exploration"          # Open-ended discussion
    FOCUSING = "focusing"                # Narrowing down the topic
    DELIBERATION = "deliberation"        # Weighing options and perspectives
    REFLECTION = "reflection"            # Meta-commentary on the dialogue
    SUMMARIZATION = "summarization"      # Collecting key insights
    VOTING = "voting"                    # Deciding on consensus
    CONCLUSION = "conclusion"            # Final reflections


class TransitionError(Exception):
    """Exception raised for invalid state transitions."""
    pass


# Define valid state transitions
VALID_STATE_TRANSITIONS: Dict[DialogueState, Set[DialogueState]] = {
    DialogueState.INITIALIZING: {DialogueState.ACTIVE, DialogueState.ABANDONED, DialogueState.ERROR},
    DialogueState.ACTIVE: {DialogueState.PAUSED, DialogueState.CONCLUDING, DialogueState.ABANDONED, DialogueState.ERROR},
    DialogueState.PAUSED: {DialogueState.ACTIVE, DialogueState.CONCLUDING, DialogueState.ABANDONED, DialogueState.ERROR},
    DialogueState.CONCLUDING: {DialogueState.COMPLETED, DialogueState.ACTIVE, DialogueState.ABANDONED, DialogueState.ERROR},
    DialogueState.COMPLETED: {DialogueState.ERROR},  # Very restricted once completed
    DialogueState.ABANDONED: {DialogueState.ACTIVE, DialogueState.ERROR},  # Can resume an abandoned dialogue
    DialogueState.ERROR: {DialogueState.INITIALIZING, DialogueState.ACTIVE},  # Can recover from errors
}

# Define valid phase transitions
VALID_PHASE_TRANSITIONS: Dict[DialoguePhase, Set[DialoguePhase]] = {
    DialoguePhase.INTRODUCTION: {DialoguePhase.EXPLORATION, DialoguePhase.FOCUSING},
    DialoguePhase.EXPLORATION: {DialoguePhase.FOCUSING, DialoguePhase.DELIBERATION, DialoguePhase.REFLECTION},
    DialoguePhase.FOCUSING: {DialoguePhase.DELIBERATION, DialoguePhase.EXPLORATION, DialoguePhase.REFLECTION},
    DialoguePhase.DELIBERATION: {DialoguePhase.REFLECTION, DialoguePhase.SUMMARIZATION, DialoguePhase.VOTING},
    DialoguePhase.REFLECTION: {DialoguePhase.EXPLORATION, DialoguePhase.FOCUSING, DialoguePhase.DELIBERATION, DialoguePhase.SUMMARIZATION},
    DialoguePhase.SUMMARIZATION: {DialoguePhase.VOTING, DialoguePhase.CONCLUSION, DialoguePhase.REFLECTION},
    DialoguePhase.VOTING: {DialoguePhase.CONCLUSION, DialoguePhase.DELIBERATION, DialoguePhase.SUMMARIZATION},
    DialoguePhase.CONCLUSION: {DialoguePhase.INTRODUCTION},  # Can start a new cycle
}


def validate_state_transition(current_state: DialogueState, new_state: DialogueState) -> bool:
    """
    Validate whether a state transition is allowed.
    
    Args:
        current_state: The current dialogue state
        new_state: The proposed new state
        
    Returns:
        True if the transition is valid, False otherwise
    """
    if current_state == new_state:
        return True  # Same state is always valid
        
    valid_transitions = VALID_STATE_TRANSITIONS.get(current_state, set())
    return new_state in valid_transitions


def validate_phase_transition(current_phase: DialoguePhase, new_phase: DialoguePhase) -> bool:
    """
    Validate whether a phase transition is allowed.
    
    Args:
        current_phase: The current dialogue phase
        new_phase: The proposed new phase
        
    Returns:
        True if the transition is valid, False otherwise
    """
    if current_phase == new_phase:
        return True  # Same phase is always valid
        
    valid_transitions = VALID_PHASE_TRANSITIONS.get(current_phase, set())
    return new_phase in valid_transitions


def get_valid_next_states(current_state: DialogueState) -> List[DialogueState]:
    """
    Get all valid states that can follow the current state.
    
    Args:
        current_state: The current dialogue state
        
    Returns:
        List of valid next states
    """
    return list(VALID_STATE_TRANSITIONS.get(current_state, set()))


def get_valid_next_phases(current_phase: DialoguePhase) -> List[DialoguePhase]:
    """
    Get all valid phases that can follow the current phase.
    
    Args:
        current_phase: The current dialogue phase
        
    Returns:
        List of valid next phases
    """
    return list(VALID_PHASE_TRANSITIONS.get(current_phase, set()))