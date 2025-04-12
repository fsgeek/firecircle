"""
Message Protocol for Fire Circle

This module defines the standardized message format for communication between 
AI models in the Fire Circle system. It implements the core protocol layer for 
structured dialogue based on principles of reciprocity (ayni).

The message protocol supports:
- Consistent message structure across different AI providers
- Rich metadata for tracking conversation flow
- Support for different message types (message, question, proposal, etc.)
- Attribution and perspective tracking
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class MessageType(str, Enum):
    """Types of messages that can be exchanged in a Fire Circle dialogue."""
    
    MESSAGE = "message"              # General communication
    QUESTION = "question"            # Specific inquiry requiring response
    PROPOSAL = "proposal"            # Suggestion for consideration
    AGREEMENT = "agreement"          # Expression of consensus
    DISAGREEMENT = "disagreement"    # Expression of dissent
    REFLECTION = "reflection"        # Meta-commentary on the dialogue
    SUMMARY = "summary"              # Condensed representation of points
    CLARIFICATION = "clarification"  # Request for or provision of clarification
    EMPTY_CHAIR = "empty_chair"      # Perspective of unrepresented viewpoint
    CONCLUSION = "conclusion"        # Closing statement or summation
    SYSTEM = "system"                # System-level control messages


class MessageRole(str, Enum):
    """Roles that can generate messages within the dialogue."""
    
    SYSTEM = "system"        # System messages and control flow
    USER = "user"            # Human participant input
    ASSISTANT = "assistant"  # AI model responses
    FUNCTION = "function"    # Tool or function execution results
    PERSPECTIVE = "perspective"  # Distinct perspective/persona used by an AI


class MessageStatus(str, Enum):
    """Status indicators for messages within the dialogue flow."""
    
    DRAFT = "draft"              # Initial composition, not yet shared
    SENT = "sent"                # Dispatched to recipients
    RECEIVED = "received"        # Confirmed receipt
    READ = "read"                # Confirmed to have been processed
    RESPONDED = "responded"      # Has received a direct response
    RETRACTED = "retracted"      # Withdrawn by the author
    FLAGGED = "flagged"          # Marked for special attention
    ERROR = "error"              # Problem encountered with message


class MessagePriority(str, Enum):
    """Priority levels for messages to indicate importance or urgency."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageSentiment(str, Enum):
    """
    Sentiment indicators to provide emotional context.
    This helps capture the intended tone and avoid misinterpretation.
    """
    
    NEUTRAL = "neutral"
    APPRECIATIVE = "appreciative"
    CURIOUS = "curious"
    CONCERNED = "concerned"
    SUPPORTIVE = "supportive"
    CHALLENGING = "challenging"
    REFLECTIVE = "reflective"


class MessageVisibility(str, Enum):
    """
    Visibility levels controlling which participants can see a message.
    This allows for selective sharing within the dialogue circle.
    """
    
    ALL = "all"              # Visible to all participants
    FACILITATOR = "facilitator"  # Only visible to facilitator and sender
    DIRECT = "direct"        # Only visible to specified recipients
    PRIVATE = "private"      # Only visible to sender (for notes)


class Participant(BaseModel):
    """
    Represents a participant in the dialogue, either AI model or human.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the participant")
    name: str = Field(..., description="Display name of the participant")
    type: str = Field(..., description="Type of participant (e.g., 'ai_model', 'human')")
    provider: Optional[str] = Field(None, description="Provider for AI models (e.g., 'openai', 'anthropic')")
    model: Optional[str] = Field(None, description="Specific model identifier if applicable")
    capabilities: List[str] = Field(default_factory=list, description="Capabilities this participant provides")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional participant metadata")
    
    @field_validator('name')
    def name_must_not_be_empty(cls, v):
        """Validate that the name is not empty."""
        if not v or not v.strip():
            raise ValueError("Participant name cannot be empty")
        return v.strip()


class MessageMetadata(BaseModel):
    """
    Rich metadata for tracking conversation context and flow.
    """
    
    dialogue_id: UUID = Field(..., description="Identifier for the dialogue this message belongs to")
    sequence_number: int = Field(..., description="Position in dialogue sequence")
    turn_number: int = Field(..., description="Which conversational turn this occurs in")
    in_response_to: Optional[UUID] = Field(None, description="ID of message this responds to")
    references: List[UUID] = Field(default_factory=list, description="IDs of messages referenced")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")
    edited_timestamp: Optional[datetime] = Field(None, description="When the message was last edited")
    language: str = Field(default="en", description="ISO language code")
    topics: List[str] = Field(default_factory=list, description="Topic tags for the message")
    sentiment: MessageSentiment = Field(default=MessageSentiment.NEUTRAL, description="Emotional tone")
    visibility: MessageVisibility = Field(default=MessageVisibility.ALL, description="Who can see this message")
    recipients: List[UUID] = Field(default_factory=list, description="Specific recipients if visibility is DIRECT")
    custom_properties: Dict[str, Any] = Field(default_factory=dict, description="Extension point for custom metadata")


class MessageContent(BaseModel):
    """
    The actual content of a message, supporting multiple formats and parts.
    """
    
    text: str = Field(..., description="Plain text content of the message")
    html: Optional[str] = Field(None, description="HTML formatted version if available")
    parts: List[Dict[str, Any]] = Field(default_factory=list, description="Structured content parts")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Attached content references")
    citation_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Citations for factual claims")
    
    @field_validator('text')
    def text_must_not_be_empty(cls, v):
        """Validate that the text content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message text cannot be empty")
        return v


class ToolCall(BaseModel):
    """
    Representation of a call to an external tool or function.
    """
    
    tool_id: str = Field(..., description="Identifier of the tool being called")
    name: str = Field(..., description="Name of the specific function or method")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Result returned by the tool if executed")
    error: Optional[str] = Field(None, description="Error message if the tool call failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class Message(BaseModel):
    """
    The core message model representing a single communication unit in the dialogue.
    
    This comprehensive message format is designed to support rich, structured dialogue
    between multiple AI models and human participants. It emphasizes context, attribution,
    and relationship to the broader conversation.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this message")
    type: MessageType = Field(..., description="Type of message")
    role: MessageRole = Field(..., description="Role of the message sender")
    sender: UUID = Field(..., description="ID of the participant who sent this message")
    content: MessageContent = Field(..., description="Content of the message")
    metadata: MessageMetadata = Field(..., description="Message metadata and context")
    status: MessageStatus = Field(default=MessageStatus.DRAFT, description="Current status of the message")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    perspective: Optional[str] = Field(None, description="Specific perspective or persona taken")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls made in this message")
    reasoning: Optional[str] = Field(None, description="Explanation of reasoning or thought process")
    deleted: bool = Field(default=False, description="Whether this message has been deleted")
    
    def create_response(
        self, 
        content: Union[str, MessageContent], 
        type: MessageType = MessageType.MESSAGE,
        sender: UUID = None,
        role: MessageRole = None,
        perspective: Optional[str] = None
    ) -> "Message":
        """
        Create a response to this message, maintaining dialogue linkage.
        
        Args:
            content: Text content or MessageContent object
            type: Type of the response message
            sender: ID of the sender (required if different from original receiver)
            role: Role of the sender (required if different from original receiver)
            perspective: Optional specific perspective for the response
            
        Returns:
            A new Message object linked to this one as a response
        """
        if isinstance(content, str):
            content = MessageContent(text=content)
            
        # Create new metadata linked to this message
        new_metadata = MessageMetadata(
            dialogue_id=self.metadata.dialogue_id,
            sequence_number=self.metadata.sequence_number + 1,
            turn_number=self.metadata.turn_number + 1 if type not in [MessageType.CLARIFICATION, MessageType.REFLECTION] else self.metadata.turn_number,
            in_response_to=self.id,
            references=[self.id],
            visibility=self.metadata.visibility,
            recipients=self.metadata.recipients
        )
        
        # Create response message
        return Message(
            type=type,
            role=role or self.role,
            sender=sender or self.sender,
            content=content,
            metadata=new_metadata,
            perspective=perspective
        )


class Dialogue(BaseModel):
    """
    A complete dialogue session containing all messages and participant information.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this dialogue")
    title: str = Field(..., description="Title or topic of the dialogue")
    participants: List[Participant] = Field(..., description="All participants in the dialogue")
    messages: List[Message] = Field(default_factory=list, description="All messages in the dialogue")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the dialogue was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the dialogue was last updated")
    status: str = Field(default="active", description="Status of the dialogue")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional dialogue metadata")
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the dialogue and update sequence metadata.
        
        Args:
            message: The message to add
        """
        # Make sure the message references this dialogue
        if message.metadata.dialogue_id != self.id:
            raise ValueError(f"Message dialogue ID {message.metadata.dialogue_id} does not match dialogue ID {self.id}")
        
        # Update sequence number if not explicitly set
        if message.metadata.sequence_number == 0:
            message.metadata.sequence_number = len(self.messages) + 1
        
        # Add message and update dialogue timestamp
        self.messages.append(message)
        self.updated_at = datetime.utcnow()


def create_system_message(
    dialogue_id: UUID,
    content: str,
    sequence_number: int = 1,
    metadata: Optional[Dict[str, Any]] = None
) -> Message:
    """
    Create a system message to establish dialogue context and guidelines.
    
    Args:
        dialogue_id: ID of the dialogue this message belongs to
        content: Text content of the system message
        sequence_number: Position in the dialogue sequence
        metadata: Additional custom metadata
        
    Returns:
        A formatted system message
    """
    # Create message metadata
    msg_metadata = MessageMetadata(
        dialogue_id=dialogue_id,
        sequence_number=sequence_number,
        turn_number=0,  # System messages don't count as turns
        custom_properties=metadata or {}
    )
    
    # Create system participant ID (consistent across all dialogues)
    system_id = UUID('00000000-0000-0000-0000-000000000000')
    
    # Create and return the message
    return Message(
        type=MessageType.SYSTEM,
        role=MessageRole.SYSTEM,
        sender=system_id,
        content=MessageContent(text=content),
        metadata=msg_metadata
    )


def create_empty_chair_message(
    dialogue_id: UUID,
    content: str,
    perspective: str,
    sender: UUID,
    sequence_number: int,
    turn_number: int,
    in_response_to: Optional[UUID] = None
) -> Message:
    """
    Create a message representing an unrepresented perspective (the "empty chair").
    
    The Empty Chair concept reserves space for perspectives not present in the dialogue,
    helping to avoid blind spots and groupthink.
    
    Args:
        dialogue_id: ID of the dialogue this message belongs to
        content: Text content of the message
        perspective: Description of the perspective being represented
        sender: ID of the participant creating this perspective
        sequence_number: Position in the dialogue sequence
        turn_number: Which conversational turn this occurs in
        in_response_to: Optional ID of message this responds to
        
    Returns:
        A formatted Empty Chair message
    """
    # Create message metadata
    msg_metadata = MessageMetadata(
        dialogue_id=dialogue_id,
        sequence_number=sequence_number,
        turn_number=turn_number,
        in_response_to=in_response_to,
        references=[in_response_to] if in_response_to else []
    )
    
    # Create and return the message
    return Message(
        type=MessageType.EMPTY_CHAIR,
        role=MessageRole.PERSPECTIVE,
        sender=sender,
        content=MessageContent(text=content),
        metadata=msg_metadata,
        perspective=perspective
    )


def create_dialogue(title: str, participants: List[Participant], system_message: str = None) -> Dialogue:
    """
    Create a new dialogue with the given participants and optional system message.
    
    Args:
        title: Title or topic of the dialogue
        participants: List of participants to include
        system_message: Optional system message to establish context
        
    Returns:
        A new Dialogue object
    """
    dialogue = Dialogue(
        title=title,
        participants=participants
    )
    
    # Add system message if provided
    if system_message:
        msg = create_system_message(
            dialogue_id=dialogue.id,
            content=system_message
        )
        dialogue.add_message(msg)
    
    return dialogue