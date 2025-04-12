"""
Base adapter interface for Fire Circle.

This module defines the base adapter interface and abstract classes that all provider-specific
adapters must implement to ensure consistent behavior across different AI model providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import asyncio

from pydantic import BaseModel

from firecircle.protocol.message import Message, MessageContent, Participant, Dialogue


class AdapterConfig(BaseModel):
    """Configuration for an AI model adapter."""
    
    api_key: str
    base_url: Optional[str] = None
    organization_id: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    

class ModelCapabilities(BaseModel):
    """Capabilities of an AI model."""
    
    max_tokens: int
    context_window: int
    supports_functions: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_system_messages: bool = True
    supports_empty_chair: bool = False  # Special capability for Fire Circle


class ConnectionStatus(BaseModel):
    """Status of a connection to an AI service provider."""
    
    connected: bool
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    rate_limited: bool = False
    quota_remaining: Optional[float] = None
    
    
class ModelAdapter(ABC):
    """Base interface for AI model adapters."""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self._connection_status = ConnectionStatus(connected=False)
        
    @property
    def connection_status(self) -> ConnectionStatus:
        """Get the current connection status."""
        return self._connection_status
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the AI service provider."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the AI service provider."""
        pass
    
    @abstractmethod
    async def list_available_models(self) -> List[str]:
        """List available models from this provider."""
        pass
    
    @abstractmethod
    async def get_model_capabilities(self, model_id: str) -> ModelCapabilities:
        """Get capabilities of a specific model."""
        pass
    
    @abstractmethod
    async def send_message(self, 
                          message: Message, 
                          dialogue: Dialogue,
                          stream: bool = False) -> Message:
        """
        Send a message to the AI model and get a response.
        
        Args:
            message: The message to send to the AI model
            dialogue: The dialogue context for the message
            stream: Whether to stream the response
            
        Returns:
            A Message object containing the AI model's response
        """
        pass
    
    @abstractmethod
    async def stream_message(self, 
                            message: Message,
                            dialogue: Dialogue) -> asyncio.AsyncIterator[MessageContent]:
        """
        Stream a message to the AI model and get a streaming response.
        
        Args:
            message: The message to send to the AI model
            dialogue: The dialogue context for the message
            
        Returns:
            An async iterator yielding message content chunks
        """
        pass
    
    @abstractmethod
    def message_to_provider_format(self, 
                                  message: Message, 
                                  dialogue: Dialogue) -> Dict[str, Any]:
        """
        Convert a Fire Circle message to the provider's format.
        
        Args:
            message: The message to convert
            dialogue: The dialogue context for the message
            
        Returns:
            A dictionary in the provider's expected format
        """
        pass
    
    @abstractmethod
    def provider_response_to_message(self, 
                                    response: Any, 
                                    source_message: Message,
                                    participant: Participant) -> Message:
        """
        Convert a provider response to a Fire Circle message.
        
        Args:
            response: The provider's response
            source_message: The original message that prompted this response
            participant: The participant (AI model) that generated the response
            
        Returns:
            A Fire Circle Message object
        """
        pass


class AdapterRegistry:
    """Registry of available model adapters."""
    
    _adapters: Dict[str, ModelAdapter] = {}
    
    @classmethod
    def register(cls, name: str, adapter: ModelAdapter) -> None:
        """Register an adapter with the registry."""
        cls._adapters[name] = adapter
        
    @classmethod
    def get(cls, name: str) -> Optional[ModelAdapter]:
        """Get an adapter by name."""
        return cls._adapters.get(name)
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapters."""
        return list(cls._adapters.keys())


class AdapterFactory:
    """Factory for creating adapters."""
    
    @staticmethod
    async def create_adapter(provider: str, config: AdapterConfig) -> ModelAdapter:
        """
        Create and initialize an adapter for the specified provider.
        
        Args:
            provider: The name of the provider (e.g., "openai", "anthropic")
            config: Configuration for the adapter
            
        Returns:
            An initialized ModelAdapter instance
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider.lower() == "openai":
            # Dynamically import to avoid circular imports
            from firecircle.adapters.openai.adapter import OpenAIAdapter
            adapter = OpenAIAdapter(config)
        elif provider.lower() == "anthropic":
            from firecircle.adapters.anthropic.adapter import AnthropicAdapter
            adapter = AnthropicAdapter(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Initialize the connection
        await adapter.connect()
        
        # Register the adapter
        AdapterRegistry.register(provider, adapter)
        
        return adapter