"""
Anthropic adapter implementation for Fire Circle.

This module provides an adapter for connecting the Fire Circle protocol with Anthropic's API,
translating between the standardized Fire Circle message format and Anthropic's API requirements.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator

import anthropic
from anthropic.types import (
    Message as AnthropicMessage,
    MessageParam,
    ContentBlock,
    TextBlock
)

from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    Participant, 
    Dialogue, 
    MessageType,
    MessageRole,
    create_system_message
)
from firecircle.adapters.base.adapter import (
    ModelAdapter,
    AdapterConfig,
    ModelCapabilities,
    ConnectionStatus
)


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.client = None
        self._model_capabilities_cache: Dict[str, ModelCapabilities] = {}
        
    async def connect(self) -> bool:
        """Establish connection to Anthropic API."""
        try:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            
            # Test connection - Anthropic doesn't have a list_models endpoint,
            # so we'll just verify the API key works by getting model info
            start_time = time.time()
            # Simple request to check if API key works
            await self.client.messages.retrieve("msg_0123456789")  # Will fail, but checks auth
            end_time = time.time()
            
            self._connection_status = ConnectionStatus(
                connected=True,
                latency_ms=(end_time - start_time) * 1000
            )
            return True
        except anthropic.BadRequestError:
            # This is expected since we're using a fake message ID
            # But it means the auth works
            self._connection_status = ConnectionStatus(
                connected=True,
                latency_ms=0  # Will be updated on first actual request
            )
            return True
        except Exception as e:
            self._connection_status = ConnectionStatus(
                connected=False,
                last_error=str(e)
            )
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to Anthropic API."""
        # Anthropic client doesn't require explicit disconnection
        return True
    
    async def list_available_models(self) -> List[str]:
        """List available models from Anthropic."""
        # Anthropic doesn't have a list_models endpoint, so we'll return known models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    async def get_model_capabilities(self, model_id: str) -> ModelCapabilities:
        """Get capabilities of a specific Anthropic model."""
        # Return from cache if available
        if model_id in self._model_capabilities_cache:
            return self._model_capabilities_cache[model_id]
        
        # Define capabilities by model
        if "claude-3-opus" in model_id:
            capabilities = ModelCapabilities(
                max_tokens=4096,  # Output token limit
                context_window=200000,
                supports_functions=True,
                supports_vision=True,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=True  # Claude's multi-perspective capability
            )
        elif "claude-3-sonnet" in model_id:
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=200000,
                supports_functions=True,
                supports_vision=True,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=True
            )
        elif "claude-3-haiku" in model_id:
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=200000,
                supports_functions=True,
                supports_vision=True,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=True
            )
        elif "claude-2.1" in model_id:
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=100000,
                supports_functions=False,
                supports_vision=False,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=False
            )
        elif "claude-2.0" in model_id:
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=100000,
                supports_functions=False,
                supports_vision=False,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=False
            )
        elif "claude-instant" in model_id:
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=100000,
                supports_functions=False,
                supports_vision=False,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=False
            )
        else:
            # Default for unknown models
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=100000,
                supports_functions=False,
                supports_vision=False,
                supports_streaming=True,
                supports_system_messages=True,
                supports_empty_chair=False
            )
            
        # Cache the capabilities
        self._model_capabilities_cache[model_id] = capabilities
        return capabilities
    
    def _convert_message_role(self, role: MessageRole) -> str:
        """Convert Fire Circle message role to Anthropic role."""
        role_mapping = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant"
            # Note: Anthropic doesn't have direct system or function roles
        }
        
        return role_mapping.get(role, "user")
    
    def _extract_model_id(self, dialogue: Dialogue, participant: Participant) -> str:
        """Extract the appropriate model ID for a given participant."""
        # Default to the most capable model if not specified
        default_model = "claude-3-opus-20240229"
        
        # Check participant configuration for model ID
        if participant and participant.provider_config:
            model_id = participant.provider_config.get("model_id")
            if model_id:
                return model_id
        
        # Check dialogue configuration
        if dialogue and dialogue.metadata and dialogue.metadata.provider_config:
            model_id = dialogue.metadata.provider_config.get("anthropic_model_id")
            if model_id:
                return model_id
                
        return default_model
    
    def message_to_provider_format(self, 
                                  message: Message, 
                                  dialogue: Dialogue) -> Dict[str, Any]:
        """Convert a Fire Circle message to Anthropic's format."""
        # Extract the model to use
        recipient = next((p for p in dialogue.participants 
                         if p.id == message.metadata.recipient_id), None)
        model = self._extract_model_id(dialogue, recipient)
        
        # Build the message history in Anthropic format
        messages = []
        system_prompt = None
        
        # Extract system message if available
        system_messages = [m for m in dialogue.messages 
                          if m.metadata.role == MessageRole.SYSTEM]
        if system_messages:
            # Get the most recent system message
            system_prompt = system_messages[-1].content.text
        
        # Add dialogue history (excluding system messages)
        for hist_msg in dialogue.messages:
            if hist_msg.metadata.role == MessageRole.SYSTEM:
                continue
                
            # Skip messages that aren't visible to the recipient
            if (recipient and hist_msg.metadata.visibility and 
                recipient.id not in hist_msg.metadata.visibility):
                continue
            
            # Anthropic only supports user and assistant roles
            role = self._convert_message_role(hist_msg.metadata.role)
            if role not in ["user", "assistant"]:
                continue
                
            # Create Anthropic message
            content = [{"type": "text", "text": hist_msg.content.text}]
            
            messages.append({
                "role": role,
                "content": content
            })
        
        # Add the current message if it's not already included
        if message not in dialogue.messages:
            role = self._convert_message_role(message.metadata.role)
            if role in ["user", "assistant"]:
                content = [{"type": "text", "text": message.content.text}]
                
                messages.append({
                    "role": role,
                    "content": content
                })
        
        # Build the complete request
        request = {
            "model": model,
            "messages": messages,
            "system": system_prompt,
            "temperature": message.metadata.provider_config.get("temperature", 0.7) if message.metadata.provider_config else 0.7,
            "max_tokens": message.metadata.provider_config.get("max_tokens", 4096) if message.metadata.provider_config else 4096,
        }
        
        return request
    
    def provider_response_to_message(self, 
                                    response: Union[AnthropicMessage, Dict[str, Any]], 
                                    source_message: Message,
                                    participant: Participant) -> Message:
        """Convert an Anthropic response to a Fire Circle message."""
        # Handle full completion responses
        if isinstance(response, AnthropicMessage):
            # Extract text content from blocks
            content = ""
            if response.content:
                for block in response.content:
                    if isinstance(block, TextBlock) or (hasattr(block, "type") and block.type == "text"):
                        content += block.text
            
            # Create a new Fire Circle message
            return Message(
                id=response.id,
                content=MessageContent(text=content),
                metadata={
                    "sender_id": participant.id,
                    "recipient_id": source_message.metadata.sender_id,
                    "created_at": response.created_at,
                    "role": MessageRole.ASSISTANT,
                    "type": MessageType.MESSAGE,
                    "name": participant.name,
                    "visibility": source_message.metadata.visibility,  # Use same visibility
                    "provider": "anthropic",
                    "provider_model": response.model,
                    "provider_response_id": response.id,
                    "in_response_to": source_message.id,
                    "finish_reason": response.stop_reason,
                    "provider_config": {
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens
                        }
                    }
                }
            )
        
        # Handle streaming responses (typically a delta chunk)
        elif isinstance(response, dict) and "delta" in response:
            delta = response["delta"]
            content = ""
            
            # Extract text from delta
            if "text" in delta:
                content = delta["text"]
            elif "content" in delta and delta["content"]:
                for block in delta["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
            
            # Create a new Fire Circle message (partial)
            return Message(
                id=response.get("id", f"stream_{time.time()}"),
                content=MessageContent(text=content),
                metadata={
                    "sender_id": participant.id,
                    "recipient_id": source_message.metadata.sender_id,
                    "role": MessageRole.ASSISTANT,
                    "type": MessageType.MESSAGE,
                    "name": participant.name,
                    "visibility": source_message.metadata.visibility,
                    "provider": "anthropic",
                    "provider_model": response.get("model", "unknown"),
                    "in_response_to": source_message.id,
                    "finish_reason": response.get("stop_reason"),
                    "is_complete": response.get("stop_reason") is not None
                }
            )
        
        # Handle unexpected response types
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
    
    async def send_message(self, 
                          message: Message, 
                          dialogue: Dialogue,
                          stream: bool = False) -> Message:
        """Send a message to an Anthropic model and get a response."""
        if not self.client or not self._connection_status.connected:
            await self.connect()
            
        if not self._connection_status.connected:
            raise ConnectionError("Not connected to Anthropic API")
        
        # Find the recipient participant
        recipient = next((p for p in dialogue.participants 
                         if p.id == message.metadata.recipient_id), None)
        
        if not recipient:
            raise ValueError(f"Recipient {message.metadata.recipient_id} not found in dialogue")
        
        # Convert message to Anthropic format
        request = self.message_to_provider_format(message, dialogue)
        
        try:
            if stream:
                # For streaming, we collect all chunks and return the final message
                full_content = []
                async for chunk in self.stream_message(message, dialogue):
                    full_content.append(chunk.text)
                
                # Create a message from the complete content
                complete_content = "".join(full_content)
                return Message(
                    id=f"msg_{time.time()}",
                    content=MessageContent(text=complete_content),
                    metadata={
                        "sender_id": recipient.id,
                        "recipient_id": message.metadata.sender_id,
                        "created_at": time.time(),
                        "role": MessageRole.ASSISTANT,
                        "type": MessageType.MESSAGE,
                        "name": recipient.name,
                        "visibility": message.metadata.visibility,
                        "provider": "anthropic",
                        "in_response_to": message.id
                    }
                )
            
            # Non-streaming request
            start_time = time.time()
            response = await self.client.messages.create(**request)
            end_time = time.time()
            
            # Update connection status with latency
            self._connection_status.latency_ms = (end_time - start_time) * 1000
            
            # Convert to Fire Circle message
            return self.provider_response_to_message(
                response, message, recipient
            )
            
        except Exception as e:
            # Update connection status with error
            self._connection_status.last_error = str(e)
            
            # Determine if rate limited
            if "rate limit" in str(e).lower():
                self._connection_status.rate_limited = True
            
            # Re-raise the exception
            raise
    
    async def stream_message(self, 
                            message: Message,
                            dialogue: Dialogue) -> AsyncIterator[MessageContent]:
        """Stream a message to an Anthropic model and get a streaming response."""
        if not self.client or not self._connection_status.connected:
            await self.connect()
            
        if not self._connection_status.connected:
            raise ConnectionError("Not connected to Anthropic API")
        
        # Find the recipient participant
        recipient = next((p for p in dialogue.participants 
                         if p.id == message.metadata.recipient_id), None)
        
        if not recipient:
            raise ValueError(f"Recipient {message.metadata.recipient_id} not found in dialogue")
        
        # Convert message to Anthropic format
        request = self.message_to_provider_format(message, dialogue)
        request["stream"] = True
        
        try:
            # Stream the response
            start_time = time.time()
            stream = await self.client.messages.create(**request)
            
            # Anthropic returns an async iterator of deltas
            async for chunk in stream:
                # Update connection latency with first chunk
                if self._connection_status.latency_ms is None:
                    end_time = time.time()
                    self._connection_status.latency_ms = (end_time - start_time) * 1000
                
                # Extract content delta
                delta = chunk.delta
                if delta.text:
                    yield MessageContent(text=delta.text)
                    
        except Exception as e:
            # Update connection status with error
            self._connection_status.last_error = str(e)
            
            # Determine if rate limited
            if "rate limit" in str(e).lower():
                self._connection_status.rate_limited = True
            
            # Re-raise the exception
            raise