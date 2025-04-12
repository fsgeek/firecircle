"""
OpenAI adapter implementation for Fire Circle.

This module provides an adapter for connecting the Fire Circle protocol with OpenAI's API,
translating between the standardized Fire Circle message format and OpenAI's API requirements.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta

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


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.client = None
        self._model_capabilities_cache: Dict[str, ModelCapabilities] = {}
        
    async def connect(self) -> bool:
        """Establish connection to OpenAI API."""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization_id,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            
            # Test connection by listing models
            start_time = time.time()
            await self.client.models.list()
            end_time = time.time()
            
            self._connection_status = ConnectionStatus(
                connected=True,
                latency_ms=(end_time - start_time) * 1000
            )
            return True
        except Exception as e:
            self._connection_status = ConnectionStatus(
                connected=False,
                last_error=str(e)
            )
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to OpenAI API."""
        # OpenAI client doesn't require explicit disconnection
        return True
    
    async def list_available_models(self) -> List[str]:
        """List available models from OpenAI."""
        if not self.client or not self._connection_status.connected:
            await self.connect()
            
        if not self._connection_status.connected:
            return []
            
        try:
            response = await self.client.models.list()
            # Filter for chat models only
            chat_models = [model.id for model in response.data 
                          if model.id.startswith("gpt-") or "instruct" in model.id]
            return chat_models
        except Exception as e:
            self._connection_status.last_error = str(e)
            return []
    
    async def get_model_capabilities(self, model_id: str) -> ModelCapabilities:
        """Get capabilities of a specific OpenAI model."""
        # Return from cache if available
        if model_id in self._model_capabilities_cache:
            return self._model_capabilities_cache[model_id]
        
        # Default capabilities by model family
        if "gpt-4" in model_id:
            if "32k" in model_id:
                capabilities = ModelCapabilities(
                    max_tokens=32768,
                    context_window=32768,
                    supports_functions=True,
                    supports_vision="vision" in model_id,
                    supports_streaming=True,
                    supports_system_messages=True
                )
            else:
                capabilities = ModelCapabilities(
                    max_tokens=8192,
                    context_window=8192,
                    supports_functions=True,
                    supports_vision="vision" in model_id,
                    supports_streaming=True,
                    supports_system_messages=True
                )
        elif "gpt-3.5-turbo" in model_id:
            if "16k" in model_id:
                capabilities = ModelCapabilities(
                    max_tokens=16384,
                    context_window=16384,
                    supports_functions=True,
                    supports_vision="vision" in model_id,
                    supports_streaming=True,
                    supports_system_messages=True
                )
            else:
                capabilities = ModelCapabilities(
                    max_tokens=4096,
                    context_window=4096,
                    supports_functions=True,
                    supports_vision="vision" in model_id,
                    supports_streaming=True,
                    supports_system_messages=True
                )
        else:
            # Default for unknown models
            capabilities = ModelCapabilities(
                max_tokens=4096,
                context_window=4096,
                supports_functions=False,
                supports_vision=False,
                supports_streaming=True,
                supports_system_messages=True
            )
            
        # Cache the capabilities
        self._model_capabilities_cache[model_id] = capabilities
        return capabilities
    
    def _convert_message_role(self, role: MessageRole) -> str:
        """Convert Fire Circle message role to OpenAI role."""
        role_mapping = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.SYSTEM: "system",
            MessageRole.FUNCTION: "function"
        }
        
        return role_mapping.get(role, "user")
    
    def _extract_model_id(self, dialogue: Dialogue, participant: Participant) -> str:
        """Extract the appropriate model ID for a given participant."""
        # Default to the most capable model if not specified
        default_model = "gpt-4-turbo"
        
        # Check participant configuration for model ID
        if participant and participant.provider_config:
            model_id = participant.provider_config.get("model_id")
            if model_id:
                return model_id
        
        # Check dialogue configuration
        if dialogue and dialogue.metadata and dialogue.metadata.provider_config:
            model_id = dialogue.metadata.provider_config.get("openai_model_id")
            if model_id:
                return model_id
                
        return default_model
        
    def message_to_provider_format(self, 
                                  message: Message, 
                                  dialogue: Dialogue) -> Dict[str, Any]:
        """Convert a Fire Circle message to OpenAI's format."""
        # Extract the model to use
        recipient = next((p for p in dialogue.participants 
                         if p.id == message.metadata.recipient_id), None)
        model = self._extract_model_id(dialogue, recipient)
        
        # Build the message history
        messages = []
        
        # Add system message if available in dialogue
        system_messages = [m for m in dialogue.messages 
                          if m.metadata.role == MessageRole.SYSTEM]
        if system_messages:
            # Get the most recent system message
            system_message = system_messages[-1]
            messages.append({
                "role": "system",
                "content": system_message.content.text
            })
        
        # Add dialogue history (excluding system messages already added)
        for hist_msg in dialogue.messages:
            if hist_msg.metadata.role == MessageRole.SYSTEM:
                continue
                
            # Skip messages that aren't visible to the recipient
            if (recipient and hist_msg.metadata.visibility and 
                recipient.id not in hist_msg.metadata.visibility):
                continue
                
            # Convert to OpenAI format
            openai_message = {
                "role": self._convert_message_role(hist_msg.metadata.role),
                "content": hist_msg.content.text
            }
            
            if hist_msg.metadata.name:
                openai_message["name"] = hist_msg.metadata.name
                
            messages.append(openai_message)
        
        # Add the current message
        current_message = {
            "role": self._convert_message_role(message.metadata.role),
            "content": message.content.text
        }
        
        if message.metadata.name:
            current_message["name"] = message.metadata.name
            
        messages.append(current_message)
        
        # Build the complete request
        request = {
            "model": model,
            "messages": messages,
            "temperature": message.metadata.provider_config.get("temperature", 0.7) if message.metadata.provider_config else 0.7,
            "max_tokens": message.metadata.provider_config.get("max_tokens") if message.metadata.provider_config else None,
        }
        
        return request
    
    def provider_response_to_message(self, 
                                    response: Union[ChatCompletion, ChatCompletionChunk], 
                                    source_message: Message,
                                    participant: Participant) -> Message:
        """Convert an OpenAI response to a Fire Circle message."""
        # Handle full completion responses
        if isinstance(response, ChatCompletion):
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Create a new Fire Circle message
            return Message(
                id=response.id,
                content=MessageContent(text=content),
                metadata={
                    "sender_id": participant.id,
                    "recipient_id": source_message.metadata.sender_id,
                    "created_at": response.created,
                    "role": MessageRole.ASSISTANT,
                    "type": MessageType.MESSAGE,
                    "name": participant.name,
                    "visibility": source_message.metadata.visibility,  # Use same visibility
                    "provider": "openai",
                    "provider_model": response.model,
                    "provider_response_id": response.id,
                    "in_response_to": source_message.id,
                    "finish_reason": choice.finish_reason,
                    "provider_config": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            )
        
        # Handle streaming responses (typically the final chunk)
        elif isinstance(response, ChatCompletionChunk):
            choice = response.choices[0]
            # For streaming, we typically collect content in the calling code,
            # but this handles individual chunks
            content = choice.delta.content or ""
            
            # Create a new Fire Circle message (partial)
            return Message(
                id=response.id,
                content=MessageContent(text=content),
                metadata={
                    "sender_id": participant.id,
                    "recipient_id": source_message.metadata.sender_id,
                    "role": MessageRole.ASSISTANT,
                    "type": MessageType.MESSAGE,
                    "name": participant.name,
                    "visibility": source_message.metadata.visibility,
                    "provider": "openai",
                    "provider_model": response.model,
                    "provider_response_id": response.id,
                    "in_response_to": source_message.id,
                    "finish_reason": choice.finish_reason,
                    "is_complete": choice.finish_reason is not None
                }
            )
        
        # Handle unexpected response types
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
    
    async def send_message(self, 
                          message: Message, 
                          dialogue: Dialogue,
                          stream: bool = False) -> Message:
        """Send a message to an OpenAI model and get a response."""
        if not self.client or not self._connection_status.connected:
            await self.connect()
            
        if not self._connection_status.connected:
            raise ConnectionError("Not connected to OpenAI API")
        
        # Find the recipient participant
        recipient = next((p for p in dialogue.participants 
                         if p.id == message.metadata.recipient_id), None)
        
        if not recipient:
            raise ValueError(f"Recipient {message.metadata.recipient_id} not found in dialogue")
        
        # Convert message to OpenAI format
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
                        "provider": "openai",
                        "in_response_to": message.id
                    }
                )
            
            # Non-streaming request
            start_time = time.time()
            response = await self.client.chat.completions.create(**request)
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
        """Stream a message to an OpenAI model and get a streaming response."""
        if not self.client or not self._connection_status.connected:
            await self.connect()
            
        if not self._connection_status.connected:
            raise ConnectionError("Not connected to OpenAI API")
        
        # Find the recipient participant
        recipient = next((p for p in dialogue.participants 
                         if p.id == message.metadata.recipient_id), None)
        
        if not recipient:
            raise ValueError(f"Recipient {message.metadata.recipient_id} not found in dialogue")
        
        # Convert message to OpenAI format
        request = self.message_to_provider_format(message, dialogue)
        request["stream"] = True
        
        try:
            # Stream the response
            start_time = time.time()
            stream = await self.client.chat.completions.create(**request)
            
            # Yield content chunks as they arrive
            async for chunk in stream:
                # Update connection latency with first chunk
                if self._connection_status.latency_ms is None:
                    end_time = time.time()
                    self._connection_status.latency_ms = (end_time - start_time) * 1000
                
                # Extract content delta if available
                if chunk.choices and chunk.choices[0].delta.content:
                    yield MessageContent(text=chunk.choices[0].delta.content)
                    
        except Exception as e:
            # Update connection status with error
            self._connection_status.last_error = str(e)
            
            # Determine if rate limited
            if "rate limit" in str(e).lower():
                self._connection_status.rate_limited = True
            
            # Re-raise the exception
            raise