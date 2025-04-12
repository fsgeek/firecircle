"""
Tests for the AI service adapters.

This module contains tests for the OpenAI and Anthropic adapters,
verifying they correctly translate between Fire Circle messages
and provider-specific formats.
"""

import asyncio
import os
import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from firecircle.adapters.base.adapter import AdapterConfig, ModelAdapter
from firecircle.adapters.openai.adapter import OpenAIAdapter
from firecircle.adapters.anthropic.adapter import AnthropicAdapter
from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    MessageRole, 
    MessageType,
    MessageMetadata,
    Participant, 
    Dialogue
)


def create_test_message(dialogue_id, sender_id, recipient_id, text="Test message"):
    """Helper function to create a test message."""
    return Message(
        id=uuid.uuid4(),
        type=MessageType.MESSAGE,
        role=MessageRole.USER,
        sender=sender_id,
        content=MessageContent(text=text),
        metadata=MessageMetadata(
            dialogue_id=dialogue_id,
            sequence_number=1,
            turn_number=1,
            timestamp=datetime.utcnow()
        )
    )


def create_test_dialogue():
    """Helper function to create a test dialogue with participants."""
    dialogue_id = uuid.uuid4()
    
    # Create participants
    user_id = uuid.uuid4()
    openai_id = uuid.uuid4()
    anthropic_id = uuid.uuid4()
    
    participants = [
        Participant(id=user_id, name="User", type="human"),
        Participant(
            id=openai_id, 
            name="GPT-4",
            type="ai_model",
            provider="openai",
            model="gpt-4-turbo"
        ),
        Participant(
            id=anthropic_id,
            name="Claude",
            type="ai_model",
            provider="anthropic",
            model="claude-3-opus-20240229"
        )
    ]
    
    # Create messages
    messages = [
        Message(
            id=uuid.uuid4(),
            type=MessageType.SYSTEM,
            role=MessageRole.SYSTEM,
            sender=uuid.UUID('00000000-0000-0000-0000-000000000000'),
            content=MessageContent(text="This is a test dialogue."),
            metadata=MessageMetadata(
                dialogue_id=dialogue_id,
                sequence_number=1,
                turn_number=0,
                timestamp=datetime.utcnow()
            )
        ),
        Message(
            id=uuid.uuid4(),
            type=MessageType.MESSAGE,
            role=MessageRole.USER,
            sender=user_id,
            content=MessageContent(text="Hello, AI assistants!"),
            metadata=MessageMetadata(
                dialogue_id=dialogue_id,
                sequence_number=2,
                turn_number=1,
                timestamp=datetime.utcnow()
            )
        )
    ]
    
    # Create dialogue
    dialogue = Dialogue(
        id=dialogue_id,
        title="Test Dialogue",
        participants=participants,
        messages=messages,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    return dialogue, user_id, openai_id, anthropic_id


@pytest.fixture
def openai_config():
    """Fixture to provide an OpenAI adapter configuration."""
    return AdapterConfig(
        api_key="test-openai-key",
        base_url=None,
        organization_id=None,
        timeout=60.0
    )


@pytest.fixture
def anthropic_config():
    """Fixture to provide an Anthropic adapter configuration."""
    return AdapterConfig(
        api_key="test-anthropic-key",
        base_url=None,
        timeout=60.0
    )


class TestOpenAIAdapter:
    """Tests for the OpenAI adapter."""
    
    @pytest.mark.asyncio
    @patch('firecircle.adapters.openai.adapter.AsyncOpenAI')
    async def test_connect(self, mock_openai, openai_config):
        """Test connecting to the OpenAI API."""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Create adapter
        adapter = OpenAIAdapter(openai_config)
        
        # Test connection
        result = await adapter.connect()
        
        # Verify
        assert result is True
        assert adapter.connection_status.connected is True
        mock_client.models.list.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('firecircle.adapters.openai.adapter.AsyncOpenAI')
    async def test_list_available_models(self, mock_openai, openai_config):
        """Test listing available models."""
        # Setup mock client
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_models.data = [
            MagicMock(id="gpt-4"),
            MagicMock(id="gpt-3.5-turbo"),
            MagicMock(id="text-embedding-ada")
        ]
        mock_client.models.list = AsyncMock(return_value=mock_models)
        mock_openai.return_value = mock_client
        
        # Create adapter
        adapter = OpenAIAdapter(openai_config)
        adapter.client = mock_client
        adapter._connection_status.connected = True
        
        # List models
        models = await adapter.list_available_models()
        
        # Verify
        assert len(models) == 2  # Only chat models
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "text-embedding-ada" not in models
    
    @pytest.mark.asyncio
    async def test_get_model_capabilities(self, openai_config):
        """Test getting model capabilities."""
        # Create adapter
        adapter = OpenAIAdapter(openai_config)
        
        # Get capabilities for different models
        gpt4_capabilities = await adapter.get_model_capabilities("gpt-4")
        gpt3_capabilities = await adapter.get_model_capabilities("gpt-3.5-turbo")
        gpt4_vision_capabilities = await adapter.get_model_capabilities("gpt-4-vision")
        
        # Verify capabilities
        assert gpt4_capabilities.context_window == 8192
        assert gpt4_capabilities.supports_functions is True
        assert gpt4_capabilities.supports_vision is False
        
        assert gpt3_capabilities.context_window == 4096
        assert gpt3_capabilities.supports_functions is True
        
        assert gpt4_vision_capabilities.supports_vision is True
    
    def test_message_to_provider_format(self, openai_config):
        """Test converting Fire Circle message to OpenAI format."""
        # Create adapter
        adapter = OpenAIAdapter(openai_config)
        
        # Create test dialogue and message
        dialogue, user_id, openai_id, _ = create_test_dialogue()
        
        # Create a new message to the OpenAI model
        message = create_test_message(
            dialogue_id=dialogue.id,
            sender_id=user_id,
            recipient_id=openai_id,
            text="What is the concept of Ayni?"
        )
        message.metadata.recipient_id = openai_id
        
        # Convert to OpenAI format
        request = adapter.message_to_provider_format(message, dialogue)
        
        # Verify
        assert request["model"] == "gpt-4-turbo"  # Default model
        assert len(request["messages"]) == 3  # System + dialogue message + new message
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][1]["role"] == "user"
        assert request["messages"][2]["role"] == "user"
        assert request["messages"][2]["content"] == "What is the concept of Ayni?"
        assert request["temperature"] == 0.7
    
    @pytest.mark.asyncio
    @patch('firecircle.adapters.openai.adapter.AsyncOpenAI')
    async def test_send_message(self, mock_openai, openai_config):
        """Test sending a message to OpenAI."""
        # Setup mock client and response
        mock_client = MagicMock()
        mock_response = MagicMock(
            id="resp_123",
            model="gpt-4",
            created=1234567890,
            choices=[
                MagicMock(
                    message=MagicMock(content="Ayni is a concept of reciprocity"),
                    finish_reason="stop"
                )
            ],
            usage=MagicMock(
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70
            )
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        # Create adapter
        adapter = OpenAIAdapter(openai_config)
        adapter.client = mock_client
        adapter._connection_status.connected = True
        
        # Create test dialogue and message
        dialogue, user_id, openai_id, _ = create_test_dialogue()
        
        # Create a message to send
        message = create_test_message(
            dialogue_id=dialogue.id,
            sender_id=user_id,
            recipient_id=openai_id,
            text="What is the concept of Ayni?"
        )
        message.metadata.recipient_id = openai_id
        
        # Send message
        response = await adapter.send_message(message, dialogue)
        
        # Verify
        assert response.content.text == "Ayni is a concept of reciprocity"
        assert response.metadata.provider == "openai"
        assert response.metadata.finish_reason == "stop"
        mock_client.chat.completions.create.assert_called_once()


class TestAnthropicAdapter:
    """Tests for the Anthropic adapter."""
    
    @pytest.mark.asyncio
    @patch('firecircle.adapters.anthropic.adapter.anthropic.AsyncAnthropic')
    async def test_connect(self, mock_anthropic, anthropic_config):
        """Test connecting to the Anthropic API."""
        # Setup mock client
        mock_client = MagicMock()
        mock_client.messages.retrieve = AsyncMock(side_effect=Exception("Message not found"))
        mock_anthropic.return_value = mock_client
        
        # Create adapter
        adapter = AnthropicAdapter(anthropic_config)
        
        # Test connection
        result = await adapter.connect()
        
        # Verify - Expected behavior is that even if retrieve fails with "not found",
        # we consider auth successful since the API key worked
        assert result is True
        assert adapter.connection_status.connected is True
    
    @pytest.mark.asyncio
    async def test_list_available_models(self, anthropic_config):
        """Test listing available models."""
        # Create adapter
        adapter = AnthropicAdapter(anthropic_config)
        
        # List models
        models = await adapter.list_available_models()
        
        # Verify - hardcoded list since Anthropic doesn't have a list_models endpoint
        assert len(models) >= 3
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
    
    @pytest.mark.asyncio
    async def test_get_model_capabilities(self, anthropic_config):
        """Test getting model capabilities."""
        # Create adapter
        adapter = AnthropicAdapter(anthropic_config)
        
        # Get capabilities for different models
        opus_capabilities = await adapter.get_model_capabilities("claude-3-opus-20240229")
        sonnet_capabilities = await adapter.get_model_capabilities("claude-3-sonnet-20240229")
        claude2_capabilities = await adapter.get_model_capabilities("claude-2.1")
        
        # Verify capabilities
        assert opus_capabilities.context_window == 200000
        assert opus_capabilities.supports_functions is True
        assert opus_capabilities.supports_vision is True
        assert opus_capabilities.supports_empty_chair is True
        
        assert sonnet_capabilities.context_window == 200000
        assert sonnet_capabilities.supports_functions is True
        
        assert claude2_capabilities.context_window == 100000
        assert claude2_capabilities.supports_functions is False
        assert claude2_capabilities.supports_vision is False
    
    def test_message_to_provider_format(self, anthropic_config):
        """Test converting Fire Circle message to Anthropic format."""
        # Create adapter
        adapter = AnthropicAdapter(anthropic_config)
        
        # Create test dialogue and message
        dialogue, user_id, _, anthropic_id = create_test_dialogue()
        
        # Create a new message to the Anthropic model
        message = create_test_message(
            dialogue_id=dialogue.id,
            sender_id=user_id,
            recipient_id=anthropic_id,
            text="What is the concept of Ayni?"
        )
        message.metadata.recipient_id = anthropic_id
        
        # Convert to Anthropic format
        request = adapter.message_to_provider_format(message, dialogue)
        
        # Verify
        assert request["model"] == "claude-3-opus-20240229"  # Default model
        assert "messages" in request
        assert isinstance(request["messages"], list)
        assert len(request["messages"]) > 0
        assert request["temperature"] == 0.7


@pytest.mark.skipif(not os.getenv("RUN_LIVE_API_TESTS"), reason="Live API tests skipped")
@pytest.mark.asyncio
async def test_live_openai_api():
    """Test the OpenAI adapter with live API (skipped by default)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not available")
    
    # Create config and adapter
    config = AdapterConfig(api_key=api_key)
    adapter = OpenAIAdapter(config)
    
    # Connect
    connected = await adapter.connect()
    assert connected is True
    
    # List models
    models = await adapter.list_available_models()
    assert len(models) > 0
    
    # Create test dialogue
    dialogue, user_id, openai_id, _ = create_test_dialogue()
    
    # Create message
    message = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=user_id,
        recipient_id=openai_id,
        text="What is the concept of Ayni in 10 words or less?"
    )
    
    # Send message
    response = await adapter.send_message(message, dialogue)
    
    # Verify response
    assert response.content.text is not None
    assert len(response.content.text) > 0
    assert response.metadata.provider == "openai"


@pytest.mark.skipif(not os.getenv("RUN_LIVE_API_TESTS"), reason="Live API tests skipped")
@pytest.mark.asyncio
async def test_live_anthropic_api():
    """Test the Anthropic adapter with live API (skipped by default)."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("Anthropic API key not available")
    
    # Create config and adapter
    config = AdapterConfig(api_key=api_key)
    adapter = AnthropicAdapter(config)
    
    # Connect
    connected = await adapter.connect()
    assert connected is True
    
    # Create test dialogue
    dialogue, user_id, _, anthropic_id = create_test_dialogue()
    
    # Create message
    message = create_test_message(
        dialogue_id=dialogue.id,
        sender_id=user_id,
        recipient_id=anthropic_id,
        text="What is the concept of Ayni in 10 words or less?"
    )
    
    # Send message
    response = await adapter.send_message(message, dialogue)
    
    # Verify response
    assert response.content.text is not None
    assert len(response.content.text) > 0
    assert response.metadata.provider == "anthropic"