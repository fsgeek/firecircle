"""
Test script for Fire Circle adapters.

This script provides functionality to test the OpenAI and Anthropic adapters
by sending test messages and verifying responses.
"""

import asyncio
import argparse
import os
import sys
import uuid
from typing import Dict, Any, Optional

from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    Participant, 
    Dialogue, 
    MessageType, 
    MessageRole,
    MessageMetadata,
    create_system_message
)
from firecircle.adapters.base.adapter import AdapterConfig, AdapterFactory
from firecircle.adapters.openai.adapter import OpenAIAdapter
from firecircle.adapters.anthropic.adapter import AnthropicAdapter


async def test_adapter(
    provider: str, 
    api_key: str,
    model_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_message: Optional[str] = None
) -> None:
    """
    Test an AI model adapter with a sample dialogue.
    
    Args:
        provider: The provider name ("openai" or "anthropic")
        api_key: The API key for the provider
        model_id: Optional specific model ID to test
        system_prompt: Optional system prompt to use
        user_message: Optional user message to send
    """
    # Set defaults
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant in a Fire Circle dialogue."
    
    if not user_message:
        user_message = "Hello! Can you tell me about the concept of reciprocity or 'Ayni'?"
    
    if not model_id:
        model_id = "gpt-4" if provider == "openai" else "claude-3-opus-20240229"
    
    # Create adapter config
    config = AdapterConfig(api_key=api_key)
    
    # Create adapter
    adapter = await AdapterFactory.create_adapter(provider, config)
    
    # Test connection
    connected = adapter.connection_status.connected
    print(f"Connected to {provider}: {connected}")
    
    if not connected:
        print(f"Connection error: {adapter.connection_status.last_error}")
        return
    
    # List available models
    models = await adapter.list_available_models()
    print(f"Available models from {provider}:")
    for model in models:
        print(f"  - {model}")
    
    # Create test participants
    user_id = str(uuid.uuid4())
    ai_id = str(uuid.uuid4())
    
    user = Participant(
        id=user_id,
        name="User",
        provider=None,
        provider_config={}
    )
    
    ai_assistant = Participant(
        id=ai_id,
        name="AI Assistant",
        provider=provider,
        provider_config={
            "model_id": model_id
        }
    )
    
    # Create system message
    system_msg = create_system_message(
        content=system_prompt,
        sender_id="system",
        recipient_id=ai_id
    )
    
    # Create user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        content=MessageContent(text=user_message),
        metadata=MessageMetadata(
            sender_id=user_id,
            recipient_id=ai_id,
            role=MessageRole.USER,
            type=MessageType.MESSAGE,
            name=user.name
        )
    )
    
    # Create dialogue
    dialogue = Dialogue(
        id=str(uuid.uuid4()),
        name="Test Dialogue",
        participants=[user, ai_assistant],
        messages=[system_msg, user_msg]
    )
    
    # Get model capabilities
    capabilities = await adapter.get_model_capabilities(model_id)
    print(f"\nModel capabilities for {model_id}:")
    print(f"  Max tokens: {capabilities.max_tokens}")
    print(f"  Context window: {capabilities.context_window}")
    print(f"  Supports functions: {capabilities.supports_functions}")
    print(f"  Supports vision: {capabilities.supports_vision}")
    print(f"  Supports streaming: {capabilities.supports_streaming}")
    print(f"  Supports system messages: {capabilities.supports_system_messages}")
    print(f"  Supports empty chair: {capabilities.supports_empty_chair}")
    
    # Test non-streaming request
    print("\nTesting non-streaming request...")
    try:
        response = await adapter.send_message(user_msg, dialogue, stream=False)
        print(f"\nResponse from {provider} ({response.metadata.provider_model}):")
        print(f"ID: {response.id}")
        print(f"Content: {response.content.text[:500]}...")
        if "provider_config" in response.metadata and response.metadata.provider_config:
            if "usage" in response.metadata.provider_config:
                usage = response.metadata.provider_config["usage"]
                if "input_tokens" in usage and "output_tokens" in usage:
                    print(f"Tokens: {usage['input_tokens']} input, {usage['output_tokens']} output")
                elif "prompt_tokens" in usage and "completion_tokens" in usage:
                    print(f"Tokens: {usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion")
            
        print(f"Latency: {adapter.connection_status.latency_ms:.2f}ms")
    except Exception as e:
        print(f"Error testing non-streaming request: {e}")
    
    # Test streaming request
    if capabilities.supports_streaming:
        print("\nTesting streaming request...")
        try:
            print("\nStreaming response:")
            async for content_chunk in adapter.stream_message(user_msg, dialogue):
                print(content_chunk.text, end="", flush=True)
            print("\n")
            print(f"Latency: {adapter.connection_status.latency_ms:.2f}ms")
        except Exception as e:
            print(f"Error testing streaming request: {e}")
    
    # Disconnect
    await adapter.disconnect()
    print("\nTest completed and disconnected.")


async def main() -> None:
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test Fire Circle adapters")
    parser.add_argument(
        "--provider", 
        type=str, 
        choices=["openai", "anthropic", "both"], 
        default="both",
        help="The provider to test"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model ID to test"
    )
    parser.add_argument(
        "--system",
        type=str,
        help="System prompt to use"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="User message to send"
    )
    
    args = parser.parse_args()
    
    # Get API keys from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if args.provider in ["openai", "both"]:
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            if args.provider == "openai":
                sys.exit(1)
        else:
            model_id = args.model if args.model else "gpt-4"
            await test_adapter(
                provider="openai", 
                api_key=openai_api_key,
                model_id=model_id,
                system_prompt=args.system,
                user_message=args.message
            )
    
    if args.provider in ["anthropic", "both"]:
        if not anthropic_api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set.")
            if args.provider == "anthropic":
                sys.exit(1)
        else:
            model_id = args.model if args.model else "claude-3-opus-20240229"
            await test_adapter(
                provider="anthropic", 
                api_key=anthropic_api_key,
                model_id=model_id,
                system_prompt=args.system,
                user_message=args.message
            )


if __name__ == "__main__":
    asyncio.run(main())