"""
CLI utility for Fire Circle adapters.

This module provides a command-line interface for creating, testing, and managing
adapters for various AI model providers.
"""

import asyncio
import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional

from firecircle.adapters.base.adapter import AdapterConfig, AdapterFactory, AdapterRegistry
from firecircle.protocol.message import (
    Message, 
    MessageContent, 
    Participant, 
    Dialogue, 
    MessageType, 
    MessageRole,
    create_system_message
)


async def setup_adapter(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    organization_id: Optional[str] = None
) -> None:
    """
    Set up and register an adapter for a provider.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        api_key: The API key for the provider
        base_url: Optional custom base URL for the API
        organization_id: Optional organization ID (for OpenAI)
    """
    config = AdapterConfig(
        api_key=api_key,
        base_url=base_url,
        organization_id=organization_id
    )
    
    try:
        adapter = await AdapterFactory.create_adapter(provider, config)
        print(f"Successfully created and registered adapter for {provider}")
        print(f"Connection status: {adapter.connection_status.connected}")
        
        if adapter.connection_status.connected:
            models = await adapter.list_available_models()
            print(f"Available models ({len(models)}):")
            for model in models[:10]:  # Show first 10 models
                print(f"  - {model}")
            
            if len(models) > 10:
                print(f"  ... and {len(models) - 10} more")
        else:
            print(f"Connection error: {adapter.connection_status.last_error}")
    
    except Exception as e:
        print(f"Error creating adapter for {provider}: {e}")


async def list_adapters() -> None:
    """List all registered adapters and their status."""
    adapters = AdapterRegistry.list_adapters()
    
    if not adapters:
        print("No adapters registered.")
        return
    
    print(f"Registered adapters ({len(adapters)}):")
    for name in adapters:
        adapter = AdapterRegistry.get(name)
        if adapter:
            status = "Connected" if adapter.connection_status.connected else "Disconnected"
            print(f"  - {name}: {status}")
            
            if adapter.connection_status.last_error:
                print(f"    Last error: {adapter.connection_status.last_error}")


async def test_message(
    provider: str,
    system_prompt: Optional[str] = None,
    user_message: Optional[str] = None,
    model_id: Optional[str] = None,
    stream: bool = False
) -> None:
    """
    Test sending a message with a registered adapter.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        system_prompt: Optional system prompt
        user_message: The user message to send
        model_id: Optional specific model to use
        stream: Whether to stream the response
    """
    adapter = AdapterRegistry.get(provider)
    
    if not adapter:
        print(f"No adapter registered for {provider}. Use --setup first.")
        return
    
    if not adapter.connection_status.connected:
        print(f"Adapter for {provider} is not connected. Attempting to reconnect...")
        connected = await adapter.connect()
        
        if not connected:
            print(f"Failed to connect: {adapter.connection_status.last_error}")
            return
    
    # Set defaults
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant in a Fire Circle dialogue."
    
    if not user_message:
        user_message = "Tell me about the concept of reciprocity or 'Ayni' in 3-4 sentences."
    
    # Create test participants and dialogue
    user_id = "user-123"
    ai_id = f"{provider}-assistant"
    
    user = Participant(
        id=user_id,
        name="User",
        provider=None,
        provider_config={}
    )
    
    ai_assistant = Participant(
        id=ai_id,
        name=f"{provider.capitalize()} Assistant",
        provider=provider,
        provider_config={"model_id": model_id} if model_id else {}
    )
    
    # Create system message
    system_msg = create_system_message(
        content=system_prompt,
        sender_id="system",
        recipient_id=ai_id
    )
    
    # Create user message
    user_msg = Message(
        id="user-msg-1",
        content=MessageContent(text=user_message),
        metadata={
            "sender_id": user_id,
            "recipient_id": ai_id,
            "role": MessageRole.USER,
            "type": MessageType.MESSAGE,
            "name": user.name
        }
    )
    
    # Create dialogue
    dialogue = Dialogue(
        id="dialogue-1",
        name="Test Dialogue",
        participants=[user, ai_assistant],
        messages=[system_msg, user_msg]
    )
    
    # Send message
    try:
        if stream:
            print(f"\nSending message to {provider} (streaming)...")
            print(f"User: {user_message}")
            print(f"\n{ai_assistant.name}: ", end="", flush=True)
            
            async for content_chunk in adapter.stream_message(user_msg, dialogue):
                print(content_chunk.text, end="", flush=True)
            print("\n")
        else:
            print(f"\nSending message to {provider}...")
            print(f"User: {user_message}")
            
            response = await adapter.send_message(user_msg, dialogue)
            
            print(f"\n{ai_assistant.name}: {response.content.text}")
            
            # Print token usage if available
            if "provider_config" in response.metadata and response.metadata.provider_config:
                if "usage" in response.metadata.provider_config:
                    usage = response.metadata.provider_config["usage"]
                    print("\nToken usage:")
                    for key, value in usage.items():
                        print(f"  {key}: {value}")
        
        print(f"\nLatency: {adapter.connection_status.latency_ms:.2f}ms")
        
    except Exception as e:
        print(f"Error sending message: {e}")


async def main() -> None:
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(description="Fire Circle Adapter CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up an adapter")
    setup_parser.add_argument(
        "--provider", 
        type=str, 
        required=True,
        choices=["openai", "anthropic"],
        help="The provider to set up"
    )
    setup_parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the provider"
    )
    setup_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for the API (optional)"
    )
    setup_parser.add_argument(
        "--org-id",
        type=str,
        help="Organization ID (for OpenAI)"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List registered adapters")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test an adapter")
    test_parser.add_argument(
        "--provider", 
        type=str, 
        required=True,
        help="The provider to test"
    )
    test_parser.add_argument(
        "--system",
        type=str,
        help="System prompt to use"
    )
    test_parser.add_argument(
        "--message",
        type=str,
        help="User message to send"
    )
    test_parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use"
    )
    test_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response"
    )
    
    args = parser.parse_args()
    
    if args.command == "setup":
        # Get API key from args or environment
        api_key = args.api_key
        if not api_key:
            env_var = f"{args.provider.upper()}_API_KEY"
            api_key = os.environ.get(env_var)
            
            if not api_key:
                print(f"Error: No API key provided. Please provide --api-key or set {env_var} environment variable.")
                sys.exit(1)
        
        await setup_adapter(
            provider=args.provider,
            api_key=api_key,
            base_url=args.base_url,
            organization_id=args.org_id
        )
    
    elif args.command == "list":
        await list_adapters()
    
    elif args.command == "test":
        await test_message(
            provider=args.provider,
            system_prompt=args.system,
            user_message=args.message,
            model_id=args.model,
            stream=args.stream
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())