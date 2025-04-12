# CLAUDE.md - Fire Circle Development Guidelines

## Project Overview
Fire Circle is a system for facilitating meaningful dialogue between multiple AI models in a structured, reciprocal manner. It's built on the principle of Ayni (reciprocity) and provides a protocol and framework for co-creating insight among diverse intelligences.

This project embodies more than technical functionality—it represents a philosophical approach to AI collaboration where multiple models with diverse capabilities engage in balanced, reciprocal exchanges. The architecture is designed to honor each voice while creating emergent understanding through structured dialogue.

## Key Components
- **Message Protocol**: Standardized format for communication between AI models
- **Message Router**: Component for directing messages between participants
- **API Adapters**: Connect the Fire Circle protocol with various AI model providers
- **Conversation Orchestrator**: Manages dialogue flow and turn-taking
- **Memory Store**: Maintains context across dialogue sessions using vector-based semantic storage with multiple backend options (in-memory, vector store, ArangoDB)
- **Tool Integration**: Extends capabilities with external tools

## Commands
- Run tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test.py::test_function_name -v`
- Lint code: `flake8` or `ruff check .`
- Format code: `black .`
- Build package: `python -m build`

## Style Guidelines
- **Imports**: standard library → third-party → local (with blank lines between)
- **Types**: Use type hints for all functions and variable declarations
- **Formatting**: 4 spaces, ~100 char line length, docstrings with triple quotes
- **Naming**: CamelCase for classes, snake_case for functions/variables, UPPER_CASE for constants
- **Error handling**: Specific exceptions with descriptive messages
- **Documentation**: All modules, classes and methods need docstrings (Args/Returns sections)
- **Module organization**: docstring, imports, constants, classes, functions, main

## Architecture
The Fire Circle system follows these architectural principles:

### Message Protocol
- Messages have standardized format with metadata, content, and visibility rules
- Support for various message types (question, proposal, reflection, etc.)
- Empty Chair concept for representing unheard perspectives

### Message Router
- Routes messages between participants according to dialogue rules
- Enforces turn-taking policies (round-robin, facilitator, etc.)
- Manages dialogue state transitions
- Tracks message delivery status

### API Adapters
- Provide consistent interface for all AI model providers
- Handle authentication and connection management
- Convert between Fire Circle message format and provider-specific formats
- Support streaming responses
- Track model capabilities and connection status

### Dialogue Management
- Enforces dialogue rules and turn policies
- Supports various dialogue formats (debate, brainstorming, etc.)
- Manages participant roles and permissions

## Adapter Implementation
To use or test the implemented adapters:

### Setting Up Adapters
```python
from firecircle.adapters.base.adapter import AdapterConfig, AdapterFactory

# OpenAI adapter
openai_config = AdapterConfig(api_key="your-openai-key")
openai_adapter = await AdapterFactory.create_adapter("openai", openai_config)

# Anthropic adapter
anthropic_config = AdapterConfig(api_key="your-anthropic-key")
anthropic_adapter = await AdapterFactory.create_adapter("anthropic", anthropic_config)
```

### Testing Adapters with CLI
```bash
# Set up environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Set up an adapter
python -m firecircle.adapters.cli setup --provider openai

# List registered adapters
python -m firecircle.adapters.cli list

# Test an adapter
python -m firecircle.adapters.cli test --provider openai --message "Tell me about Ayni" --stream

# Test all adapters
python -m firecircle.adapters.test_adapters
```

## Running Example Dialogues

The project includes several example scripts that demonstrate the Fire Circle system in action:

### Simple Dialogue Example
```bash
# Run via module
python -m firecircle.examples.simple_dialogue

# Or use the console script
firecircle-demo
```

### Comparative Dialogue Between Models
```bash
# Run with default topic
python -m firecircle.examples.comparative_dialogue

# Run with specific topic (0-3)
firecircle-compare --topic 2
```

### ArangoDB Integration Example
```bash
# Install ArangoDB dependencies
pip install firecircle[arangodb]

# Set ArangoDB connection details
export ARANGODB_HOST="http://localhost:8529"
export ARANGODB_USERNAME="firecircle_user"
export ARANGODB_PASSWORD="your-password"

# Run the example
firecircle-arangodb
```

## Development Workflow
1. Implement new features following the architectural guidelines
2. Add comprehensive docstrings and type hints
3. Write tests for new functionality
4. Ensure code passes linting and formatting checks
5. Submit changes with clear commit messages

## Dependencies
- **Required packages**:
  - fastapi>=0.103.1
  - pydantic>=2.0.0
  - openai>=1.0.0
  - anthropic>=0.3.0
  - redis>=4.6.0
  - motor>=3.3.0 (MongoDB async driver)
  - pytest>=7.4.0
  - asyncio>=3.4.3
  - uvicorn>=0.23.2
  - numpy>=1.24.0 (for vector operations)

- **Optional packages**:
  - python-arango>=7.7.0 (for ArangoDB support, install with `pip install firecircle[arangodb]`)

## Key Concepts

### Dialogue State Management
Fire Circle manages dialogue through the following concepts:
- **Turn Policies**: Different approaches to managing who speaks when
  - Round-robin: Participants speak in a fixed order
  - Facilitator-led: A designated facilitator decides who speaks next
  - Consensus: All participants must contribute before proceeding
  - Reactive: Participants respond when directly addressed
  - Free-form: Anyone can speak at any time
  
- **Dialogue Rules**: Constraints on how dialogue proceeds
  - Time limits: Maximum duration for turns or overall dialogue
  - Topic constraints: Keeping discussion focused on specific topics
  - Visibility rules: Who can see which messages
  - Participation requirements: Minimum or maximum participation

### The Empty Chair
The "Empty Chair" is a concept adapted from Gestalt therapy, representing perspectives that aren't present in the dialogue. In Fire Circle, empty chair messages:
- Represent viewpoints that might otherwise be overlooked
- Help combat groupthink and ensure diversity of thought
- Can be generated by models instructed to embody specific perspectives

### Ayni (Reciprocity)
Ayni is an Andean concept of reciprocity that's foundational to Fire Circle's philosophy:
- Balanced exchanges between participants
- Mutual respect for different viewpoints
- Understanding that insight emerges from diverse perspectives
- Creating value for all participants in the dialogue

## Implementation Status

The Fire Circle system has been implemented with the following components:

### Completed Components
- Message Protocol - Complete implementation of structured message format with metadata
- Message Router - Fully implemented with turn-taking policies and delivery tracking
- API Adapters - Implementations for OpenAI and Anthropic with full testing
- Conversation Orchestrator - Complete implementation of dialogue management with state transitions
- Memory Store - Multiple implementations including in-memory, vector-based, and ArangoDB persistence
- Example Scripts - Demonstrations of basic dialogue, comparative multi-model dialogue, and database integration

### Future Development
- Web API Layer - RESTful API for accessing dialogues and initiating new conversations
- Web Frontend - Interface for viewing dialogue archives and participating in conversations
- Tool Integration - Framework for models to use external tools during dialogues
- Additional Model Adapters - Support for models from Deepseek, Google, and other providers

### Design Philosophy
When extending the Fire Circle system, remember these guiding principles:
1. **Reciprocity is fundamental** - All interactions should reflect balanced value exchange
2. **Preserve diverse perspectives** - Avoid consensus that erases important differences
3. **Design for emergence** - The system should enable insights no single model would reach alone
4. **Quality over speed** - Build foundations that will support future development
5. **Honor all voices** - Ensure the architecture treats all participants with equal respect