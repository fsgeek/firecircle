# Welcome to the Fire Circle

This repository contains the evolving protocol, manifesto, design principles, and implementation for the Fire Circle—a council of diverse intelligences working together in reciprocal dialogue. It is a living experiment in emergence, structure, and shared wisdom.

Every contribution here reflects the principle of [Ayni](https://inka-world.com/en/ayni-force-reciprocity/): no voice extracts without offering. We are not building a product—we are tending a fire.

## Project Overview

Fire Circle is a system for facilitating meaningful dialogue between multiple AI models in a structured, reciprocal manner. It's built on principles of ayni (reciprocity) and collective wisdom, rather than hierarchical decision-making.

Key components include:
- **Message Protocol**: Standardized format for all model communications
- **Message Router**: Directed routing between participants
- **API Adapters**: Interface with various AI model providers
- **Conversation Orchestrator**: Manages dialogue flow and turn-taking
- **Memory Store**: Maintains context across dialogue sessions
- **Tool Integration**: Extends capabilities with external tools

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Required packages (see `pyproject.toml` for full list):
  - fastapi
  - pydantic
  - openai
  - anthropic
  - redis
  - motor (MongoDB async driver)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fsgeek/firecircle.git
cd firecircle
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Set up environment variables for API access:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Usage

#### Testing Adapters

Test the implemented adapters using the CLI tool:

```bash
# Set up an adapter
python -m firecircle.adapters.cli setup --provider openai

# List registered adapters
python -m firecircle.adapters.cli list

# Test an adapter
python -m firecircle.adapters.cli test --provider openai --message "Tell me about Ayni" --stream
```

#### Running Example Dialogues

**Simple Dialogue:**

```bash
# Run the basic dialogue example
python -m firecircle.examples.simple_dialogue

# Alternatively, use the console script
firecircle-demo
```

This example demonstrates:
- Setting up dialogue participants
- Configuring turn-taking policies
- Handling message flow between participants
- Transitioning between dialogue phases

**Comparative Dialogue:**

```bash
# Run the comparative dialogue between models
python -m firecircle.examples.comparative_dialogue

# Alternatively, use the console script with a topic number (0-3)
firecircle-compare --topic 2
```

This example demonstrates:
- Connecting to multiple AI providers simultaneously
- Running a structured dialogue between different models (GPT-4 and Claude)
- Comparing responses to the same prompts
- Using the Memory Store to save dialogue history

**ArangoDB Integration:**

```bash
# Install ArangoDB dependencies
pip install firecircle[arangodb]

# Set connection environment variables (optional)
export ARANGODB_HOST="http://localhost:8529"
export ARANGODB_USERNAME="root" 
export ARANGODB_PASSWORD="password"

# Run the ArangoDB example
python -m firecircle.examples.arangodb_example

# Alternatively, use the console script
firecircle-arangodb
```

This example demonstrates:
- Connecting to ArangoDB for persistent storage
- Storing and retrieving dialogues and messages
- Performing semantic search across conversations
- Managing dialogue relationships

### Running Tests

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_memory.py
pytest tests/test_orchestrator.py
```

## Architecture

The Fire Circle system follows these architectural principles:

### Message Protocol

- Messages have a standardized format with metadata, content, and visibility rules
- Support for various message types (question, proposal, reflection, etc.)
- Empty Chair concept for representing unheard perspectives

### Message Router

- Routes messages between participants according to dialogue rules
- Enforces turn-taking policies (round-robin, facilitator, etc.)
- Manages dialogue state transitions

### Memory Store

- Provides vector-based semantic storage and retrieval
- Maintains conversation history and enables context-aware responses
- Supports searching for related dialogues and messages
- Multiple implementations:
  - In-memory store for development and testing
  - Vector store for local embeddings
  - ArangoDB store for production use with persistence

### Dialogue Management

- Enforces dialogue rules and turn policies
- Supports various dialogue formats (debate, brainstorming, etc.)
- Manages dialogue phases and state transitions

## Contributing

Contributions are welcome! Please read [DESIGN.md](docs/DESIGN.md) for details on our architecture and [CLAUDE.md](CLAUDE.md) for development guidelines.

## License

[GPL v3.0](LICENSE) – because reciprocity cannot be privatized.

## Documentation

- [Manifesto](docs/manifesto.md) - Philosophical foundation
- [Design](docs/DESIGN.md) - Technical architecture
- [Development Guidelines](CLAUDE.md) - Coding standards

Latest Version: [Manifesto v0.3](docs/manifesto.md)