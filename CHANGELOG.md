# Changelog

All notable changes to the Fire Circle project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Memory Store implementations
  - Abstract base class defining the storage interface
  - In-memory implementation for development and testing
  - Vector-based implementation with semantic search capabilities
  - ArangoDB implementation for production persistence
- Conversation Orchestrator implementation
  - Turn-taking policies (round-robin, facilitator, free-form, etc.)
  - Dialogue state management
  - Dialogue phase transitions
- Comprehensive test suite for Memory Store and Orchestrator
- Example dialogue scripts:
  - Simple dialogue demonstrating core functionality
  - Comparative dialogue between different AI models
- Console script entry points for running examples
- Unit tests for OpenAI and Anthropic adapters

### Changed
- Updated README with installation and usage instructions
- Enhanced project documentation with architecture details

## [0.1.0] - 2024-04-11

### Added
- Initial project structure
- Message Protocol implementation
- Message Router implementation
- Adapter base classes
- Anthropic adapter implementation
- CLI utilities for testing adapters
- Project documentation
  - DESIGN.md with architectural overview
  - CLAUDE.md with development guidelines
  - manifesto.md with philosophical foundation