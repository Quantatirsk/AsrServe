# Contributing to AsrServe

Thank you for your interest in contributing to AsrServe! This guide will help you set up your development environment and understand our contribution process.

## Table of Contents
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Branching Strategy](#branching-strategy)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Development Environment

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- Rust toolchain (for CPU backend)
- Docker (for container builds)
- FFmpeg, libsndfile, OpenBLAS (system dependencies)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/AsrServe.git
cd AsrServe

# Install dependencies
uv sync --frozen

# Build the Rust CPU backend (if needed)
./scripts/build-rust.sh

# Run the application
uv run python start.py
```

### Development Dependencies

```bash
# Install development tools
uv pip install ruff ty pytest pytest-cov httpx
```

## Code Style

We use the following tools to maintain code quality:

- **ruff** - Fast Python linter and formatter
- **ty** - Fast type checker

### Running Checks

```bash
# Lint
ruff check app/ tests/

# Format
ruff format app/ tests/

# Type check
ty app/
```

### Style Guidelines
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Keep lines under 100 characters
- Use f-strings for string formatting

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_api_contract.py

# Run specific test
pytest tests/test_api_contract.py::APIContractTest::test_offline_all_response_formats
```

### Writing Tests
- Use `pytest` for test discovery and execution
- Use `unittest.mock` for mocking
- Test both happy paths and error conditions
- Aim for high coverage of business logic
- Mock external dependencies (models, network) in unit tests

### Coverage Requirements
- Minimum 70% line coverage
- New code should not decrease overall coverage

## Branching Strategy

We use feature branches for all development:

```
main ← feat/issue-N-short-description
  ↑        ↑
  |        └── Your feature branch
  └── Upstream main (Quantatirsk/AsrServe)
```

### Branch Naming
- `feat/issue-N-description` - New features
- `fix/issue-N-description` - Bug fixes
- `docs/issue-N-description` - Documentation changes
- `chore/issue-N-description` - Maintenance tasks

### Example
```bash
git checkout -b feat/issue-1-localization-en
```

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** with appropriate tests
3. **Run all checks**: lint, format, type check, tests
4. **Push your branch** and create a Pull Request
5. **Address review comments** and update the PR
6. **Squash and merge** once approved

### PR Checklist
- [ ] Code follows style guidelines (ruff check passes)
- [ ] Type checking passes (ty)
- [ ] All tests pass (pytest)
- [ ] Coverage is maintained or improved
- [ ] Documentation is updated (if needed)
- [ ] PR description explains the changes and motivation

### PR Description Template
```markdown
## Summary
Brief description of the changes.

## Motivation
Why are these changes needed? Link to issue if applicable.

## Testing
- What was tested
- How it was tested
- Test results

## Screenshots/Demos
(For UI changes)

## Checklist
- [ ] Lint passes
- [ ] Type check passes
- [ ] Tests pass
- [ ] Coverage maintained
- [ ] Documentation updated
```

## Issue Guidelines

When creating an issue, please:
- Use a clear, descriptive title
- Provide context and motivation
- Include steps to reproduce (for bugs)
- Specify expected vs actual behavior
- Add relevant labels

### Bug Report Template
```markdown
## Description
What is the issue?

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen?

## Actual Behavior
What actually happened?

## Environment
- OS:
- Python version:
- AsrServe version:
- GPU (if applicable):
```

### Feature Request Template
```markdown
## Summary
Brief description of the feature.

## Motivation
Why is this feature needed?

## Proposed Implementation
How should this be implemented?

## Alternatives Considered
What other approaches were considered?
```

## Release Process

Releases are managed by the maintainers. To request a release:
1. Ensure all changes are merged to `main`
2. Create an issue with the proposed release version
3. Maintainers will create a release tag and update Docker images

## Communication

- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for general questions
- Be respectful and constructive in all communications

## License

By contributing, you agree that your contributions will be licensed under the project's license.
