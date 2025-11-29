# Claude Development Guidelines

This document provides guidelines for AI assistants (like Claude) working on this codebase.

## Project Structure

### Test Scripts Location

**All test scripts should be placed in the `test/` directory.**

This includes:
- Unit tests (e.g., `test_board.py`, `test_piece.py`)
- Integration tests
- Example/demo scripts (e.g., `test_pdn_generator.py`, `test_cnn_forward_pass.py`)
- Manual test scripts (e.g., `test_pdn_manual.py`)
- Interactive test applications (e.g., `play_gui.py`, `play_cli_with_mcts.py`)

**Rationale:** Keeping all test-related files in `test/` maintains a clean project root and makes it easy to:
- Find all tests in one location
- Run tests collectively
- Distinguish test code from production code
- Keep the repository organized

### Source Code Location

Production code should be organized under `src/checkers/`:
- `src/checkers/core/` - Core game logic (Board, Piece, utilities)
- `src/checkers/api/` - Stateless API environment
- `src/checkers/ai/` - AI implementations (MCTS, minimax, neural networks)
- `src/checkers/gui/` - GUI components
- `src/checkers/cli/` - CLI interface

## Running Tests

All test scripts should be runnable with:
```bash
PYTHONPATH=./src python3 test/<test_file>.py
```

Or with venv:
```bash
PYTHONPATH=./src ./venv/bin/python3 test/<test_file>.py
```

## Code Style

- Use clear, descriptive function and variable names
- Include docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Prefer readability over cleverness

## Machine Learning Components

### Neural Network Architecture
- Input format: 4 channels × 8 rows × 4 columns (4, 8, 4)
  - Channel 0: White Man pieces
  - Channel 1: White King pieces
  - Channel 2: Black Man pieces
  - Channel 3: Black King pieces
- Output format: 256-dimensional vector (8 × 32 action space)

### State Representation
Use `state_to_cnn_input(state)` from `checkers.core.state_utils` to convert board states to CNN input tensors.

## PDN (Portable Draughts Notation) Parsing

- Use `parse_pdn_games(filepath)` to iterate through games one at a time (generator)
- Use `parse_pdn_file(filepath)` to get all moves from all games as a single list
- Multi-jump captures are split by default: `26x17x10x1` → `['26x17', '17x10', '10x1']`

## Dependencies

Core dependencies:
- `pygame` - GUI support
- `numpy` - Numerical operations
- `torch` - Neural network support (for AI components)

Install with:
```bash
./venv/bin/pip install pygame numpy torch
```
