# Checkers in Python

## About
From [Wikipedia](https://en.wikipedia.org/wiki/Draughts):
>Draughts or checkers is a group of strategy board games for two players which involve diagonal moves of uniform game pieces and mandatory captures by jumping over opponent pieces.

This was made using Python and the pygame module with the goal of studying OOP.


## Installation and usage
1. Clone the project with `git clone https://github.com/lucaskenji/python-checkers`.
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the virtual environment:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install the package and dependencies: `pip install -e .`
5. Run the GUI game: `python -m test.play_gui <gamemode>` (gamemode can be "cpu" or "pvp")
   - Or run the CLI version: `python -m checkers.cli.interface`
6. Run tests: `pytest test/`

## Project Structure

```
python-checkers/
├── src/checkers/          # Main package
│   ├── core/              # Core game logic (Board, Piece, utils)
│   ├── ai/                # AI implementations (minimax, MCTS)
│   ├── gui/               # Pygame GUI components
│   ├── cli/               # Command-line interface
│   └── api/               # Stateless API for ML/RL
├── test/                  # Tests and example scripts
│   ├── test_*.py          # Unit tests
│   ├── play_gui.py        # GUI example
│   └── play_mcts.py       # MCTS example
└── docs/                  # Documentation
```


## Example

![Screenshot of the game](https://github.com/lucaskenji/python-checkers/blob/master/preview/screenshot.png)


## Stateless API for Machine Learning

A new **stateless, functional API** is available for integrating checkers with ML frameworks, web APIs, and other systems. This API provides:

- **Pure functional interface** - all state is JSON-serializable
- **Easy integration** with AlphaZero-style training, MCTS, and RL
- **Clean semantics** for self-play and training data generation
- **No object management** - just pass state dictionaries

**Quick example:**
```python
from checkers.api.environment import reset, legal_moves, step, current_player

# Start game
state = reset()

# Get legal moves
moves = legal_moves(state)

# Execute move
action = moves[0]
next_state, reward, done, info = step(state, action)
```

**API functions:**
- `reset() -> state` - Initialize new game
- `legal_moves(state) -> list[action]` - Get legal moves
- `step(state, action) -> (next_state, reward, done, info)` - Execute move
- `current_player(state) -> {+1, -1}` - Get current player

See [STATELESS_API.md](STATELESS_API.md) for complete documentation and examples.

## Singleplayer
The computer you can play against in this game is a fairly simple one implemented using the [minimax algorithm](https://en.wikipedia.org/wiki/Minimax).

In a nutshell, it works by simulating every possible outcome from the current board and assuming each player will make the "best" move.
This is a rather simple algorithm, which means the computer will not play using any strategies such as baiting the opponent to jump one of its pieces.

Inspired / branched off of @lucaskenji's [repo](https://github.com/lucaskenji/python-checkers).