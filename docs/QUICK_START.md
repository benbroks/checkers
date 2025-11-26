# Quick Start: Stateless Checkers API

This guide gets you up and running with the stateless checkers API in under 5 minutes.

## Installation

```bash
# Clone and setup
git clone <your-repo>
cd python-checkers

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (if you have requirements.txt)
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Game Loop

```python
from checkers_env import reset, legal_moves, step, current_player, render_state
import random

# Initialize game
state = reset()
print(render_state(state))

# Play until done
done = False
while not done:
    moves = legal_moves(state)
    action = random.choice(moves)  # Random move

    next_state, reward, done, info = step(state, action)

    print(f"Player {current_player(state)} -> {action}")
    if done:
        print(f"Winner: Player {info['winner']}")

    state = next_state
```

### 2. Self-Play for Training Data

```python
from checkers_env import reset, legal_moves, step, current_player

def generate_game():
    state = reset()
    trajectory = []
    done = False

    while not done:
        moves = legal_moves(state)
        action = random.choice(moves)

        trajectory.append({
            "state": state,
            "action": action,
            "player": current_player(state)
        })

        next_state, reward, done, info = step(state, action)
        state = next_state

    # Assign value targets
    winner = info['winner']
    for entry in trajectory:
        if winner is None:
            entry['value'] = 0.0
        elif entry['player'] == winner:
            entry['value'] = 1.0
        else:
            entry['value'] = -1.0

    return trajectory

# Generate training data
games = [generate_game() for _ in range(10)]
print(f"Generated {sum(len(g) for g in games)} positions")
```

### 3. Web API Example

```python
from flask import Flask, request, jsonify
from checkers_env import reset, legal_moves, step

app = Flask(__name__)

@app.route('/new', methods=['POST'])
def new_game():
    return jsonify(reset())

@app.route('/moves', methods=['POST'])
def get_moves():
    state = request.json
    moves = legal_moves(state)
    return jsonify({"moves": moves})

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    state = data['state']
    action = tuple(data['action'])

    next_state, reward, done, info = step(state, action)
    return jsonify({
        "state": next_state,
        "reward": reward,
        "done": done,
        "winner": info['winner']
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## Running the Examples

### Test Basic Functionality
```bash
python3 test_stateless_api.py
```

### Play Random Games
```bash
# Quick analysis without board display
python3 example_game_loop.py

# Watch the full game with board displayed after each move
python3 example_game_loop.py --verbose
# or
python3 example_game_loop.py -v
```

### Generate Training Data
```bash
python3 example_training_data.py
```

## API Reference (Quick)

| Function | Signature | Description |
|----------|-----------|-------------|
| `reset()` | `-> state` | Start new game |
| `legal_moves(state)` | `-> list[action]` | Get legal moves |
| `step(state, action)` | `-> (next_state, reward, done, info)` | Execute move |
| `current_player(state)` | `-> {+1, -1}` | Get current player |
| `hash_state(state)` | `-> str` | Create hashable state key |

### State Format
```python
{
    "pieces": [
        {"name": "12WN", "has_eaten": False},
        ...
    ],
    "current_turn": "W",  # "W" or "B"
    "color_up": "W"
}
```

### Action Format
```python
action = (from_position, to_position)
# Example: (20, 16) means move from position 20 to 16
# Positions are integers 0-31
```

### Player Convention
- `+1` = White (W)
- `-1` = Black (B)

### Reward Convention
- Reward is always from perspective of player who just moved
- `+1.0` = Player who moved won
- `0.0` = Game continues
- `None` = Draw (if implemented)

## Common Patterns

### Check if Game is Over
```python
moves = legal_moves(state)
if not moves:
    print("Game over!")
```

### Serialize/Save State
```python
import json

# Save
with open('game.json', 'w') as f:
    json.dump(state, f)

# Load
with open('game.json', 'r') as f:
    state = json.load(f)
```

### Handle Multi-Jumps
```python
# After a capture, check for more jumps
next_state, reward, done, info = step(state, capture_action)

if info['captured'] and not done:
    # Check if same piece has more jumps
    next_moves = legal_moves(next_state)
    # Continue with next jump
```

### Use State Hashing for MCTS
```python
from checkers_env import reset, hash_state, legal_moves, step

# Transposition table for MCTS
transposition_table = {}

state = reset()
key = hash_state(state)

# Check if state already visited
if key in transposition_table:
    node_info = transposition_table[key]
else:
    # Create new node
    transposition_table[key] = {
        'visits': 0,
        'value': 0.0,
        'prior': 0.0
    }
```

## Next Steps

- Read [STATELESS_API.md](STATELESS_API.md) for complete documentation
- Implement MCTS for better move selection
- Add neural network for position evaluation
- Build web interface
- Implement training pipeline

## Key Design Principles

1. **Stateless** - Functions don't modify inputs, always return new state
2. **Serializable** - All state is JSON-compatible
3. **Functional** - No hidden state, no side effects
4. **Simple** - Minimal API surface, easy to understand

## Troubleshooting

**Q: ValueError: Illegal move**
- Check that action is in `legal_moves(state)`
- Forced jump rule: if captures available, only captures are legal

**Q: Multi-jumps not working**
- Call `legal_moves()` again after each jump
- Turn won't switch until all jumps complete

**Q: Can't serialize state**
- State is already JSON-compatible
- Use `json.dumps(state)` to serialize

## Performance Tips

- Use `legal_moves()` result to validate actions before `step()`
- Cache state encodings for neural networks
- Batch multiple games for parallel processing
- Consider using numpy/torch for state representation in production
