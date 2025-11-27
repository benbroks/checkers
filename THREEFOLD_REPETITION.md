# 3-Fold Repetition Draw Detection

## Overview

The checkers game now tracks position repetitions and enforces a **3-fold repetition** rule: if the exact same position (board configuration + current turn + has_eaten flags) occurs 3 times during a game, the game ends in a **draw** (no winner).

## Implementation Details

### State Representation Changes

The state dictionary now includes a `visit_counts` field:

```python
{
    "pieces": [{"name": "12WN", "has_eaten": false}, ...],
    "current_turn": "W" or "B",
    "color_up": "W" or "B",
    "visit_counts": {<position_hash>: count, ...}  # NEW
}
```

### How It Works

1. **Position Hashing**: Each unique board position gets a hash based on:
   - Piece positions and types
   - `has_eaten` flags
   - Current turn
   - Color orientation

2. **Visit Tracking**: The `visit_counts` dict maps position hashes to the number of times that position has been reached in the current game.

3. **Hash State Changes**: The `hash_state()` function now includes the visit count in its output:
   - Before: `<pieces>|<turn>|<color_up>`
   - After: `<pieces>|<turn>|<color_up>|<visit_count>`

   This ensures MCTS and other algorithms see different states for different visit counts.

4. **Draw Detection**: In `step()`, after each move:
   - The position hash is computed
   - Visit count for that position is incremented
   - If visit count reaches 3, the game ends with `done=True` and `winner=None`

### API Changes

#### `reset()`
- Now initializes `visit_counts` with the starting position having a count of 1

#### `step(state, action)`
- Returns updated `info` dict with new field:
  - `"threefold_repetition"`: `bool` - True if game ended by 3-fold repetition

- When `threefold_repetition=True`:
  - `done` = `True`
  - `winner` = `None` (draw)
  - `reward` = `0.0`

#### `hash_state(state)`
- Now includes visit count in the hash
- Different visit counts produce different hashes
- This ensures MCTS treats repeated positions differently based on repetition count

#### New Internal Function: `_position_hash(state)`
- Creates hash of position WITHOUT visit counts
- Used internally to track position repetitions
- Not exposed in public API

## Example Usage

```python
from checkers.api.environment import reset, step, legal_moves

state = reset()

# Play game...
for _ in range(100):
    moves = legal_moves(state)
    if not moves:
        break

    move = moves[0]  # Pick some move
    state, reward, done, info = step(state, move)

    if done:
        if info['threefold_repetition']:
            print("Game ended in a draw by 3-fold repetition!")
            print(f"Winner: {info['winner']}")  # Will be None
        else:
            print(f"Game ended - Winner: {info['winner']}")
        break
```

## Testing

Run the test suite:

```bash
# Simple visit count tracking test
PYTHONPATH=src python3 test/test_threefold_simple.py

# Comprehensive tests
PYTHONPATH=src python3 test/test_threefold_repetition.py
```

## Impact on MCTS

The MCTS implementation will now:
- See different states for the same position at different visit counts
- Potentially learn to avoid repetitions or force draws when losing
- Correctly handle draw scenarios in the reward calculation

## Backward Compatibility

**Breaking Change**: The state representation has changed. Old saved states or MCTS data may need migration:

1. Old states missing `visit_counts` will get an empty dict `{}`
2. `hash_state()` now includes visit count - old hashes won't match
3. MCTS data (N_s, N_s_a, R_s_a) should be retrained with new state representation

To migrate existing code:
```python
# If you have an old state without visit_counts
if "visit_counts" not in state:
    state["visit_counts"] = {}
    # Optionally record the current position
    from checkers.api.environment import _position_hash
    pos_hash = _position_hash(state)
    state["visit_counts"][pos_hash] = 1
```
