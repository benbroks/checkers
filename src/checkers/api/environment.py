"""
Stateless checkers environment API.

This module provides a functional, stateless interface to the checkers game.
All functions accept and return JSON-serializable state dictionaries, making it
easy to integrate with external systems or APIs.

The state representation follows the format:
{
    "pieces": [{"name": "12WN", "has_eaten": false}, ...],
    "current_turn": "W" or "B",
    "color_up": "W" or "B"
}

Actions are represented as tuples: (from_position, to_position)
where positions are integers in the range [0, 31].

Player representation:
- +1 represents White (W)
- -1 represents Black (B)
"""

from checkers.core.board import Board
from checkers.core.piece import Piece
import copy


def _board_to_state(board):
    """Convert a Board object to a JSON-serializable state dict."""
    return {
        "pieces": [
            {
                "name": piece.get_name(),
                "has_eaten": piece.get_has_eaten()
            }
            for piece in board.get_pieces()
        ],
        "current_turn": board.get_current_turn(),
        "color_up": board.get_color_up()
    }


def _state_to_board(state):
    """Convert a state dict to a Board object."""
    pieces = []
    for piece_data in state["pieces"]:
        piece = Piece(piece_data["name"])
        piece.set_has_eaten(piece_data["has_eaten"])
        pieces.append(piece)

    board = Board(pieces, state["color_up"])
    board.set_current_turn(state["current_turn"])
    return board


def _color_to_player(color):
    """Convert color string to player integer."""
    return +1 if color == "W" else -1


def _player_to_color(player):
    """Convert player integer to color string."""
    return "W" if player == +1 else "B"


def reset():
    """
    Reset the game to initial state.

    Returns:
        dict: Initial game state with all pieces in starting positions.
    """
    board = Board([], "W")  # Empty pieces, white moves up
    board.reset()
    return _board_to_state(board)


def legal_moves(state):
    """
    Get all legal moves for the current player.

    Enforces the forced jump rule: if any capture is available, only capture
    moves are returned. For multi-jump scenarios, only the first jump is
    returned; call again after executing the move to get the next jump.

    Args:
        state (dict): Current game state

    Returns:
        list[tuple]: List of legal actions as (from_position, to_position) tuples
    """
    board = _state_to_board(state)
    return board.legal_moves()


def step(state, action):
    """
    Execute an action and return the resulting state and outcome.

    This function validates the action, applies it to the board, checks for
    terminal conditions, and computes rewards.

    Args:
        state (dict): Current game state
        action (tuple): Action to execute as (from_position, to_position)

    Returns:
        tuple: (next_state, reward, done, info) where:
            - next_state (dict): The resulting game state
            - reward (float): Reward from the perspective of the player who just moved
                            +1.0 if they won, 0.0 otherwise
            - done (bool): Whether the game has terminated
            - info (dict): Additional information including:
                - "winner": +1, -1, or None
                - "captured": bool indicating if a piece was captured
                - "was_kinged": bool indicating if a piece became a king
                - "illegal_move": bool (only present if move was illegal)

    Raises:
        ValueError: If the action is not in the list of legal moves
    """
    # Make a copy of the state to avoid mutations
    board = _state_to_board(state)

    from_pos, to_pos = action

    # 1. Validate action
    legal = board.legal_moves()
    if action not in legal:
        raise ValueError(f"Illegal move: {action}. Legal moves: {legal}")

    # Track the player who is making this move
    prev_player = _color_to_player(board.get_current_turn())

    # Find the piece index for the from_position
    piece_index = None
    for i, piece in enumerate(board.get_pieces()):
        if int(piece.get_position()) == from_pos:
            piece_index = i
            break

    if piece_index is None:
        raise ValueError(f"No piece found at position {from_pos}")

    # Track if piece was already a king before move
    piece = board.get_piece_by_index(piece_index)
    was_king_before = piece.is_king()

    # Count pieces before move to detect captures
    pieces_before = len(board.get_pieces())

    # 2. Apply the move
    board.move_piece(piece_index, to_pos)

    # 3. Check if a piece was captured
    pieces_after = len(board.get_pieces())
    captured = pieces_after < pieces_before

    # 4. Check if piece was kinged
    piece_after = None
    for p in board.get_pieces():
        if int(p.get_position()) == to_pos:
            piece_after = p
            break
    was_kinged = piece_after is not None and not was_king_before and piece_after.is_king()

    # 5. Check for terminal condition
    # Note: current_turn has already been switched by move_piece (if appropriate)
    legal_next = board.legal_moves()
    done = len(legal_next) == 0

    # 6. Compute reward
    if not done:
        reward = 0.0
        winner = None
    else:
        # The player who just moved has won (opponent has no legal moves)
        winner = prev_player
        reward = 1.0  # Reward from perspective of the player who just moved

    # 7. Encode next state
    next_state = _board_to_state(board)

    # 8. Build info dict
    info = {
        "winner": winner,
        "captured": captured,
        "was_kinged": was_kinged,
    }

    return next_state, reward, done, info


def current_player(state):
    """
    Get the player whose turn it is to move.

    Args:
        state (dict): Current game state

    Returns:
        int: +1 for White, -1 for Black
    """
    return _color_to_player(state["current_turn"])


# Convenience functions for working with the stateless API

def get_winner(state):
    """
    Check if there is a winner in the current state.

    Args:
        state (dict): Current game state

    Returns:
        int or None: +1 if White won, -1 if Black won, None if game ongoing
    """
    board = _state_to_board(state)
    winner_color = board.get_winner()
    if winner_color is None:
        return None
    return _color_to_player(winner_color)


def hash_state(state):
    """
    Create a hashable representation of a game state.

    This function creates a canonical hash of the state that can be used as a
    dictionary key. The hash is stable across equivalent states regardless of
    piece ordering in the pieces list.

    Args:
        state (dict): Current game state

    Returns:
        str: A canonical hash string that can be used as a dictionary key

    Example:
        >>> state = reset()
        >>> hash1 = hash_state(state)
        >>> # Same state produces same hash
        >>> hash2 = hash_state(state)
        >>> assert hash1 == hash2
        >>> # Can be used as dict key
        >>> visited = {hash_state(state): True}
    """
    # Sort pieces by name to ensure consistent ordering
    # Name format is: <position><color><king> (e.g., "12WN", "5BY")
    sorted_pieces = sorted(state["pieces"], key=lambda p: p["name"])

    # Create canonical string representation
    pieces_str = ",".join(
        f"{p['name']}:{int(p['has_eaten'])}"
        for p in sorted_pieces
    )

    # Combine all state components
    return f"{pieces_str}|{state['current_turn']}|{state['color_up']}"


def render_state(state):
    """
    Create a simple text representation of the board state.

    Args:
        state (dict): Current game state

    Returns:
        str: ASCII art representation of the board
    """
    board = _state_to_board(state)

    # Create a coordinate-based map using the board's own position methods
    coord_map = {}
    for piece in board.get_pieces():
        pos = int(piece.get_position())
        row = board.get_row_number(pos)
        col = board.get_col_number(pos)
        color = piece.get_color()
        is_king = piece.is_king()

        if color == 'W':
            symbol = 'W' if not is_king else 'K'
        else:
            symbol = 'b' if not is_king else 'k'

        coord_map[(row, col)] = symbol

    # Build string representation
    lines = []
    lines.append("    0   1   2   3   4   5   6   7")
    lines.append("  +---+---+---+---+---+---+---+---+")

    for row in range(8):
        line = f"{row} |"
        for col in range(8):
            # Dark squares only on alternating pattern
            # In this checkers implementation, pieces start at row 0 col 0,
            # so dark squares are where (row + col) is EVEN
            is_dark = (row + col) % 2 == 0
            if is_dark:
                symbol = coord_map.get((row, col), ' ')
                line += f" {symbol} |"
            else:
                line += "///|"
        lines.append(line)
        lines.append("  +---+---+---+---+---+---+---+---+")

    lines.append(f"\nCurrent turn: {state['current_turn']} (Player {current_player(state)})")
    return '\n'.join(lines)
