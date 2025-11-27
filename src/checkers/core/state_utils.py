"""
Shared utilities for state representation and hashing.

This module provides functions for creating canonical representations
of game states that can be used across both the core Board class and
the API layer.
"""


def create_position_hash(board):
    """
    Create a hash of the board position WITHOUT visit counts.

    This is used to track position repetitions for draw detection.

    Args:
        board: Board object with pieces, current_turn, color_up

    Returns:
        str: A canonical hash of the board position only
    """
    # Sort pieces by name to ensure consistent ordering
    # Name format is: <position><color><king> (e.g., "12WN", "5BY")
    sorted_pieces = sorted(board.get_pieces(), key=lambda p: p.get_name())

    # Create canonical string representation
    pieces_str = ",".join(
        f"{p.get_name()}:{int(p.get_has_eaten())}"
        for p in sorted_pieces
    )

    # Combine all state components (excluding visit_counts)
    return f"{pieces_str}|{board.get_current_turn()}|{board.get_color_up()}"


def create_state_hash(board):
    """
    Create a hashable representation of the complete game state.

    This includes the position hash AND the current visit count for that position.
    Used by MCTS and other algorithms to distinguish between states with different
    repetition counts.

    Args:
        board: Board object with pieces, current_turn, color_up, visit_counts

    Returns:
        str: A canonical hash string including visit count
    """
    position_hash = create_position_hash(board)

    # Get the visit count for the current position
    visit_counts = board.get_visit_counts()
    current_count = visit_counts.get(position_hash, 0)

    # Include visit count in the hash
    return f"{position_hash}|{current_count}"
