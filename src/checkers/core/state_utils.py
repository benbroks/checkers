import numpy as np
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


def state_to_cnn_input(state):
    f"""
    Convert a board state to CNN input tensor format.

    The CNN expects 4 channels of 8x4 input (height=8, width=4).
    - Channel 0: Current Player Man pieces (non-king current player pieces)
    - Channel 1: Current Player King pieces
    - Channel 2: Opposing Player Man pieces (non-king opposing player pieces)
    - Channel 3: Opposing Player King pieces

    Position mapping:
    - row = position // 4
    - col = position % 4

    Args:
        state: State dict
    Returns:
        numpy array of shape (4, 8, 4) where:
        - 4 channels (current player man, current player king, opposing player man, opposing player king)
        - 8 rows
        - 4 columns
    """
    current_player = state['current_turn']

    # Initialize 4 channels of 8x4 with zeros
    cnn_input = np.zeros((4, 8, 4), dtype=int)

    # Process each piece
    for piece_data in state['pieces']:
        name = piece_data['name']

        # Parse piece name format: <position><color><king>
        # Example: "12WN" = position 12, White, Not king
        # Example: "5BY" = position 5, Black, Yes king
        position_str = ""
        for char in name:
            if char.isdigit():
                position_str += char
            else:
                break

        position = int(position_str)
        color = name[len(position_str)]  # 'W' or 'B'
        is_king = name[len(position_str) + 1] == 'Y'  # 'Y' or 'N'

        # Calculate row and column
        row = position // 4
        col = position % 4

        # Determine which channel
        if color == current_player:
            if is_king:
                channel = 1  # Current Player King
            else:
                channel = 0  # Current Player Man
        else:  # color == 'B'
            if is_king:
                channel = 3  # Opposing Player King
            else:
                channel = 2  # Opposing Player Man

        # Set the position to 1
        cnn_input[channel, row, col] = 1

    return cnn_input


def action_to_cnn_output(action: tuple[int, int]) -> np.ndarray:
    """
    Each row corresponds to the source index.
    Each column corresponds to a potential action. There are 8 columns.
    The first four columns correspond to "taking"/"jumping" moves.
    The latter four columns are just normal moves.
    The directions are:
    - NW (cols 0 & 4)
    - NE (cols 1 & 5)
    - SE (cols 2 & 6)
    - SW (cols 3 & 7)
    """
    source_idx = action[0]
    dest_idx = action[1]

    source_row, source_col = source_idx // 4, source_idx % 4
    dest_row, dest_col = dest_idx // 4, dest_idx % 4

    # is jump?
    is_jump = abs(source_row - dest_row) == 2

    # Nw/NE/SE/SW
    if dest_row < source_row:
        # NW
        if dest_col < source_col:
            direction = 0
        # NE
        else:
            direction = 1
    else:
        # SE
        if dest_col > source_col:
            direction = 2
        # SW
        else:
            direction = 3
    action_direction_idx = 4 * is_jump + direction

    base_output = np.zeros((32, 8), dtype=int)
    base_output[source_idx, action_direction_idx] = 1

    return base_output
