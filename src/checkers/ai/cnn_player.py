"""
CNN Action Policy Player.

Provides functions to use a trained CNN action policy model to select moves
during gameplay. The CNN outputs a (32, 8) tensor representing action logits,
which is filtered to only consider legal moves.
"""

import numpy as np
import torch
from checkers.core.state_utils import state_to_cnn_input, action_to_cnn_output
from checkers.api.environment import legal_moves, step


def cnn_output_to_action(source_idx: int, action_col: int) -> tuple[int, int]:
    """
    Convert CNN output position to action tuple.

    This is the inverse function of action_to_cnn_output().

    Args:
        source_idx: Row in CNN output (0-31) - the piece position
        action_col: Column in CNN output (0-7) - the action direction
            - Columns 0-3: Normal moves (NW, NE, SE, SW)
            - Columns 4-7: Jump moves (NW, NE, SE, SW)

    Returns:
        (source_idx, dest_idx) tuple
    """
    # Decode column index
    # action_to_cnn_output uses: action_direction_idx = 4 * is_jump + direction
    # So columns 0-3 are normal moves (is_jump=False)
    # And columns 4-7 are jump moves (is_jump=True)
    is_jump = action_col >= 4
    direction = action_col % 4  # 0=NW, 1=NE, 2=SE, 3=SW

    # Calculate source position coordinates
    source_row = source_idx // 4
    source_col = source_idx % 4

    # Calculate destination based on direction
    # From action_to_cnn_output:
    # - direction 0: dest_row < source_row AND dest_col < source_col (NW)
    # - direction 1: dest_row < source_row AND dest_col > source_col (NE)
    # - direction 2: dest_row > source_row AND dest_col > source_col (SE)
    # - direction 3: dest_row > source_row AND dest_col < source_col (SW)
    move_distance = 2 if is_jump else 1

    if direction == 0:  # NW: up and left
        dest_row = source_row - move_distance
        dest_col = source_col - move_distance
    elif direction == 1:  # NE: up and right
        dest_row = source_row - move_distance
        dest_col = source_col + move_distance
    elif direction == 2:  # SE: down and right
        dest_row = source_row + move_distance
        dest_col = source_col + move_distance
    else:  # direction == 3, SW: down and left
        dest_row = source_row + move_distance
        dest_col = source_col - move_distance

    # Convert back to position index
    dest_idx = dest_row * 4 + dest_col

    return (source_idx, dest_idx)


def select_cnn_move(model, state, device='cpu'):
    """
    Select the best legal move using the CNN action policy.

    The function:
    1. Converts the state to CNN input format
    2. Runs forward pass through the model
    3. Filters the output to only consider legal moves
    4. Returns the legal move with the highest value

    Args:
        model: Trained CheckersCNN model
        state: Current game state dict
        device: torch device ('cpu', 'mps', 'cuda')

    Returns:
        (source_idx, dest_idx): The selected action tuple

    Raises:
        ValueError: If no legal moves are available
    """
    # Get legal moves
    legal_actions = legal_moves(state)

    if not legal_actions:
        raise ValueError("No legal moves available")

    # Convert state to CNN input
    cnn_input = state_to_cnn_input(state)
    tensor_input = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(tensor_input)  # (1, 32, 8)

    output = output.squeeze(0)  # (32, 8)

    # Convert legal actions to CNN output positions and find best move
    best_value = float('-inf')
    best_action = None

    for action in legal_actions:
        # Convert action to CNN tensor position
        action_tensor = action_to_cnn_output(action)
        # Find the (row, col) where action_tensor == 1
        row, col = np.where(action_tensor == 1)

        # Get the value from the model output
        value = output[int(row[0]), int(col[0])].item()

        if value > best_value:
            best_value = value
            best_action = action

    return best_action


def single_turn_cnn_player(model, state, device='cpu'):
    """
    Execute one turn using CNN action policy.

    Args:
        model: Trained CheckersCNN model
        state: Current game state dict
        device: torch device ('cpu', 'mps', 'cuda')

    Returns:
        (next_state, reward, done, info): Result of the move
    """
    # Select move using CNN
    action = select_cnn_move(model, state, device)

    # Execute the move
    next_state, reward, done, info = step(state, action)

    return next_state, reward, done, info
