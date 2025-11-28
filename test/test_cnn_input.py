"""
Test script showing how to convert board states to CNN input.
Run with: PYTHONPATH=./src python3 test_cnn_input.py
"""
from checkers.api.environment import reset, _state_to_board
from checkers.cli.interface import display_board
from checkers.core.state_utils import state_to_cnn_input
import numpy as np


def visualize_channel(channel_data, channel_name):
    """Visualize a single channel of the CNN input."""
    print(f"\n{channel_name}:")
    print("     Col: 0  1  2  3")
    print("   " + "-" * 20)
    for row in range(8):
        row_str = f" Row {row}:"
        for col in range(4):
            val = int(channel_data[row, col])
            row_str += f" {val} "
        print(row_str)


def main():
    print("="*60)
    print("Board State to CNN Input Conversion")
    print("="*60)

    # Get initial game state
    state = reset()

    print("\nInitial game state:")
    print(f"  Total pieces: {len(state['pieces'])}")
    print(f"  Current turn: {state['current_turn']}")
    board = _state_to_board(state)
    display_board(board)

    # Convert to CNN input
    cnn_input = state_to_cnn_input(state)

    print(f"\nCNN Input shape: {cnn_input.shape}")
    print("  4 channels (Current Player Man, Current Player King, Opposing Player Man, Opposing Player King)")
    print("  8 rows × 4 columns")

    # Count pieces in each channel
    channel_names = ['Current Player Man', 'Current Player King', 'Opposing Player Man', 'Opposing Player King']
    print("\nPiece counts per channel:")
    for i, name in enumerate(channel_names):
        count = int(np.sum(cnn_input[i]))
        print(f"  Channel {i} ({name}): {count} pieces")

    # Visualize each channel
    print("\n" + "="*60)
    print("Channel Visualizations (1 = piece present, 0 = empty)")
    print("="*60)

    for i, name in enumerate(channel_names):
        visualize_channel(cnn_input[i], f"Channel {i}: {name}")

    print("\n" + "="*60)
    print("Position Mapping Reference")
    print("="*60)
    print("\nPosition number to (row, col) mapping:")
    print("  row = position // 4")
    print("  col = position % 4")
    print("\nExamples:")
    for pos in [0, 1, 4, 5, 20, 21, 31]:
        row = pos // 4
        col = pos % 4
        print(f"  Position {pos:2d} -> (row={row}, col={col})")

    print("\n" + "="*60)
    print("Using with PyTorch CNN")
    print("="*60)

if __name__ == '__main__':
    main()
