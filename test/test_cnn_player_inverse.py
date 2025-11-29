"""
Unit test to verify that cnn_output_to_action is the true inverse of action_to_cnn_output.

Usage:
    PYTHONPATH=./src python test/test_cnn_player_inverse.py
"""

import numpy as np
from checkers.core.state_utils import action_to_cnn_output
from checkers.ai.cnn_player import cnn_output_to_action


def test_inverse_function():
    """
    Test that cnn_output_to_action correctly inverts action_to_cnn_output.

    We test all possible valid actions on a checkers board.
    """
    test_cases = []
    errors = []

    # Generate test cases for all possible board positions and directions
    for source_idx in range(32):
        source_row = source_idx // 4
        source_col = source_idx % 4

        # Test all 8 action types (4 normal moves + 4 jumps)
        # Columns 0-3: Normal moves (NW, NE, SE, SW)
        # Columns 4-7: Jump moves (NW, NE, SE, SW)
        for action_col in range(8):
            is_jump = action_col >= 4
            direction = action_col % 4  # 0=NW, 1=NE, 2=SE, 3=SW

            move_distance = 2 if is_jump else 1

            # Calculate destination
            if direction == 0:  # NW
                dest_row = source_row - move_distance
                dest_col = source_col - move_distance
            elif direction == 1:  # NE
                dest_row = source_row - move_distance
                dest_col = source_col + move_distance
            elif direction == 2:  # SE
                dest_row = source_row + move_distance
                dest_col = source_col + move_distance
            elif direction == 3:  # SW
                dest_row = source_row + move_distance
                dest_col = source_col - move_distance

            # Check if destination is valid (on board)
            if 0 <= dest_row < 8 and 0 <= dest_col < 4:
                dest_idx = dest_row * 4 + dest_col
                action = (source_idx, dest_idx)
                test_cases.append((action, action_col))

    print(f"Testing {len(test_cases)} valid action conversions...")

    # Test each case
    for action, expected_action_col in test_cases:
        # Convert action to CNN output tensor
        tensor = action_to_cnn_output(action)

        # Find where the tensor is 1
        row_indices, col_indices = np.where(tensor == 1)

        if len(row_indices) != 1:
            errors.append(f"Action {action} produced tensor with {len(row_indices)} ones (expected 1)")
            continue

        row = int(row_indices[0])
        col = int(col_indices[0])

        # Convert back using inverse function
        reconstructed_action = cnn_output_to_action(row, col)

        # Verify
        if reconstructed_action != action:
            errors.append(
                f"Failed: action={action}, tensor_pos=({row},{col}), "
                f"reconstructed={reconstructed_action}"
            )

        # Also verify the position matches expectations
        if row != action[0]:
            errors.append(f"Row mismatch: action source={action[0]}, tensor row={row}")
        if col != expected_action_col:
            errors.append(
                f"Column mismatch for action {action}: expected={expected_action_col}, got={col}"
            )

    # Report results
    if errors:
        print(f"\n❌ FAILED: {len(errors)} errors found:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print(f"✅ SUCCESS: All {len(test_cases)} conversions are correct!")
        print("The cnn_output_to_action function is a perfect inverse of action_to_cnn_output.")


if __name__ == '__main__':
    test_inverse_function()
