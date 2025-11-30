#!/usr/bin/env python3
"""
Test script to evaluate value network on real board positions.

This script tests the value network on a few handcrafted board positions to ensure
it produces different outputs for different game states.

Usage:
    PYTHONPATH=./src python3 test/test_value_network_real_positions.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/test_value_network_real_positions.py
"""

import torch
import numpy as np
from checkers.ai.value_network import create_model


def create_position(white_men=None, white_kings=None, black_men=None, black_kings=None):
    """
    Create a board position tensor.

    Args:
        white_men: List of square indices for white men (0-31)
        white_kings: List of square indices for white kings
        black_men: List of square indices for black men
        black_kings: List of square indices for black kings

    Returns:
        Tensor of shape (1, 4, 8, 4) representing the board
    """
    # Initialize empty board
    state = torch.zeros(1, 4, 8, 4)

    # Helper to convert square index (0-31) to (row, col) in 8x4 representation
    def idx_to_pos(idx):
        row = idx // 4
        col = idx % 4
        return row, col

    # Place pieces
    if white_men:
        for idx in white_men:
            row, col = idx_to_pos(idx)
            state[0, 0, row, col] = 1

    if white_kings:
        for idx in white_kings:
            row, col = idx_to_pos(idx)
            state[0, 1, row, col] = 1

    if black_men:
        for idx in black_men:
            row, col = idx_to_pos(idx)
            state[0, 2, row, col] = 1

    if black_kings:
        for idx in black_kings:
            row, col = idx_to_pos(idx)
            state[0, 3, row, col] = 1

    return state


def main():
    print("=" * 70)
    print(" " * 12 + "VALUE NETWORK REAL POSITION TEST")
    print("=" * 70)

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create model
    print("Creating fresh model...")
    model = create_model().to(device)
    model.eval()
    print("✓ Model created")

    # Test positions
    positions = [
        {
            "name": "Empty board",
            "white_men": [],
            "white_kings": [],
            "black_men": [],
            "black_kings": []
        },
        {
            "name": "Only white pieces (3 men)",
            "white_men": [1, 2, 3],
            "white_kings": [],
            "black_men": [],
            "black_kings": []
        },
        {
            "name": "Only black pieces (3 men)",
            "white_men": [],
            "white_kings": [],
            "black_men": [28, 29, 30],
            "black_kings": []
        },
        {
            "name": "Balanced position (3v3 men)",
            "white_men": [1, 2, 3],
            "white_kings": [],
            "black_men": [28, 29, 30],
            "black_kings": []
        },
        {
            "name": "White has king advantage",
            "white_men": [1],
            "white_kings": [15],
            "black_men": [28, 29],
            "black_kings": []
        },
        {
            "name": "Black has king advantage",
            "white_men": [1, 2],
            "white_kings": [],
            "black_men": [28],
            "black_kings": [16]
        },
        {
            "name": "Material advantage white (6v3)",
            "white_men": [1, 2, 3, 5, 6, 7],
            "white_kings": [],
            "black_men": [28, 29, 30],
            "black_kings": []
        },
        {
            "name": "Material advantage black (3v6)",
            "white_men": [1, 2, 3],
            "white_kings": [],
            "black_men": [24, 25, 26, 28, 29, 30],
            "black_kings": []
        },
    ]

    print("\n" + "=" * 70)
    print("Testing different board positions")
    print("=" * 70)

    outputs = []

    for i, pos in enumerate(positions):
        # Create position
        state = create_position(
            white_men=pos["white_men"],
            white_kings=pos["white_kings"],
            black_men=pos["black_men"],
            black_kings=pos["black_kings"]
        ).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(state).item()

        outputs.append(output)

        # Print result
        w_men = len(pos["white_men"])
        w_kings = len(pos["white_kings"])
        b_men = len(pos["black_men"])
        b_kings = len(pos["black_kings"])

        print(f"\nPosition {i+1}: {pos['name']}")
        print(f"  Pieces: W={w_men}m+{w_kings}k, B={b_men}m+{b_kings}k")
        print(f"  Output: {output:+.6f}")

        # Visual indicator
        bar_length = int(abs(output) * 50)
        if output >= 0:
            bar = " " * 10 + "│" + "█" * bar_length
            lean = "White" if output > 0.01 else "Neutral"
        else:
            bar = "█" * bar_length + "│" + " " * 10
            lean = "Black"
        print(f"  [{bar}] ({lean})")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    outputs_np = np.array(outputs)
    print(f"Output range: [{outputs_np.min():.6f}, {outputs_np.max():.6f}]")
    print(f"Mean: {outputs_np.mean():.6f}")
    print(f"Std Dev: {outputs_np.std():.6f}")
    print(f"Variance: {outputs_np.var():.6f}")

    # Check if outputs are different
    unique_outputs = len(set([round(x, 6) for x in outputs]))
    print(f"\nUnique outputs: {unique_outputs}/{len(outputs)}")

    if unique_outputs >= len(outputs) - 1:
        print("✓ Model produces different outputs for different positions")
    else:
        print("⚠️  Model outputs are very similar across positions")

    if outputs_np.std() > 0.001:
        print("✓ Outputs have reasonable variance")
    else:
        print("⚠️  Variance is very low")

    print("\n" + "=" * 70)
    print("NOTE: Untrained model outputs will be small and close to 0.")
    print("After training, outputs should differentiate positions better.")
    print("=" * 70)


if __name__ == '__main__':
    main()
