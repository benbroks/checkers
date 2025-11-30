#!/usr/bin/env python3
"""
Test script to demonstrate softmax sampling vs greedy selection.

Usage:
    PYTHONPATH=./src python3 test/test_softmax_sampling.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/test_softmax_sampling.py
"""

import torch
from checkers.api.environment import reset
from checkers.ai.sl_action_policy import create_model
from checkers.ai.cnn_player import select_cnn_move, select_cnn_move_softmax


def main():
    print("Softmax Sampling vs Greedy Selection Test")
    print("=" * 60)

    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}\n")

    # Load model
    print("Loading policy network from checkpoints/rl/iter_1990.pth...")
    model = create_model()
    checkpoint = torch.load('checkpoints/rl/iter_1990.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded!\n")

    # Test from initial position
    state = reset()

    print("Testing from initial board position:")
    print("-" * 60)

    # Greedy selection (deterministic - always picks best)
    print("\nGreedy selection (5 samples - should be identical):")
    greedy_moves = []
    for i in range(5):
        move = select_cnn_move(model, state, device)
        greedy_moves.append(move)
        print(f"  Sample {i+1}: {move}")

    all_same = all(m == greedy_moves[0] for m in greedy_moves)
    print(f"  All greedy moves identical: {all_same} ✓" if all_same else f"  ERROR: Greedy moves differ!")

    # Softmax sampling (stochastic - varies)
    print("\nSoftmax sampling (5 samples - should vary):")
    softmax_moves = []
    for i in range(5):
        move = select_cnn_move_softmax(model, state, device, temperature=1.0)
        softmax_moves.append(move)
        print(f"  Sample {i+1}: {move}")

    unique_count = len(set(softmax_moves))
    print(f"  Unique moves: {unique_count}/5")
    print(f"  Sampling is stochastic: {unique_count > 1} ✓" if unique_count > 1 else "  Note: Got same move by chance")

    # Test different temperatures
    print("\n" + "-" * 60)
    print("Temperature effect on diversity:")
    print("-" * 60)

    for temp in [0.5, 1.0, 2.0]:
        print(f"\nTemperature = {temp}:")
        temp_moves = []
        for i in range(10):
            move = select_cnn_move_softmax(model, state, device, temperature=temp)
            temp_moves.append(move)

        unique_count = len(set(temp_moves))
        print(f"  Unique moves: {unique_count}/10")
        print(f"  Most common: {max(set(temp_moves), key=temp_moves.count)} (appears {temp_moves.count(max(set(temp_moves), key=temp_moves.count))} times)")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("Lower temperature → more greedy (less diversity)")
    print("Higher temperature → more exploratory (more diversity)")
    print("=" * 60)


if __name__ == '__main__':
    main()
