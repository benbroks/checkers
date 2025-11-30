#!/usr/bin/env python3
"""
Analyze the value network dataset for quality issues.

This script checks:
1. Class balance (wins/losses/draws)
2. Data distribution
3. Potential label quality issues

Usage:
    PYTHONPATH=./src python3 test/analyze_value_dataset.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/analyze_value_dataset.py
"""

import torch
import numpy as np
import pickle
from collections import Counter


def main():
    print("=" * 70)
    print(" " * 18 + "VALUE DATASET ANALYSIS")
    print("=" * 70)

    dataset_path = "data/value_network_dataset.pkl"
    print(f"\nLoading dataset from: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"Total samples: {len(samples)}")

    # Extract values
    values = np.array([v for _, v in samples])

    print("\n" + "=" * 70)
    print("VALUE DISTRIBUTION")
    print("=" * 70)

    # Count exact outcomes
    wins = np.sum(values == 1.0)
    losses = np.sum(values == -1.0)
    draws = np.sum(values == 0.0)
    other = len(values) - (wins + losses + draws)

    print(f"\nExact outcomes:")
    print(f"  Wins  (+1.0): {wins:5d} ({100*wins/len(values):5.1f}%)")
    print(f"  Draws ( 0.0): {draws:5d} ({100*draws/len(values):5.1f}%)")
    print(f"  Losses(-1.0): {losses:5d} ({100*losses/len(values):5.1f}%)")
    print(f"  Other:        {other:5d} ({100*other/len(values):5.1f}%)")

    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean:   {values.mean():.4f}")
    print(f"  Median: {np.median(values):.4f}")
    print(f"  Std:    {values.std():.4f}")
    print(f"  Min:    {values.min():.4f}")
    print(f"  Max:    {values.max():.4f}")

    # Check for class imbalance
    print("\n" + "=" * 70)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 70)

    positive = np.sum(values > 0)
    negative = np.sum(values < 0)
    zero = np.sum(values == 0)

    print(f"\nBinary classification:")
    print(f"  Positive (>0): {positive:5d} ({100*positive/len(values):5.1f}%)")
    print(f"  Zero     (=0): {zero:5d} ({100*zero/len(values):5.1f}%)")
    print(f"  Negative (<0): {negative:5d} ({100*negative/len(values):5.1f}%)")

    if positive > 0 and negative > 0:
        imbalance_ratio = max(positive, negative) / min(positive, negative)
        print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2:
            print(f"  ⚠️  WARNING: Significant class imbalance!")
        else:
            print(f"  ✓ Class balance is reasonable")

    # Check if draws are too common
    if zero / len(values) > 0.5:
        print(f"\n  ⚠️  WARNING: More than 50% draws! Model may learn to predict 0")

    # Distribution histogram
    print("\n" + "=" * 70)
    print("VALUE HISTOGRAM")
    print("=" * 70)

    bins = [
        (-1.0, -0.9, "Strong losses"),
        (-0.9, -0.5, "Moderate losses"),
        (-0.5, -0.1, "Slight losses"),
        (-0.1, 0.1, "Draws"),
        (0.1, 0.5, "Slight wins"),
        (0.5, 0.9, "Moderate wins"),
        (0.9, 1.1, "Strong wins")
    ]

    for low, high, label in bins:
        count = np.sum((values >= low) & (values < high))
        percentage = 100 * count / len(values)
        bar = "█" * int(percentage / 2)
        print(f"[{low:+.1f}, {high:+.1f}): {count:5d} ({percentage:5.1f}%) {bar:<30} {label}")

    # Check for unique values
    print("\n" + "=" * 70)
    print("UNIQUE VALUES")
    print("=" * 70)

    unique_values = np.unique(values)
    print(f"\nNumber of unique values: {len(unique_values)}")

    if len(unique_values) <= 20:
        print("\nAll unique values:")
        value_counts = Counter(values)
        for val in sorted(unique_values):
            count = value_counts[val]
            print(f"  {val:+.4f}: {count:5d} samples ({100*count/len(values):5.1f}%)")

    # Check states
    print("\n" + "=" * 70)
    print("STATE ANALYSIS")
    print("=" * 70)

    # Sample a few states to check piece counts
    print("\nAnalyzing first 100 states...")
    piece_counts = []
    for i in range(min(100, len(samples))):
        state_tensor, _ = samples[i]
        # state_tensor shape: (4, 8, 4)
        # Channel 0: White men, 1: White kings, 2: Black men, 3: Black kings
        white_pieces = state_tensor[0].sum() + state_tensor[1].sum()
        black_pieces = state_tensor[2].sum() + state_tensor[3].sum()
        total_pieces = white_pieces + black_pieces
        piece_counts.append(total_pieces)

    piece_counts = np.array(piece_counts)
    print(f"  Average pieces per position: {piece_counts.mean():.1f}")
    print(f"  Min pieces: {piece_counts.min():.0f}")
    print(f"  Max pieces: {piece_counts.max():.0f}")

    if piece_counts.mean() < 5:
        print("  ⚠️  WARNING: Very few pieces on average - mostly endgame positions")

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    issues = []

    if imbalance_ratio > 2:
        issues.append("High class imbalance - consider rebalancing or weighted loss")

    if zero / len(values) > 0.5:
        issues.append("Too many draws - policy may be too weak or games too short")

    if values.std() < 0.4:
        issues.append("Low variance in outcomes - not enough diversity")

    if len(samples) < 10000:
        issues.append("Small dataset - consider generating more games")

    if issues:
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ Dataset looks healthy!")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
