#!/usr/bin/env python3
"""
Inspect the generated value network dataset.

Usage:
    PYTHONPATH=./src python3 test/inspect_value_dataset.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/inspect_value_dataset.py
"""

import torch
from checkers.ai.value_network_dataset import CheckersValueNetworkDataset


def main():
    print("Value Network Dataset Inspector")
    print("=" * 60)

    # Load dataset (will use cached version if available)
    try:
        dataset = CheckersValueNetworkDataset(
            max_games=100,
            checkpoint_path='checkpoints/rl/iter_1990.pth',
            device='cpu',  # Use CPU for inspection
            dataset_cache_path='data/value_network_dataset_100.pkl'
        )

        print("\n" + "=" * 60)
        print("DATASET INSPECTION")
        print("=" * 60)
        print(f"Total samples: {len(dataset)}")

        if len(dataset) == 0:
            print("Dataset is empty! Run generate_value_dataset.py first.")
            return

        # Sample inspection
        print("\n" + "-" * 60)
        print("Sample Inspection (first 5 samples)")
        print("-" * 60)

        for i in range(min(5, len(dataset))):
            state_tensor, value = dataset[i]
            print(f"\nSample {i}:")
            print(f"  State shape: {state_tensor.shape}")
            print(f"  Value: {value.item()}")
            print(f"  State channels (W man, W king, B man, B king):")
            for ch in range(4):
                piece_count = (state_tensor[ch] > 0).sum().item()
                print(f"    Channel {ch}: {piece_count} pieces")

        # Value distribution
        print("\n" + "-" * 60)
        print("Value Distribution")
        print("-" * 60)

        values = [dataset[i][1].item() for i in range(len(dataset))]
        wins = sum(1 for v in values if v > 0)
        losses = sum(1 for v in values if v < 0)
        draws = sum(1 for v in values if v == 0)

        print(f"Total samples: {len(values)}")
        print(f"  Wins (+1):   {wins:6d} ({wins/len(values)*100:5.1f}%)")
        print(f"  Losses (-1): {losses:6d} ({losses/len(values)*100:5.1f}%)")
        print(f"  Draws (0):   {draws:6d} ({draws/len(values)*100:5.1f}%)")

        # Data type check
        print("\n" + "-" * 60)
        print("Data Type Verification")
        print("-" * 60)
        state_tensor, value = dataset[0]
        print(f"State tensor dtype: {state_tensor.dtype}")
        print(f"State tensor device: {state_tensor.device}")
        print(f"Value dtype: {value.dtype}")
        print(f"Value shape: {value.shape}")
        print(f"Value is scalar: {value.numel() == 1}")

        print("\n" + "=" * 60)
        print("Dataset is ready for training!")
        print("=" * 60)

    except FileNotFoundError:
        print("\nDataset not found. Run generate_value_dataset.py first.")
        return

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
