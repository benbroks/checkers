#!/usr/bin/env python3
"""
Test script for CheckersValueNetworkDataset.

Usage:
    PYTHONPATH=./src python3 test/test_value_network_dataset.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/test_value_network_dataset.py
"""

import torch
from checkers.ai.value_network_dataset import CheckersValueNetworkDataset


def main():
    print("Testing CheckersValueNetworkDataset")
    print("=" * 60)

    # Detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}\n")

    # Create dataset with just 10 games for testing
    try:
        dataset = CheckersValueNetworkDataset(
            max_games=10,
            checkpoint_path='checkpoints/rl/iter_1990.pth',
            device=device,
            dataset_cache_path='data/value_network_dataset_test.pkl'
        )

        print("\n" + "=" * 60)
        print("SUCCESS: Dataset initialized!")
        print("=" * 60)
        print(f"Dataset size: {len(dataset)}")

        # Test getting a sample
        if len(dataset) > 0:
            print("\nTesting sample retrieval:")
            state_tensor, value = dataset[0]
            print(f"  State tensor shape: {state_tensor.shape}")
            print(f"  Value: {value.item()}")
            print(f"  Value range should be in [-1, 0, 1]: ✓" if value.item() in [-1.0, 0.0, 1.0] else "  ERROR: Value out of range!")

    except FileNotFoundError as e:
        print(f"\nERROR: Checkpoint file not found")
        print(f"Please ensure checkpoints/rl/iter_1990.pth exists")
        return

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
