#!/usr/bin/env python3
"""
Generate the full value network dataset with 10,000 self-play games.

Usage:
    PYTHONPATH=./src python3 test/generate_value_dataset.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/generate_value_dataset.py
"""

import torch
from checkers.ai.value_network_dataset import CheckersValueNetworkDataset


def main():
    print("Value Network Dataset Generator")
    print("=" * 60)

    # Detect device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}\n")

    # Generate dataset with 10,000 games
    print("This will generate 10,000 self-play games.")
    print("Expected time: ~30-40 minutes on MPS")
    print()

    try:
        dataset = CheckersValueNetworkDataset(
            max_games=10000,
            checkpoint_path='checkpoints/rl/iter_1990.pth',
            device=device,
            dataset_cache_path='data/value_network_dataset.pkl'
        )

        print("\n" + "=" * 60)
        print("DATASET GENERATION COMPLETE!")
        print("=" * 60)
        print(f"Total samples: {len(dataset)}")
        print(f"Expected: ~20,000 samples (2 per game)")
        print(f"\nDataset saved to: data/value_network_dataset.pkl")

        # Show some statistics
        if len(dataset) > 0:
            print("\nSample statistics:")
            state_tensor, value = dataset[0]
            print(f"  State tensor shape: {state_tensor.shape}")
            print(f"  Value dtype: {value.dtype}")

            # Count value distribution
            values = [dataset[i][1].item() for i in range(min(1000, len(dataset)))]
            wins = sum(1 for v in values if v > 0)
            losses = sum(1 for v in values if v < 0)
            draws = sum(1 for v in values if v == 0)
            print(f"\nValue distribution (first 1000 samples):")
            print(f"  Wins (+1):  {wins} ({wins/len(values)*100:.1f}%)")
            print(f"  Losses (-1): {losses} ({losses/len(values)*100:.1f}%)")
            print(f"  Draws (0):  {draws} ({draws/len(values)*100:.1f}%)")

    except FileNotFoundError:
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
