#!/usr/bin/env python3
"""
Test script to evaluate value network outputs on random inputs.

Loads the best value network model and runs 100 random inputs through it
to visualize the output distribution.

Usage:
    PYTHONPATH=./src python3 test/test_value_network_outputs.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/test_value_network_outputs.py
"""

import torch
import numpy as np
from checkers.ai.value_network import create_model


def main():
    print("=" * 70)
    print(" " * 20 + "VALUE NETWORK OUTPUT TEST")
    print("=" * 70)

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load best model
    model_path = "checkpoints/value_network/best_model.pth"
    print(f"Loading model from: {model_path}")

    model = create_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    # Generate 100 random inputs
    print(f"\nGenerating 100 random inputs with shape (4, 8, 4)...")
    num_samples = 100

    # Random inputs: uniform distribution between 0 and 1
    # (Valid board states would be binary 0/1, but let's see how model handles random)
    random_inputs = torch.rand(num_samples, 4, 8, 4).to(device)

    # Run through model
    print(f"Running inference...")
    with torch.no_grad():
        outputs = model(random_inputs)  # Shape: (100, 1)

    # Convert to numpy for analysis
    outputs_np = outputs.cpu().numpy().flatten()

    # Display statistics
    print("\n" + "=" * 70)
    print(" " * 25 + "OUTPUT STATISTICS")
    print("=" * 70)
    print(f"Number of samples: {num_samples}")
    print(f"Output range: [{outputs_np.min():.4f}, {outputs_np.max():.4f}]")
    print(f"Mean: {outputs_np.mean():.4f}")
    print(f"Std Dev: {outputs_np.std():.4f}")
    print(f"Median: {np.median(outputs_np):.4f}")

    # Binned distribution
    print("\nDistribution by bins:")
    print("-" * 70)
    bins = [
        (-1.0, -0.5, "Strong Black advantage"),
        (-0.5, -0.1, "Slight Black advantage"),
        (-0.1, 0.1, "Draw/Neutral"),
        (0.1, 0.5, "Slight White advantage"),
        (0.5, 1.0, "Strong White advantage")
    ]

    for low, high, label in bins:
        count = np.sum((outputs_np >= low) & (outputs_np < high))
        percentage = 100 * count / num_samples
        bar = "█" * int(percentage / 2)
        print(f"[{low:+.1f}, {high:+.1f}): {count:3d} ({percentage:5.1f}%) {bar:<25} {label}")

    # Show first 20 outputs
    print("\n" + "=" * 70)
    print("First 20 outputs:")
    print("-" * 70)
    for i in range(min(20, num_samples)):
        value = outputs_np[i]
        bar_length = int(abs(value) * 20)
        if value >= 0:
            bar = " " * 20 + "█" * bar_length
            direction = "White"
        else:
            bar = "█" * bar_length + " " * 20
            direction = "Black"
        print(f"Sample {i+1:3d}: {value:+.4f} [{bar}] ({direction})")

    print("\n" + "=" * 70)
    print("Note: Random inputs (not valid board states)")
    print("For meaningful evaluation, use actual game positions")
    print("=" * 70)


if __name__ == '__main__':
    main()
