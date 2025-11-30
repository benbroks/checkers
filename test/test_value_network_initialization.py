#!/usr/bin/env python3
"""
Test script to verify value network initialization is working correctly.

This script creates a fresh model and tests it on various inputs to ensure:
1. Outputs are not saturated (not all -1 or all +1)
2. Outputs have reasonable variance
3. The network responds differently to different inputs

Usage:
    PYTHONPATH=./src python3 test/test_value_network_initialization.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/test_value_network_initialization.py
"""

import torch
import numpy as np
from checkers.ai.value_network import create_model


def main():
    print("=" * 70)
    print(" " * 15 + "VALUE NETWORK INITIALIZATION TEST")
    print("=" * 70)

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create a fresh model with new initialization
    print("\nCreating fresh model with new initialization...")
    model = create_model().to(device)
    model.eval()

    print("✓ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test 1: Random inputs
    print("\n" + "=" * 70)
    print("Test 1: Random continuous inputs (uniform [0, 1])")
    print("-" * 70)

    num_samples = 100
    random_inputs = torch.rand(num_samples, 4, 8, 4).to(device)

    with torch.no_grad():
        outputs = model(random_inputs).cpu().numpy().flatten()

    print(f"Number of samples: {num_samples}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"Mean: {outputs.mean():.4f}")
    print(f"Std Dev: {outputs.std():.4f}")
    print(f"Median: {np.median(outputs):.4f}")

    # Check for saturation
    saturated_low = np.sum(outputs < -0.99)
    saturated_high = np.sum(outputs > 0.99)
    print(f"\nSaturation check:")
    print(f"  Values < -0.99: {saturated_low} ({100*saturated_low/num_samples:.1f}%)")
    print(f"  Values > +0.99: {saturated_high} ({100*saturated_high/num_samples:.1f}%)")

    if saturated_low + saturated_high > 50:
        print("  ⚠️  WARNING: More than 50% of outputs are saturated!")
    else:
        print("  ✓ Good! Outputs are not saturated")

    # Test 2: Binary inputs (valid board states)
    print("\n" + "=" * 70)
    print("Test 2: Binary inputs (simulating valid board states)")
    print("-" * 70)

    binary_inputs = (torch.rand(num_samples, 4, 8, 4) > 0.5).float().to(device)

    with torch.no_grad():
        binary_outputs = model(binary_inputs).cpu().numpy().flatten()

    print(f"Number of samples: {num_samples}")
    print(f"Output range: [{binary_outputs.min():.4f}, {binary_outputs.max():.4f}]")
    print(f"Mean: {binary_outputs.mean():.4f}")
    print(f"Std Dev: {binary_outputs.std():.4f}")

    # Test 3: Same input should give same output
    print("\n" + "=" * 70)
    print("Test 3: Determinism check (same input → same output)")
    print("-" * 70)

    test_input = torch.rand(1, 4, 8, 4).to(device)

    with torch.no_grad():
        output1 = model(test_input).item()
        output2 = model(test_input).item()

    print(f"Output 1: {output1:.6f}")
    print(f"Output 2: {output2:.6f}")
    print(f"Difference: {abs(output1 - output2):.10f}")

    if abs(output1 - output2) < 1e-6:
        print("✓ Outputs are deterministic")
    else:
        print("⚠️  WARNING: Outputs differ for same input!")

    # Test 4: Different inputs should give different outputs
    print("\n" + "=" * 70)
    print("Test 4: Variance check (different inputs → different outputs)")
    print("-" * 70)

    test_inputs = torch.rand(10, 4, 8, 4).to(device)

    with torch.no_grad():
        test_outputs = model(test_inputs).cpu().numpy().flatten()

    print("Outputs for 10 different random inputs:")
    for i, val in enumerate(test_outputs):
        print(f"  Input {i+1}: {val:+.4f}")

    variance = np.var(test_outputs)
    print(f"\nVariance: {variance:.6f}")

    if variance < 0.001:
        print("⚠️  WARNING: Very low variance - model might not be sensitive to inputs")
    else:
        print("✓ Good variance - model responds to different inputs")

    # Test 5: Distribution visualization
    print("\n" + "=" * 70)
    print("Test 5: Output distribution (100 random samples)")
    print("-" * 70)

    bins = [
        (-1.0, -0.5, "Strong Black advantage"),
        (-0.5, -0.1, "Slight Black advantage"),
        (-0.1, 0.1, "Draw/Neutral"),
        (0.1, 0.5, "Slight White advantage"),
        (0.5, 1.0, "Strong White advantage")
    ]

    for low, high, label in bins:
        count = np.sum((outputs >= low) & (outputs < high))
        percentage = 100 * count / num_samples
        bar = "█" * int(percentage / 2)
        print(f"[{low:+.1f}, {high:+.1f}): {count:3d} ({percentage:5.1f}%) {bar:<25} {label}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_good = True

    # Check 1: Not all saturated
    if saturated_low + saturated_high < 50:
        print("✓ Initialization prevents saturation")
    else:
        print("✗ Initialization still causes saturation")
        all_good = False

    # Check 2: Reasonable standard deviation
    if outputs.std() > 0.05:
        print("✓ Output has good variance")
    else:
        print("✗ Output variance is too low")
        all_good = False

    # Check 3: Mean is reasonable
    if abs(outputs.mean()) < 0.5:
        print("✓ Mean output is reasonable")
    else:
        print("⚠️  Mean is somewhat biased")

    if all_good:
        print("\n🎉 All checks passed! Network initialization looks good.")
        print("   Ready to start training!")
    else:
        print("\n⚠️  Some checks failed. Review initialization.")

    print("=" * 70)


if __name__ == '__main__':
    main()
