#!/usr/bin/env python3
"""
Entry point script for value network training.

This script trains the ValueNetworkCNN model on self-play generated games
to predict game outcomes.

Usage:
    PYTHONPATH=./src python test/train_value_network.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python test/train_value_network.py

Monitor training progress with TensorBoard:
    tensorboard --logdir runs/value_network
"""

from checkers.ai.train_value_network import train
from checkers.ai.value_network_config import ValueNetworkConfig


def main():
    """Main training entry point."""
    # Create training configuration
    config = ValueNetworkConfig(
        # Data
        dataset_cache_path='data/value_network_dataset.pkl',
        val_split=0.2,

        # Training hyperparameters
        batch_size=128,      # Increased from 64 - better gradient estimates at higher LR
        num_epochs=100,
        learning_rate=5e-3,  # Increased from 1e-3 - model was learning too slowly
        weight_decay=1e-5,   # Reduced from 1e-4 - less regularization for better learning

        # Optimization
        max_grad_norm=1.0,
        early_stopping_patience=15,  # More patient - regression can be noisy
        lr_scheduler_patience=3,     # Reduce LR faster when plateaued
        lr_scheduler_factor=0.5,
        min_lr=1e-6,

        # Hardware
        # device will auto-detect MPS (M1/M2 Mac) or fallback to CPU
        num_workers=2,

        # Paths
        checkpoint_dir='checkpoints/value_network',
        log_dir='runs/value_network',

        # Logging
        log_interval=10  # Log every 10 batches
    )

    # Print configuration
    print("=" * 70)
    print(" " * 20 + "VALUE NETWORK TRAINING")
    print("=" * 70)
    print(f"Dataset: {config.dataset_cache_path}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Device: {config.device}")
    print(f"Validation Split: {config.val_split}")
    print(f"Checkpoint Dir: {config.checkpoint_dir}")
    print(f"Log Dir: {config.log_dir}")
    print("=" * 70)

    # Start training
    train(config)


if __name__ == '__main__':
    main()
