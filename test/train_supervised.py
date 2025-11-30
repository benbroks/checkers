"""
Entry point script for supervised learning training.

This script trains the CheckersCNN model on expert games from PDN files.

Usage:
    PYTHONPATH=./src python test/train_supervised.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python test/train_supervised.py

Monitor training progress with TensorBoard:
    tensorboard --logdir runs/checkers_cnn
"""

from checkers.ai.train_sl import train
from checkers.ai.config import TrainingConfig


def main():
    """Main training entry point."""
    # Create training configuration
    config = TrainingConfig(
        # Data files (update with your PDN files)
        pdn_files=['full.pdn'], 
        max_games=None,  # None = load all games, or set a number for testing

        # Training hyperparameters (CPU-optimized)
        batch_size=128,
        num_epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-4,

        # Optimization
        max_grad_norm=1.0,
        early_stopping_patience=10,
        lr_scheduler_patience=5,
        lr_scheduler_factor=0.5,
        min_lr=1e-6,

        # Data split
        val_split=0.15,

        # Hardware
        # device will auto-detect MPS (M1/M2 Mac) or fallback to CPU
        num_workers=2,

        # Paths
        checkpoint_dir='checkpoints',
        log_dir='runs/checkers_cnn',

        # Logging
        log_interval=50  # Log every 50 batches
    )

    # Print configuration
    print("=" * 70)
    print(" " * 20 + "TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"PDN Files: {config.pdn_files}")
    print(f"Max Games: {config.max_games or 'All'}")
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
