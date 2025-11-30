"""
Training configuration for value network.

This module defines the configuration dataclass for training the ValueNetworkCNN model
on self-play generated games to predict game outcomes.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class ValueNetworkConfig:
    """Configuration for value network training."""

    # Data
    dataset_cache_path: str = "data/value_network_dataset.pkl"
    val_split: float = 0.2  # 20% validation per user requirement

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Optimization
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Hardware
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
    num_workers: int = 2

    # Paths
    checkpoint_dir: str = "checkpoints/value_network"
    log_dir: str = "runs/value_network"

    # Logging
    log_interval: int = 50  # Log every N batches

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
