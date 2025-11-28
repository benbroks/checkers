"""
Training configuration for supervised learning.

This module defines the configuration dataclass for training the CheckersCNN model
on supervised learning tasks using PDN game files.
"""

from dataclasses import dataclass, field
import torch


@dataclass
class TrainingConfig:
    """Configuration for supervised learning training."""

    # Data
    pdn_files: list = field(default_factory=list)
    val_split: float = 0.15
    max_games: int = None  # None = load all games

    # Training (CPU-optimized)
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Optimization
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Hardware (Mac CPU)
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
    num_workers: int = 2

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/checkers_cnn"

    # Logging
    log_interval: int = 50  # Log every N batches

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.pdn_files:
            raise ValueError("pdn_files cannot be empty")
        if not 0 < self.val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
