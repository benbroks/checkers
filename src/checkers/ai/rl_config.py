"""
Configuration for reinforcement learning training.

This module defines the configuration dataclass for training the CheckersCNN model
using reinforcement learning with self-play.
"""

from dataclasses import dataclass, field
from datetime import datetime
import torch


@dataclass
class RLConfig:
    """Configuration for reinforcement learning training."""

    # Self-play
    games_per_iteration: int = 20  # 10 as White, 10 as Black
    opponent_pool_size: int = 5    # Sample from last N checkpoints
    self_play_prob: float = 0.5    # 50% play against self, 50% against pool

    # Temperature (exploration vs exploitation)
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay_iterations: int = 1000
    temperature_decay_type: str = "linear"  # "linear", "exponential", "constant"

    # Training
    learning_rate: float = 1e-4  # Lower than SL for fine-tuning
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_iterations: int = 2000

    # Checkpointing
    checkpoint_every: int = 1  # Save every iteration
    keep_last_n: int = 10      # Keep last 10 checkpoints
    milestone_every: int = 50  # Save milestone every 50 iterations

    # Hardware
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")

    # Paths
    checkpoint_dir: str = "checkpoints/rl"
    log_dir: str = field(default_factory=lambda: f"runs/checkers_rl/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    pretrained_model: str = "checkpoints/best_model.pth"  # Start from SL model

    # Logging
    log_every: int = 1  # Log every iteration

    # Evaluation
    eval_every: int = 25           # Evaluate every N iterations
    eval_games: int = 50           # Number of games per evaluation
    eval_enabled: bool = True      # Enable/disable evaluation

    def get_temperature(self, iteration):
        """
        Calculate temperature for current iteration based on decay schedule.

        Args:
            iteration: Current training iteration

        Returns:
            float: Temperature value for softmax sampling
        """
        if self.temperature_decay_type == "constant":
            return self.temperature_start

        # Clamp progress to [0, 1]
        progress = min(iteration / self.temperature_decay_iterations, 1.0)

        if self.temperature_decay_type == "linear":
            # Linear interpolation from start to end
            temp = self.temperature_start + progress * (self.temperature_end - self.temperature_start)
        elif self.temperature_decay_type == "exponential":
            # Exponential decay from start to end
            temp = self.temperature_start * (self.temperature_end / self.temperature_start) ** progress
        else:
            raise ValueError(f"Unknown temperature_decay_type: {self.temperature_decay_type}")

        # Ensure temperature doesn't go below minimum
        return max(temp, self.temperature_end)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.games_per_iteration <= 0:
            raise ValueError("games_per_iteration must be positive")
        if self.games_per_iteration % 2 != 0:
            raise ValueError("games_per_iteration must be even (half as White, half as Black)")
        if not 0 <= self.self_play_prob <= 1:
            raise ValueError("self_play_prob must be between 0 and 1")
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("Temperature values must be positive")
        if self.temperature_end > self.temperature_start:
            raise ValueError("temperature_end must be <= temperature_start")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.temperature_decay_type not in ["linear", "exponential", "constant"]:
            raise ValueError("temperature_decay_type must be 'linear', 'exponential', or 'constant'")
        if self.eval_games <= 0:
            raise ValueError("eval_games must be positive")
        if self.eval_games % 2 != 0:
            raise ValueError("eval_games must be even (half as White, half as Black)")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
