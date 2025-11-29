"""
RL training entry point.

This script configures and runs reinforcement learning training
using self-play with the REINFORCE algorithm.

Usage:
    PYTHONPATH=./src python test/train_reinforcement_learning.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/train_reinforcement_learning.py
"""

from checkers.ai.rl_config import RLConfig
from checkers.ai.train_rl import train


def main():
    """Configure and run RL training."""
    config = RLConfig(
        # Self-play configuration
        games_per_iteration=50,      # 10 as White, 10 as Black
        opponent_pool_size=5,        # Sample from last 5 checkpoints
        self_play_prob=0.1,          # 50% against self, 50% against pool

        # Temperature (exploration vs exploitation)
        temperature_start=1.0,       # High exploration initially
        temperature_end=0.1,         # Lower exploration later
        temperature_decay_iterations=1000,
        temperature_decay_type="linear",  # "linear", "exponential", or "constant"

        # Training hyperparameters
        learning_rate=1e-4,          # Lower than SL (fine-tuning)
        weight_decay=1e-4,
        max_grad_norm=1.0,
        num_iterations=2000,

        # Checkpointing
        checkpoint_every=1,          # Save every iteration
        keep_last_n=10,             # Keep last 10 for opponent pool
        milestone_every=50,         # Save milestone every 50 iterations

        # Paths
        checkpoint_dir="checkpoints/rl",
        log_dir="runs/checkers_rl",
        pretrained_model="checkpoints/best_model.pth",  # Start from SL model

        # Logging
        log_every=1,                # Log every iteration

        # Evaluation
        eval_every=25,              # Evaluate every 25 iterations
        eval_games=50,              # 50 games per evaluation
        eval_enabled=True           # Enable evaluation
    )

    # Run training
    train(config)


if __name__ == '__main__':
    main()
