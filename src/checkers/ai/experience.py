"""
Experience buffer for reinforcement learning.

This module provides data structures for collecting and managing experiences
from self-play games for training with REINFORCE.
"""

from dataclasses import dataclass
import torch


@dataclass
class Experience:
    """Single experience tuple from self-play."""
    state_tensor: torch.Tensor  # (4, 8, 4) CNN input
    action: tuple[int, int]     # (source, dest)
    log_prob: float             # Log probability of action
    reward: float               # +1 win, -1 loss, 0 draw
    player: str                 # 'W' or 'B' - which player made this move
    state_dict: dict = None     # Original state dict for getting legal moves


class ExperienceBuffer:
    """
    Buffer for collecting and managing self-play experiences.

    This buffer stores experiences during self-play games and provides
    functionality to assign rewards and convert to training tensors.
    """

    def __init__(self):
        """Initialize empty experience buffer."""
        self.experiences = []

    def add(self, state_tensor, action, log_prob, player, state_dict=None):
        """
        Add experience to buffer (reward assigned later after game ends).

        Args:
            state_tensor: CNN input tensor (4, 8, 4)
            action: Action tuple (source_idx, dest_idx)
            log_prob: Log probability of action under current policy
            player: 'W' or 'B' - which player made this move
            state_dict: Optional original state dict for getting legal moves during training
        """
        exp = Experience(
            state_tensor=state_tensor,
            action=action,
            log_prob=log_prob,
            reward=0.0,  # Assigned later
            player=player,
            state_dict=state_dict
        )
        self.experiences.append(exp)

    def assign_rewards(self, game_result, current_model_color):
        """
        Assign rewards to experiences based on game outcome.

        Only assigns rewards to experiences from the current model's moves.
        Opponent moves keep reward=0.0 (they won't be trained on).

        Args:
            game_result: +1 if current_model_color won, -1 if lost, 0 if draw
            current_model_color: 'W' or 'B' - color of the current model
        """
        for exp in self.experiences:
            if exp.player == current_model_color:
                exp.reward = game_result

    def get_current_model_experiences(self, current_model_color):
        """
        Return only experiences from current model (not opponent).

        Args:
            current_model_color: 'W' or 'B'

        Returns:
            list: Experiences where player == current_model_color
        """
        return [exp for exp in self.experiences if exp.player == current_model_color]

    def clear(self):
        """Clear all experiences from buffer."""
        self.experiences = []

    def to_tensors(self, device='cpu'):
        """
        Convert experiences to training tensors.

        Returns:
            tuple: (states, actions, log_probs, rewards)
                - states: (N, 4, 8, 4) tensor
                - actions: list of N action tuples
                - log_probs: (N,) tensor
                - rewards: (N,) tensor
        """
        if not self.experiences:
            # Return empty tensors
            return (
                torch.empty(0, 4, 8, 4, device=device),
                [],
                torch.empty(0, device=device),
                torch.empty(0, device=device)
            )

        states = torch.stack([exp.state_tensor for exp in self.experiences]).to(device)
        actions = [exp.action for exp in self.experiences]
        log_probs = torch.tensor(
            [exp.log_prob for exp in self.experiences],
            dtype=torch.float32,
            device=device
        )
        rewards = torch.tensor(
            [exp.reward for exp in self.experiences],
            dtype=torch.float32,
            device=device
        )
        return states, actions, log_probs, rewards

    def __len__(self):
        """Return number of experiences in buffer."""
        return len(self.experiences)

    def __repr__(self):
        """String representation of buffer."""
        return f"ExperienceBuffer({len(self)} experiences)"
