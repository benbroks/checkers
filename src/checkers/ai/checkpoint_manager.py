"""
Checkpoint manager for RL training.

This module provides functionality to save, load, and manage checkpoints
with a retention policy (keep last N + milestones).
"""

import os
import glob
import torch


class CheckpointManager:
    """
    Manages RL checkpoints with retention policy.

    Keeps the last N checkpoints for the opponent pool, and saves
    milestone checkpoints at specified intervals that are never deleted.
    """

    def __init__(self, checkpoint_dir, keep_last_n=10, milestone_every=50):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            milestone_every: Save milestone every N iterations
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.milestone_every = milestone_every
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, iteration, stats):
        """
        Save checkpoint and manage cleanup.

        Args:
            model: CheckersCNN model
            optimizer: Optimizer
            iteration: Current iteration number
            stats: Dict with win_rate, avg_loss, temperature, etc.
        """
        # Create checkpoint dict
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats
        }

        # Save regular checkpoint
        filepath = os.path.join(self.checkpoint_dir, f'iter_{iteration}.pth')
        torch.save(checkpoint, filepath)

        # Save milestone checkpoint (never deleted)
        if iteration % self.milestone_every == 0:
            milestone_path = os.path.join(self.checkpoint_dir, f'milestone_{iteration}.pth')
            torch.save(checkpoint, milestone_path)
            print(f"  Milestone checkpoint saved: {milestone_path}")

        # Cleanup old checkpoints (keep last N, preserve milestones)
        self._cleanup_old_checkpoints(iteration)

    def _cleanup_old_checkpoints(self, current_iteration):
        """
        Delete old non-milestone checkpoints beyond keep_last_n.

        Args:
            current_iteration: Current iteration number
        """
        # Get all regular checkpoints (not milestones)
        pattern = os.path.join(self.checkpoint_dir, 'iter_*.pth')
        checkpoints = glob.glob(pattern)

        # Sort by iteration number
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        # Keep only last N
        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[:-self.keep_last_n]
            for path in to_delete:
                try:
                    os.remove(path)
                except OSError:
                    pass  # Ignore if already deleted

    def load_checkpoint(self, iteration=None, device='cpu'):
        """
        Load checkpoint by iteration (or latest if None).

        Args:
            iteration: Specific iteration to load (None = latest)
            device: Device to load checkpoint to

        Returns:
            dict: Checkpoint dictionary, or None if no checkpoints exist
        """
        if iteration is None:
            # Load latest checkpoint
            pattern = os.path.join(self.checkpoint_dir, 'iter_*.pth')
            checkpoints = glob.glob(pattern)

            if not checkpoints:
                return None

            # Sort by iteration number and get latest
            checkpoints = sorted(
                checkpoints,
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            filepath = checkpoints[-1]
        else:
            # Load specific iteration
            filepath = os.path.join(self.checkpoint_dir, f'iter_{iteration}.pth')

            if not os.path.exists(filepath):
                return None

        return torch.load(filepath, map_location=device)

    def get_opponent_pool(self, current_iteration, pool_size=5, include_milestones=True):
        """
        Get list of checkpoint paths for opponent pool.

        Returns the most recent pool_size checkpoints (excluding current iteration),
        optionally including milestone checkpoints.

        Returns empty list if no previous checkpoints exist (caller should use self-play).
        """
        # Get recent checkpoints
        pattern = os.path.join(self.checkpoint_dir, 'iter_*.pth')
        checkpoints = glob.glob(pattern)
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        available = [
            cp for cp in checkpoints
            if int(cp.split('_')[-1].split('.')[0]) < current_iteration
        ]

        recent = available[-pool_size:] if available else []

        if include_milestones:
            # Also add milestone checkpoints (only from previous iterations)
            milestone_pattern = os.path.join(self.checkpoint_dir, 'milestone_*.pth')
            milestones = glob.glob(milestone_pattern)
            milestones = sorted(milestones, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            # Filter milestones to only include those from previous iterations
            milestones = [
                m for m in milestones
                if int(m.split('_')[-1].split('.')[0]) < current_iteration
            ]

            # Combine recent + milestones
            return recent + milestones

        return recent

    def list_checkpoints(self):
        """
        List all checkpoints (regular and milestones).

        Returns:
            dict: {'regular': [...], 'milestones': [...]}
        """
        regular_pattern = os.path.join(self.checkpoint_dir, 'iter_*.pth')
        milestone_pattern = os.path.join(self.checkpoint_dir, 'milestone_*.pth')

        regular = sorted(
            glob.glob(regular_pattern),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        milestones = sorted(
            glob.glob(milestone_pattern),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        return {'regular': regular, 'milestones': milestones}
