"""
Test that self-play happens on first iteration when no checkpoints exist.
"""

import os
import shutil
import tempfile
import torch
from checkers.ai.checkpoint_manager import CheckpointManager
from checkers.ai.sl_action_policy import create_model

def test_forces_self_play_on_first_iteration():
    """
    Verify that on iteration 0, the opponent pool is empty,
    which forces the training code to use self-play.
    """

    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()

    try:
        # Create checkpoint manager with empty directory
        ckpt_manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            keep_last_n=5,
            milestone_every=50  # Use 50 so iteration 0 won't be a milestone
        )

        print("=" * 60)
        print("Testing opponent pool on first iteration")
        print("=" * 60)

        # Simulate iteration 0 (before any checkpoints are saved)
        opponent_pool = ckpt_manager.get_opponent_pool(
            current_iteration=0,
            pool_size=5
        )

        print(f"\nIteration 0:")
        print(f"  Opponent pool: {opponent_pool}")
        print(f"  Pool size: {len(opponent_pool)}")

        # KEY ASSERTION: Pool should be empty on first iteration
        assert opponent_pool == [], \
            "Opponent pool MUST be empty on iteration 0"

        print("\n✓ Opponent pool is empty on iteration 0")
        print("✓ Training will use self-play (model vs itself)")

        # Now save a checkpoint for iteration 0
        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        stats = {'win_rate': 0.5, 'loss': 0.1, 'temperature': 1.0}
        ckpt_manager.save_checkpoint(model, optimizer, 0, stats)

        print("\n" + "-" * 60)
        print("Checkpoint saved for iteration 0")
        print("-" * 60)

        # Simulate iteration 1 (after iteration 0 checkpoint exists)
        opponent_pool = ckpt_manager.get_opponent_pool(
            current_iteration=1,
            pool_size=5
        )

        print(f"\nIteration 1:")
        print(f"  Opponent pool: {[os.path.basename(cp) for cp in opponent_pool]}")
        print(f"  Pool size: {len(opponent_pool)}")

        # Now pool should have the iteration 0 checkpoint
        assert len(opponent_pool) > 0, \
            "Opponent pool should have checkpoints after iteration 0"

        print("\n✓ Opponent pool now contains previous checkpoint")
        print("✓ Training can play against previous version")

        print("\n" + "=" * 60)
        print("Summary:")
        print("  - Iteration 0: Empty pool → forced self-play")
        print("  - Iteration 1+: Pool has checkpoints → can play vs old versions")
        print("=" * 60)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_forces_self_play_on_first_iteration()
    print("\n✅ All tests passed!")
