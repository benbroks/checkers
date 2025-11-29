"""
Test that CheckpointManager returns empty opponent pool on first iteration.
"""

import os
import shutil
import tempfile
import torch
from checkers.ai.checkpoint_manager import CheckpointManager
from checkers.ai.sl_action_policy import create_model

def test_empty_pool_on_first_iteration():
    """Verify that opponent pool is empty before any checkpoints are saved."""

    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()

    try:
        # Create checkpoint manager
        ckpt_manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            keep_last_n=5,
            milestone_every=10
        )

        # On iteration 0, there should be no checkpoints yet
        opponent_pool = ckpt_manager.get_opponent_pool(
            current_iteration=0,
            pool_size=5
        )

        print(f"Iteration 0 - Opponent pool: {opponent_pool}")
        assert opponent_pool == [], "Opponent pool should be empty on first iteration"
        print("✓ Opponent pool is empty on iteration 0 (will use self-play)")

        # Now save a checkpoint for iteration 0
        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        stats = {'win_rate': 0.5, 'loss': 0.1, 'temperature': 1.0}

        ckpt_manager.save_checkpoint(model, optimizer, 0, stats)
        print("\n✓ Saved checkpoint for iteration 0")

        # On iteration 1, the pool should now contain iteration 0
        # Note: iteration 0 is also a milestone (0 % 10 == 0), so we get both iter_0 and milestone_0
        opponent_pool = ckpt_manager.get_opponent_pool(
            current_iteration=1,
            pool_size=5
        )

        print(f"Iteration 1 - Opponent pool: {opponent_pool}")
        assert len(opponent_pool) >= 1, "Opponent pool should contain at least 1 checkpoint"
        assert any("iter_0.pth" in cp for cp in opponent_pool), "Pool should contain iteration 0"
        print("✓ Opponent pool contains iteration 0 checkpoint on iteration 1")

        # Save a few more checkpoints
        for i in range(1, 4):
            ckpt_manager.save_checkpoint(model, optimizer, i, stats)

        # On iteration 4, pool should contain iterations 0-3 (+ milestone_0)
        opponent_pool = ckpt_manager.get_opponent_pool(
            current_iteration=4,
            pool_size=5
        )

        print(f"\nIteration 4 - Opponent pool size: {len(opponent_pool)}")
        assert len(opponent_pool) >= 4, "Opponent pool should contain at least 4 checkpoints"
        print("✓ Opponent pool grows with saved checkpoints")

        # Test milestone checkpoint (should be included in pool)
        ckpt_manager.save_checkpoint(model, optimizer, 10, stats)
        print("\n✓ Saved milestone checkpoint for iteration 10")

        opponent_pool = ckpt_manager.get_opponent_pool(
            current_iteration=11,
            pool_size=5
        )

        # Should have recent checkpoints + milestone
        milestone_count = sum(1 for cp in opponent_pool if "milestone" in cp)
        print(f"Iteration 11 - Opponent pool size: {len(opponent_pool)}")
        print(f"Milestone checkpoints in pool: {milestone_count}")
        assert milestone_count >= 1, "Pool should include milestone checkpoint"
        print("✓ Opponent pool includes milestone checkpoints")

        print("\n✓ All tests passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_empty_pool_on_first_iteration()
