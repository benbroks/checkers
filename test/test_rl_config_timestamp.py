"""
Test that RLConfig creates timestamped log directories.
"""

from checkers.ai.rl_config import RLConfig
import time

def test_timestamped_log_dir():
    """Verify that each config instance gets a unique timestamped directory."""

    # Create first config
    config1 = RLConfig()
    print(f"Config 1 log_dir: {config1.log_dir}")

    # Wait a bit to ensure different timestamp
    time.sleep(1.1)

    # Create second config
    config2 = RLConfig()
    print(f"Config 2 log_dir: {config2.log_dir}")

    # Verify they're different
    assert config1.log_dir != config2.log_dir, "Log directories should be unique"

    # Verify format
    assert config1.log_dir.startswith("runs/checkers_rl/run_"), "Should have correct prefix"
    assert len(config1.log_dir.split("_")) >= 3, "Should have timestamp"

    print("\n✓ Each config instance creates a unique timestamped log directory!")

if __name__ == "__main__":
    test_timestamped_log_dir()
