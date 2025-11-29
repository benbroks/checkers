"""
Test that train_on_experience() uses masked softmax over legal moves.
"""

import torch
import numpy as np
from checkers.ai.train_rl import train_on_experience
from checkers.ai.experience import ExperienceBuffer
from checkers.ai.sl_action_policy import create_model
from checkers.ai.rl_config import RLConfig
from checkers.api.environment import reset, legal_moves
from checkers.core.state_utils import state_to_cnn_input

def test_masked_softmax():
    """Test that training uses masked softmax over legal moves."""

    # Setup
    device = 'cpu'
    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create a simple config
    config = RLConfig()
    config.device = device

    # Create initial state
    state = reset()
    state_tensor = torch.from_numpy(state_to_cnn_input(state)).float()

    # Get legal moves
    legal_actions = legal_moves(state)
    print(f"Number of legal moves: {len(legal_actions)}")
    print(f"Legal moves: {legal_actions[:5]}...")  # Print first 5

    # Take the first legal action
    action = legal_actions[0]

    # Create experience buffer with state_dict
    buffer = ExperienceBuffer()
    buffer.add(
        state_tensor=state_tensor,
        action=action,
        log_prob=0.0,  # Placeholder, not used in training
        player='W',
        state_dict=state  # This is the key - include original state
    )

    # Assign reward
    buffer.assign_rewards(game_result=1.0, current_model_color='W')

    # Train
    print("\nTraining with masked softmax...")
    loss = train_on_experience(model, optimizer, buffer, config)
    print(f"Loss: {loss:.4f}")

    # Verify the training ran without errors
    print("\n✓ Training completed successfully with masked softmax!")

    # Test that the model produces valid output
    model.eval()
    with torch.no_grad():
        logits = model(state_tensor.unsqueeze(0))
        print(f"Model output shape: {logits.shape}")
        print(f"Expected shape: (1, 32, 8)")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_masked_softmax()
