"""
Verify that masked softmax assigns zero probability to illegal moves.
"""

import torch
import torch.nn.functional as F
import numpy as np
from checkers.ai.sl_action_policy import create_model
from checkers.api.environment import reset, legal_moves
from checkers.core.state_utils import state_to_cnn_input, action_to_cnn_output

def test_masked_softmax_zeros_illegal():
    """Verify that illegal moves get zero probability under masked softmax."""

    # Setup
    device = 'cpu'
    model = create_model()
    model.eval()

    # Create initial state
    state = reset()
    state_tensor = torch.from_numpy(state_to_cnn_input(state)).float()

    # Get legal moves
    legal_actions = legal_moves(state)
    print(f"Number of legal moves: {len(legal_actions)}")

    # Convert legal actions to flat indices
    legal_indices = []
    for legal_action in legal_actions:
        a_tensor = action_to_cnn_output(legal_action)
        row, col = np.where(a_tensor == 1)
        idx = int(row[0] * 8 + col[0])
        legal_indices.append(idx)

    print(f"Legal action indices: {sorted(legal_indices)}")

    # Get model logits
    with torch.no_grad():
        logits = model(state_tensor.unsqueeze(0))  # (1, 32, 8)
        logits_flat = logits.view(-1)  # (256,)

        # Apply mask
        mask = torch.full_like(logits_flat, float('-inf'))
        mask[legal_indices] = 0.0

        masked_logits = logits_flat + mask
        probs = F.softmax(masked_logits, dim=0)

        # Check that illegal moves have zero probability
        illegal_prob_sum = 0.0
        for i in range(256):
            if i not in legal_indices:
                illegal_prob_sum += probs[i].item()

        print(f"\nSum of probabilities for illegal moves: {illegal_prob_sum:.10f}")
        print(f"Sum of probabilities for legal moves: {probs[legal_indices].sum().item():.10f}")

        # Verify
        assert illegal_prob_sum < 1e-6, f"Illegal moves have non-zero probability: {illegal_prob_sum}"
        assert abs(probs[legal_indices].sum().item() - 1.0) < 1e-6, "Legal move probabilities don't sum to 1"

        print("\n✓ Masked softmax correctly assigns zero probability to illegal moves!")
        print("✓ Legal move probabilities sum to 1.0!")

if __name__ == "__main__":
    test_masked_softmax_zeros_illegal()
