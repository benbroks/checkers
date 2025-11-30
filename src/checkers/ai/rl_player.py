"""
RL Player with softmax sampling.

This module provides functions to select moves using softmax sampling
over legal moves (instead of argmax). This enables exploration during
reinforcement learning training.

PLACEHOLDER FUNCTIONS:
- select_cnn_move_softmax() - USER IMPLEMENTS THIS
"""

import torch
import torch.nn.functional as F
import numpy as np
from checkers.core.state_utils import state_to_cnn_input, action_to_cnn_output
from checkers.api.environment import legal_moves, step
from checkers.ai.sl_action_policy import CheckersCNN


def select_cnn_move_softmax(
    model: CheckersCNN,
    state,
    temperature=1.0,
    device='cpu'
):
    """
    Select a move by sampling from softmax over legal moves.

    Args:
        model: CheckersCNN model
        state: Game state dict
        temperature: Softmax temperature (higher = more random, lower = more greedy)
        device: torch device ('cpu', 'mps', 'cuda')

    Returns:
        tuple: (action, log_prob)
            - action: (source_idx, dest_idx) tuple
            - log_prob: Log probability of the selected action

    Raises:
        ValueError: If no legal moves are available
        NotImplementedError: This is a placeholder - user must implement
    """
    legal_actions = legal_moves(state)
    cnn_input = state_to_cnn_input(state)
    tensor_input = torch.from_numpy(cnn_input).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor_input).squeeze(0)  # (32, 8)

    # Extract logits for each legal move from the output tensor
    legal_logits = []
    for action in legal_actions:
        action_tensor = action_to_cnn_output(action)
        row, col = np.where(action_tensor == 1)
        logit = output[int(row[0]), int(col[0])].item()
        legal_logits.append(logit)

    if len(legal_actions) == 0:
        raise ValueError("no legal actions")
    
    logits_tensor = torch.tensor(legal_logits) / temperature
    probs = F.softmax(logits_tensor, dim=0)

    sampled_idx = torch.multinomial(probs, 1).item()
    selected_action = legal_actions[sampled_idx]

    log_probs = F.log_softmax(logits_tensor, dim=0)
    log_prob = log_probs[sampled_idx].item()

    return selected_action, log_prob


def single_turn_rl_player(model, state, temperature, device='cpu'):
    """
    Execute one turn using softmax sampling (calls select_cnn_move_softmax).

    This is a convenience wrapper that selects a move with softmax sampling
    and executes it in the environment.

    Args:
        model: CheckersCNN model
        state: Current game state dict
        temperature: Softmax temperature
        device: torch device

    Returns:
        tuple: (next_state, reward, done, info, action, log_prob)
            - next_state: New state after move
            - reward: Reward from environment (+1 win, 0 otherwise)
            - done: Whether game ended
            - info: Additional info dict
            - action: The action that was taken
            - log_prob: Log probability of the action
    """
    # Select move using softmax sampling
    action, log_prob = select_cnn_move_softmax(
        model,
        state,
        temperature,
        device
    )

    # Execute the move in the environment
    next_state, reward, done, info = step(state, action)

    return next_state, reward, done, info, action, log_prob
