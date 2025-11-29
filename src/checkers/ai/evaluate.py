"""
Evaluation functions for RL training.

This module provides functionality to evaluate the trained model against
a random baseline to measure absolute performance improvement.
"""

import random
from checkers.api.environment import reset, legal_moves, step
from checkers.ai.cnn_player import single_turn_cnn_player


def play_vs_random(model, model_as_white, device='cpu'):
    """
    Play one game: model (greedy argmax) vs random agent.

    Args:
        model: CNN model (uses greedy argmax selection)
        model_as_white: If True, model plays White
        device: torch device ('cpu', 'mps', 'cuda')

    Returns:
        int: +1 if model won, -1 if random won, 0 if draw
    """
    state = reset()
    model_color = 'W' if model_as_white else 'B'

    while True:
        if state['current_turn'] == model_color:
            # Model's turn (greedy argmax selection)
            next_state, reward, done, info = single_turn_cnn_player(
                model, state, device
            )
            if done:
                return reward  # Model's perspective
            state = next_state
        else:
            # Random player's turn
            potential_actions = legal_moves(state)
            if not potential_actions:
                # No legal moves - model wins by default
                return 1

            random_action = random.choice(potential_actions)
            next_state, reward, done, info = step(state, random_action)
            if done:
                return -reward  # Flip for model's perspective
            state = next_state


def evaluate_vs_random(model, num_games=50, device='cpu'):
    """
    Evaluate model against random baseline.

    Plays multiple games (half as White, half as Black) and returns
    aggregate statistics. Uses greedy argmax selection (no exploration).

    Args:
        model: CNN model to evaluate
        num_games: Number of games to play (default: 50)
        device: torch device ('cpu', 'mps', 'cuda')

    Returns:
        dict: Evaluation statistics
            - 'wins': Number of games won
            - 'losses': Number of games lost
            - 'draws': Number of draws
            - 'win_rate': Win rate (wins / total_games)
            - 'total_games': Total games played
    """
    wins = 0
    losses = 0
    draws = 0

    # Play half as White, half as Black
    games_as_white = num_games // 2
    games_as_black = num_games - games_as_white

    print(f"  Evaluating: {games_as_white} games as White, {games_as_black} as Black")

    # Play as White
    for _ in range(games_as_white):
        result = play_vs_random(model, model_as_white=True, device=device)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    # Play as Black
    for _ in range(games_as_black):
        result = play_vs_random(model, model_as_white=False, device=device)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    win_rate = wins / num_games if num_games > 0 else 0.0

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'total_games': num_games
    }
