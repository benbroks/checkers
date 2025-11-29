"""
Self-play game orchestration for RL training.

This module provides functionality to play self-play games between
the current model and opponent models from the checkpoint pool.
"""

import random
import torch
from checkers.api.environment import reset
from checkers.core.state_utils import state_to_cnn_input
from checkers.ai.experience import ExperienceBuffer
from checkers.ai.rl_player import single_turn_rl_player
from checkers.ai.sl_action_policy import create_model


def play_self_play_game(current_model, opponent_model, current_as_white,
                        temperature, device='cpu'):
    """
    Play one self-play game between current model and opponent.

    Args:
        current_model: Current model being trained
        opponent_model: Opponent model (from pool or self)
        current_as_white: If True, current model plays White
        temperature: Softmax temperature for move sampling
        device: torch device

    Returns:
        tuple: (buffer, game_result)
            - buffer: ExperienceBuffer with all moves from current model
            - game_result: +1 if current won, -1 if lost, 0 if draw
    """
    state = reset()
    buffer = ExperienceBuffer()
    current_color = 'W' if current_as_white else 'B'

    while True:
        if state['current_turn'] == current_color:
            # Current model's turn - record experience
            state_tensor = torch.from_numpy(state_to_cnn_input(state)).float()
            next_state, reward, done, info, action, log_prob = single_turn_rl_player(
                current_model, state, temperature, device
            )
            buffer.add(state_tensor, action, log_prob, current_color, state_dict=state)

            if done:
                # Current model's perspective: reward is already correct
                game_result = reward  # +1 win, 0 draw
                break
            state = next_state
        else:
            # Opponent's turn - don't record experience
            next_state, reward, done, info, _, _ = single_turn_rl_player(
                opponent_model, state, temperature, device
            )

            if done:
                # Opponent won or draw - flip reward for current model's perspective
                game_result = -reward  # -1 loss (or 0 draw)
                break
            state = next_state

    # Assign rewards to all experiences from current model
    buffer.assign_rewards(game_result, current_color)

    return buffer, game_result


def run_self_play_iteration(model, checkpoint_manager, config, iteration):
    """
    Run one iteration of self-play (20 games: 10 as White, 10 as Black).

    Plays multiple games against either self or a randomly selected opponent
    from the checkpoint pool. Collects experiences only from the current model's moves.

    Args:
        model: Current model being trained
        checkpoint_manager: CheckpointManager instance
        config: RLConfig instance
        iteration: Current training iteration number

    Returns:
        tuple: (all_buffer, win_rate, draws)
            - all_buffer: ExperienceBuffer with experiences from all games
            - win_rate: Win rate for current model (wins / total_games)
            - draws: Number of draws
    """
    device = torch.device(config.device)
    temperature = config.get_temperature(iteration)

    # Select opponent: either self or from pool
    opponent_pool = checkpoint_manager.get_opponent_pool(
        iteration,
        config.opponent_pool_size
    )

    if opponent_pool and random.random() > config.self_play_prob:
        # Play against random opponent from pool
        opponent_checkpoint_path = random.choice(opponent_pool)
        opponent_model = load_model_from_checkpoint(opponent_checkpoint_path, device)
        opponent_name = opponent_checkpoint_path.split('/')[-1]
    else:
        # Play against self
        opponent_model = model
        opponent_name = "self"

    # Collect all experiences
    all_buffer = ExperienceBuffer()
    wins = 0
    losses = 0
    draws = 0

    # Play games_per_iteration / 2 games as White
    games_as_white = config.games_per_iteration // 2
    for _ in range(games_as_white):
        buffer, result = play_self_play_game(
            model, opponent_model,
            current_as_white=True,
            temperature=temperature,
            device=device
        )
        all_buffer.experiences.extend(buffer.experiences)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    # Play games_per_iteration / 2 games as Black
    games_as_black = config.games_per_iteration // 2
    for _ in range(games_as_black):
        buffer, result = play_self_play_game(
            model, opponent_model,
            current_as_white=False,
            temperature=temperature,
            device=device
        )
        all_buffer.experiences.extend(buffer.experiences)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    win_rate = wins / config.games_per_iteration

    # Print iteration summary
    print(f"  Self-play vs {opponent_name}:")
    print(f"    Temperature: {temperature:.3f}")
    print(f"    W/L/D: {wins}/{losses}/{draws} (Win rate: {win_rate*100:.1f}%)")
    print(f"    Experiences collected: {len(all_buffer)}")

    return all_buffer, win_rate, draws


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device

    Returns:
        CheckersCNN model loaded with checkpoint weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
