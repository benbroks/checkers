"""
Test script for CNN action policy vs random player.

Usage:
    PYTHONPATH=./src python test/play_cnn_policy.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/play_cnn_policy.py
"""

import torch
import random
from checkers.api.environment import reset, legal_moves, step
from checkers.ai.sl_action_policy import create_model
from checkers.ai.cnn_player import single_turn_cnn_player


def load_cnn_model(checkpoint_path='checkpoints/best_model.pth', device='cpu'):
    """
    Load trained CNN model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        device: torch device ('cpu', 'mps', 'cuda')

    Returns:
        Loaded and initialized model in eval mode
    """
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def cnn_vs_random(model, device='cpu', cnn_color='W'):
    """
    Play one game: CNN vs Random player.

    Args:
        model: Trained CNN model
        device: torch device ('cpu', 'mps', 'cuda')
        cnn_color: Color for CNN player ('W' or 'B')

    Returns:
        reward: 1 if CNN wins, -1 if random wins, 0 if draw
    """
    state = reset()

    while True:
        if state['current_turn'] == cnn_color:
            # CNN player's turn
            next_state, reward, done, info = single_turn_cnn_player(
                model, state, device
            )
            if done:
                return reward
            state = next_state
        else:
            # Random player's turn
            potential_actions = legal_moves(state)
            if not potential_actions:
                # No legal moves - CNN wins by default
                return 1

            random_action = random.choice(potential_actions)
            next_state, reward, done, info = step(state, random_action)
            if done:
                return -1 * reward  # Flip reward for opponent
            state = next_state


def main():
    """Run CNN vs Random for multiple games and report statistics."""
    # Setup device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading CNN model from checkpoints/best_model.pth...")
    try:
        model = load_cnn_model(device=device)
        print("Model loaded successfully\n")
    except FileNotFoundError:
        print("Error: Could not find checkpoints/best_model.pth")
        print("Please ensure you have trained a model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run games
    num_games = 100
    aggregate_results = {"wins": 0, "losses": 0, "ties": 0}

    print(f"Playing {num_games} games (CNN as White vs Random)...")
    print("-" * 50)

    for i in range(num_games):
        result = cnn_vs_random(model, device=device, cnn_color='W')

        if result == 1:
            aggregate_results["wins"] += 1
        elif result == -1:
            aggregate_results["losses"] += 1
        else:
            aggregate_results["ties"] += 1

        # Progress updates
        if (i + 1) % 10 == 0:
            win_rate = aggregate_results["wins"] / (i + 1) * 100
            print(f"  Completed {i + 1}/{num_games} games - Win rate: {win_rate:.1f}%")

    # Print final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Wins:   {aggregate_results['wins']:3d} ({aggregate_results['wins']/num_games*100:.1f}%)")
    print(f"Losses: {aggregate_results['losses']:3d} ({aggregate_results['losses']/num_games*100:.1f}%)")
    print(f"Draws:  {aggregate_results['ties']:3d} ({aggregate_results['ties']/num_games*100:.1f}%)")
    print("=" * 50)


if __name__ == '__main__':
    main()
