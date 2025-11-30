#!/usr/bin/env python3
"""
CLI checkers game to play against the trained RL model.

Usage:
    PYTHONPATH=./src python3 test/play_vs_ai.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/play_vs_ai.py
"""

import sys
import torch
from checkers.api.environment import (
    reset,
    step,
    legal_moves,
    _state_to_board,
    _board_to_state
)
from checkers.ai.sl_action_policy import create_model
from checkers.ai.cnn_player import select_cnn_move
from checkers.cli.interface import (
    display_board,
    parse_coordinate,
    coords_to_position,
    position_to_coords
)


def load_cnn_model(checkpoint_path='checkpoints/rl/iter_1990.pth', device='cpu'):
    """
    Load trained RL CNN model from checkpoint.

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


def format_move(action):
    """
    Format an action tuple as coordinate notation.

    Args:
        action: (source_idx, dest_idx) tuple

    Returns:
        String like "a3 b4"
    """
    from_pos, to_pos = action
    from_r, from_c = position_to_coords(from_pos)
    to_r, to_c = position_to_coords(to_pos)
    from_coord = chr(ord('a') + from_c) + str(from_r + 1)
    to_coord = chr(ord('a') + to_c) + str(to_r + 1)
    return f"{from_coord} {to_coord}"


def main():
    print("Checkers: Play vs AI")
    print("=" * 40)

    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load AI model
    print("Loading AI model from checkpoints/rl/iter_1990.pth...")
    try:
        model = load_cnn_model(device=device)
        print("AI model loaded successfully!\n")
    except FileNotFoundError:
        print("Error: Could not find checkpoints/rl/iter_1990.pth")
        print("Please ensure the checkpoint file exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize game
    state = reset()
    human_color = "W"  # Human plays White
    ai_color = "B"     # AI plays Black

    print("You are playing as White (W)")
    print("AI is playing as Black (B)")
    print("\nControls:")
    print("  - Enter moves as: 'a3 b4'")
    print("  - Type 'quit' to exit")
    print("  - Type 'show' to redisplay the board\n")

    while True:
        # Display current board
        board = _state_to_board(state)
        display_board(board)

        # Check for game over
        moves = legal_moves(state)
        if not moves:
            winner = "Black" if state['current_turn'] == "W" else "White"
            print(f"\nGame Over! {winner} wins!")
            break

        current_turn = state['current_turn']

        # Human's turn (White)
        if current_turn == human_color:
            print("Your turn (White)")

            while True:
                try:
                    move_input = input("Enter move: ").strip().lower()

                    if move_input == "quit":
                        print("Thanks for playing!")
                        sys.exit(0)

                    if move_input == "show":
                        board = _state_to_board(state)
                        display_board(board)
                        continue

                    # Parse move
                    parts = move_input.split()
                    if len(parts) != 2:
                        print("Invalid format. Use: 'a3 b4'")
                        continue

                    from_coords = parse_coordinate(parts[0])
                    to_coords = parse_coordinate(parts[1])

                    if not from_coords or not to_coords:
                        print("Invalid coordinates. Use letters a-h and numbers 1-8.")
                        continue

                    from_position = coords_to_position(*from_coords)
                    to_position = coords_to_position(*to_coords)

                    if from_position is None or to_position is None:
                        print("Invalid square. Only dark squares are valid.")
                        continue

                    # Check if move is legal
                    action = (from_position, to_position)
                    if action not in moves:
                        print(f"Illegal move. Legal moves:")
                        for m in moves:
                            print(f"  {format_move(m)}")
                        continue

                    # Execute move
                    state, reward, done, info = step(state, action)

                    if done:
                        board = _state_to_board(state)
                        display_board(board)
                        if reward > 0:
                            print("\nYou win!")
                        elif reward < 0:
                            print("\nAI wins!")
                        else:
                            print("\nDraw!")
                        return

                    break

                except KeyboardInterrupt:
                    print("\n\nThanks for playing!")
                    sys.exit(0)
                except Exception as e:
                    print(f"Error: {e}")
                    continue

        # AI's turn (Black)
        else:
            print("AI's turn (Black)...")

            try:
                # AI selects move
                ai_action = select_cnn_move(model, state, device)
                print(f"AI plays: {format_move(ai_action)}")

                # Execute AI's move
                state, reward, done, info = step(state, ai_action)

                if done:
                    board = _state_to_board(state)
                    display_board(board)
                    if reward > 0:
                        print("\nAI wins!")
                    elif reward < 0:
                        print("\nYou win!")
                    else:
                        print("\nDraw!")
                    return

            except Exception as e:
                print(f"AI error: {e}")
                print("AI forfeits!")
                return


if __name__ == "__main__":
    main()
