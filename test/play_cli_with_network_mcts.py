#!/usr/bin/env python3
"""
CLI checkers game with Network MCTS (runtime search allocation).

This script provides an interactive CLI interface where a human player
can play against an AI that uses neural network-guided MCTS with runtime
search allocation. The AI runs simulations on-demand for each move,
showing N_s_a values updating in real-time.

Usage:
    PYTHONPATH=./src python3 test/play_cli_with_network_mcts.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python3 test/play_cli_with_network_mcts.py
"""

import sys
import torch
from checkers.core.board import Board
from checkers.api.environment import (
    hash_state,
    reset,
    legal_moves,
    step,
    _board_to_state,
    _state_to_board
)
from checkers.ai.network_mcts import double_network_mcts_simulation
from checkers.ai.value_network import create_model as vn_create_model
from checkers.ai.sl_action_policy import create_model as sl_create_model
from checkers.cli.interface import (
    position_to_coords,
    parse_coordinate,
    coords_to_position,
    display_board
)


# Configuration
POLICY_NETWORK_PATH = "checkpoints/rl/iter_1990.pth"
VALUE_NETWORK_PATH = "checkpoints/value_network/best_model.pth"
NUM_SIMULATIONS = 100  # Number of MCTS simulations per AI move
AI_COLOR = "B"  # AI plays Black, human plays White
UPDATE_INTERVAL = 10  # Show N_s_a stats every N simulations
TOP_N_MOVES = 5  # Show top N moves during updates


def load_models():
    """Load policy and value networks for MCTS.

    Returns:
        tuple: (policy_model, value_model, device)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load value network
    value_model = vn_create_model().to(device)
    checkpoint = torch.load(VALUE_NETWORK_PATH, map_location=device)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    value_model.eval()

    # Load policy network
    policy_model = sl_create_model().to(device)
    checkpoint = torch.load(POLICY_NETWORK_PATH, map_location=device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    policy_model.eval()

    return policy_model, value_model, device


def display_move_stats(state, N_s_a, Q_s_a, legal_actions):
    """Display N_s_a counts and Q/N values for legal moves.

    Args:
        state: Current game state
        N_s_a: Visit count dictionary
        Q_s_a: Q-value dictionary
        legal_actions: List of legal moves
    """
    hashed_state = hash_state(state)

    # Collect stats for each legal move
    stats = []
    for move in legal_actions:
        visits = N_s_a.get((hashed_state, str(move)), 0)
        if visits > 0:
            avg_value = Q_s_a.get((hashed_state, str(move)), 0.0) / visits
            stats.append((move, visits, avg_value))

    # Sort by visit count (highest first)
    stats.sort(key=lambda x: x[1], reverse=True)

    # Display top N moves
    print(f"  {'Move':<12} {'Visits':<10} {'Q/N (avg value)':<15}")
    print(f"  {'-'*12} {'-'*10} {'-'*15}")
    for move, visits, avg_val in stats[:TOP_N_MOVES]:
        from_pos, to_pos = move
        from_r, from_c = position_to_coords(from_pos)
        to_r, to_c = position_to_coords(to_pos)
        from_coord = chr(ord('a') + from_c) + str(from_r + 1)
        to_coord = chr(ord('a') + to_c) + str(to_r + 1)
        move_str = f"{from_coord} {to_coord}"
        print(f"  {move_str:<12} {visits:<10} {avg_val:>+7.3f}")


def run_mcts_search(state, num_sims, policy_model, value_model, device):
    """Run MCTS simulations from given state with live N_s_a updates.

    Args:
        state: Current game state
        num_sims: Number of simulations (e.g., 100)
        policy_model: Loaded policy network
        value_model: Loaded value network
        device: torch device

    Returns:
        best_move: Legal move with highest N_s_a count
    """
    # Initialize empty dictionaries
    N_s_a = {}
    Q_s_a = {}
    P_s_a = {}

    hashed_state = hash_state(state)
    legal_actions = legal_moves(state)

    print(f"\nRunning {num_sims} MCTS simulations from current position...")
    print("─" * 60)

    # Run simulations with progress updates
    for i in range(num_sims):
        N_s_a, Q_s_a, P_s_a = double_network_mcts_simulation(
            N_s_a,
            Q_s_a,
            P_s_a,
            policy_model,
            value_model,
            initial_state=state  # Start from current position!
        )

        # Update display every UPDATE_INTERVAL simulations
        if (i + 1) % UPDATE_INTERVAL == 0:
            print(f"\nAfter {i+1}/{num_sims} simulations:")
            display_move_stats(state, N_s_a, Q_s_a, legal_actions)

    # Select move with highest visit count
    best_move = max(
        legal_actions,
        key=lambda m: N_s_a.get((hashed_state, str(m)), 0)
    )

    print(f"\n✓ Selected move: {best_move}")
    print(f"  Visited {N_s_a[(hashed_state, str(best_move))]} times")
    print("=" * 60)

    return best_move


def main():
    """Main CLI game loop."""
    print("=" * 70)
    print(" " * 20 + "CHECKERS CLI")
    print(" " * 15 + "Network MCTS with Runtime Search")
    print("=" * 70)

    # Load models
    print("\nLoading neural networks...")
    policy_model, value_model, device = load_models()
    print(f"✓ Models loaded on {device}")

    print(f"\nGame Settings:")
    print(f"  AI plays: {AI_COLOR} (Black)")
    print(f"  Human plays: W (White)")
    print(f"  Simulations per AI move: {NUM_SIMULATIONS}")

    print("\nControls:")
    print("  - Enter moves as: 'a3 b4'")
    print("  - Type 'quit' to exit")
    print("=" * 70)

    # Initialize game
    state = reset()
    board = _state_to_board(state)
    current_turn = "W"

    while True:
        display_board(board)
        state = _board_to_state(board)

        # Check for game over
        moves = legal_moves(state)
        if not moves:
            winner = "Black" if current_turn == "W" else "White"
            print(f"\nGame Over! {winner} wins!")
            break

        current_player = "White" if current_turn == "W" else "Black"
        print(f"\n{current_player}'s turn")

        if current_turn == AI_COLOR:
            # AI's turn - run MCTS
            print("\nAI is thinking...")
            best_move = run_mcts_search(
                state,
                NUM_SIMULATIONS,
                policy_model,
                value_model,
                device
            )

            from_position, to_position = best_move

            # Execute AI move
            piece_index = None
            for i, p in enumerate(board.get_pieces()):
                if int(p.get_position()) == from_position:
                    if p.get_color() == current_turn:
                        piece_index = i
                        break

            if piece_index is not None:
                board.move_piece(piece_index, to_position)

                # Handle multi-jump
                piece_after_move = None
                for p in board.get_pieces():
                    if int(p.get_position()) == to_position:
                        piece_after_move = p
                        break

                if piece_after_move and piece_after_move.get_has_eaten():
                    next_moves = piece_after_move.get_moves(board)
                    eating_moves = [m for m in next_moves if m["eats_piece"]]
                    if eating_moves:
                        print(f"\n** Multi-jump available! AI continues jumping. **")
                        continue  # Same player continues

            # Switch turns
            current_turn = "B" if current_turn == "W" else "W"

        else:
            # Human's turn - get input
            while True:
                try:
                    move_input = input("Enter move: ").strip().lower()

                    if move_input == "quit":
                        print("Thanks for playing!")
                        return

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
                    if (from_position, to_position) not in moves:
                        print(f"Illegal move. Legal moves:")
                        for m in moves:
                            fr, fc = position_to_coords(m[0])
                            tr, tc = position_to_coords(m[1])
                            print(f"  {chr(ord('a') + fc)}{fr + 1} {chr(ord('a') + tc)}{tr + 1}")
                        continue

                    # Execute move
                    piece_index = None
                    for i, p in enumerate(board.get_pieces()):
                        if int(p.get_position()) == from_position:
                            if p.get_color() == current_turn:
                                piece_index = i
                                break

                    if piece_index is None:
                        print("No piece at that position or not your piece.")
                        continue

                    board.move_piece(piece_index, to_position)

                    # Handle multi-jump
                    piece_after_move = None
                    for p in board.get_pieces():
                        if int(p.get_position()) == to_position:
                            piece_after_move = p
                            break

                    if piece_after_move and piece_after_move.get_has_eaten():
                        next_moves = piece_after_move.get_moves(board)
                        eating_moves = [m for m in next_moves if m["eats_piece"]]
                        if eating_moves:
                            print(f"\n** Multi-jump available! You must continue jumping. **")
                            break  # Continue same player's turn

                    # Switch turns
                    current_turn = "B" if current_turn == "W" else "W"
                    break

                except KeyboardInterrupt:
                    print("\n\nThanks for playing!")
                    return
                except Exception as e:
                    print(f"Error: {e}")
                    continue


if __name__ == "__main__":
    main()
