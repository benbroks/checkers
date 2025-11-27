#!/usr/bin/env python3
"""
CLI checkers game that displays MCTS Q-values for each available action.
"""

from checkers.core.board import Board
from checkers.core.piece import Piece
from checkers.api.environment import hash_state, reset as api_reset, legal_moves as api_legal_moves, _board_to_state, _state_to_board, reset
from checkers.ai.uct_mcts import load_mcts_data
from checkers.cli.interface import (
    position_to_coords,
    create_initial_pieces,
    parse_coordinate,
    coords_to_position,
    display_board
)
import sys

def get_mcts_stats(state, N_s, N_s_a, R_s_a):
    """Get MCTS statistics for all legal moves in the current state."""
    hashed_state = hash_state(state)
    moves = api_legal_moves(state)

    stats = []
    for move in moves:
        move_key = (hashed_state, str(move))

        if move_key in N_s_a and N_s_a[move_key] > 0:
            visits = N_s_a[move_key]
            total_reward = R_s_a.get(move_key, 0)
            q_value = total_reward / visits
            stats.append({
                'move': move,
                'visits': visits,
                'q_value': q_value,
                'total_reward': total_reward
            })
        else:
            stats.append({
                'move': move,
                'visits': 0,
                'q_value': None,
                'total_reward': 0
            })

    # Sort by Q-value (unexplored moves last)
    stats.sort(key=lambda x: (x['q_value'] is None, -x['q_value'] if x['q_value'] is not None else 0))

    return stats


def display_mcts_stats(stats):
    """Display MCTS statistics for available moves."""
    if not stats:
        print("No MCTS data available for this position.")
        return

    print("\nMCTS Statistics:")
    print("─" * 60)
    print(f"{'Move':<12} {'Visits':<10} {'Q-Value':<12} {'Total Reward':<12}")
    print("─" * 60)

    for stat in stats:
        move = stat['move']
        from_pos, to_pos = move
        from_r, from_c = position_to_coords(from_pos)
        to_r, to_c = position_to_coords(to_pos)
        from_coord = chr(ord('a') + from_c) + str(from_r + 1)
        to_coord = chr(ord('a') + to_c) + str(to_r + 1)
        move_str = f"{from_coord} {to_coord}"

        visits = stat['visits']
        q_value = stat['q_value']
        total_reward = stat['total_reward']

        if q_value is not None:
            print(f"{move_str:<12} {visits:<10} {q_value:<12.3f} {total_reward:<12}")
        else:
            print(f"{move_str:<12} {visits:<10} {'unexplored':<12} {total_reward:<12}")

    print("─" * 60)


def main():
    print("Checkers CLI with MCTS Q-Values")
    print("=" * 40)

    # Load MCTS data
    print("Loading MCTS data...")
    N_s, N_s_a, R_s_a = load_mcts_data("mcts_data.json")

    # Initialize board
    init_state = reset()
    board = _state_to_board(init_state)

    current_turn = "W"

    print("\nControls:")
    print("  - Enter moves as: 'a3 b4'")
    print("  - Type 'quit' to exit")
    print("  - Type 'show' to see MCTS stats again")

    while True:
        display_board(board)

        # Get current state and MCTS stats
        state = _board_to_state(board)
        print(state)
        print(hash_state(state))
        stats = get_mcts_stats(state, N_s, N_s_a, R_s_a)

        # Display MCTS statistics
        display_mcts_stats(stats)

        # Check for game over
        moves = api_legal_moves(state)
        if not moves:
            winner = "Black" if current_turn == "W" else "White"
            print(f"\nGame Over! {winner} wins!")
            break

        current_player = "White" if current_turn == "W" else "Black"
        print(f"\n{current_player}'s turn")

        # Get player input
        while True:
            try:
                move_input = input("Enter move: ").strip().lower()

                if move_input == "quit":
                    print("Thanks for playing!")
                    sys.exit(0)

                if move_input == "show":
                    display_mcts_stats(stats)
                    continue

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

                # Find piece and execute move
                piece_index = None
                for i, p in enumerate(board.get_pieces()):
                    if int(p.get_position()) == from_position:
                        if p.get_color() == current_turn:
                            piece_index = i
                            break

                if piece_index is None:
                    print("No piece at that position or not your piece.")
                    continue

                # Execute move
                board.move_piece(piece_index, to_position)

                # Check for multi-jump
                piece_after_move = None
                for p in board.get_pieces():
                    if int(p.get_position()) == to_position:
                        piece_after_move = p
                        break

                if piece_after_move and piece_after_move.get_has_eaten():
                    # Check if more jumps available
                    next_moves = piece_after_move.get_moves(board)
                    eating_moves = [m for m in next_moves if m["eats_piece"]]

                    if eating_moves:
                        print(f"\n** Multi-jump available! {current_player} must continue jumping. **")
                        break  # Continue same player's turn

                # Switch turns
                current_turn = "B" if current_turn == "W" else "W"
                break

            except KeyboardInterrupt:
                print("\n\nThanks for playing!")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    main()
