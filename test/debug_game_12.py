"""
Debug script for Game 12 illegal move error.

This script replays Game 12 from full.pdn and visualizes the board state
at each move to identify where the illegal move error occurs.

Usage:
    PYTHONPATH=./src python test/debug_game_12.py

Or with venv:
    PYTHONPATH=./src ./venv/bin/python test/debug_game_12.py
"""

from checkers.core.pdn_utils import parse_pdn_games, parse_move, idx_conversion
from checkers.api.environment import reset, step, legal_moves, _state_to_board
from checkers.cli.interface import display_board


def debug_game_12():
    """Debug the 12th game from full.pdn."""

    print("\n" + "="*70)
    print("Loading games from full.pdn...")
    print("="*70)

    # Load all games
    all_games = list(parse_pdn_games('full.pdn'))

    print(f"Total games loaded: {len(all_games)}")

    # Get game 12 (index 11, since 0-indexed)
    game_12_moves = all_games[11]

    print("\n" + "="*70)
    print("Game 12: Edinburgh 1847, game 12")
    print("="*70)
    print(f"Total moves in game: {len(game_12_moves)}")
    print()

    state = reset()

    for move_num, move_str in enumerate(game_12_moves, 1):
        print("\n" + "="*70)
        print(f"Move {move_num}: {move_str}")
        print("="*70)

        # Parse PDN move
        pdn_source, pdn_dest = parse_move(move_str)
        print(f"PDN indices: {pdn_source} -> {pdn_dest}")

        # Convert to internal (CURRENT INCORRECT CONVERSION)
        internal_source = idx_conversion(pdn_source)
        internal_dest = idx_conversion(pdn_dest)
        print(f"Internal indices (current conversion): ({internal_source}, {internal_dest})")

        # Show legal moves
        legal = legal_moves(state)
        print(f"Legal moves: {legal}")

        # Try to make the move
        action = (internal_source, internal_dest)

        try:
            # Display board before move
            print(f"\nBoard BEFORE move {move_num} ({move_str}):")
            print("-"*70)
            board = _state_to_board(state)
            display_board(board)

            # Attempt move
            state, _, done, _ = step(state, action)

            print(f"\n✓ Move {move_num} successful: {move_str}")

            if done:
                print("\n" + "="*70)
                print("GAME OVER")
                print("="*70)
                break

        except ValueError as e:
            print("\n" + "!"*70)
            print(f"ERROR at move {move_num}")
            print("!"*70)
            print(f"PDN move: {move_str}")
            print(f"PDN indices: ({pdn_source}, {pdn_dest})")
            print(f"Converted to internal (WRONG): ({internal_source}, {internal_dest})")
            print(f"Legal moves were: {legal}")
            print(f"Error: {e}")
            print("\n" + "!"*70)
            print("Board state at time of error:")
            print("!"*70)
            board = _state_to_board(state)
            display_board(board)

            # Show what the CORRECT conversion would be
            correct_source = pdn_source - 1
            correct_dest = pdn_dest - 1
            print("\n" + "="*70)
            print("DIAGNOSIS:")
            print("="*70)
            print(f"The idx_conversion() function uses: 32 - idx")
            print(f"  PDN {pdn_source} -> {internal_source} (WRONG)")
            print(f"  PDN {pdn_dest} -> {internal_dest} (WRONG)")
            print(f"\nCORRECT conversion should be: idx - 1")
            print(f"  PDN {pdn_source} -> {correct_source} (CORRECT)")
            print(f"  PDN {pdn_dest} -> {correct_dest} (CORRECT)")
            print(f"\nIs ({correct_source}, {correct_dest}) a legal move? {(correct_source, correct_dest) in legal}")

            # Show the legal move and convert it back to PDN
            if legal:
                legal_internal = legal[0]
                legal_pdn_source = legal_internal[0] + 1
                legal_pdn_dest = legal_internal[1] + 1
                print(f"\nThe legal move {legal_internal} in internal coords")
                print(f"  corresponds to PDN move: {legal_pdn_source}-{legal_pdn_dest}")
                print(f"\nNote: We expected PDN move {pdn_source}-{pdn_dest}")
                print(f"      But got internal {legal_internal} which is PDN {legal_pdn_source}-{legal_pdn_dest}")

                # This suggests the PDN file might use a different numbering system
                print(f"\nWAIT - if the legal move is ({legal_internal[0]}, {legal_internal[1]})")
                print(f"  and idx_conversion() uses 32-idx...")
                print(f"  Then PDN source would be: 32 - {legal_internal[0]} = {32 - legal_internal[0]}")
                print(f"  And PDN dest would be: 32 - {legal_internal[1]} = {32 - legal_internal[1]}")
                print(f"\nSo maybe idx_conversion IS correct and PDN uses inverted numbering?")
            print("="*70)

            break

    print("\n" + "="*70)
    print("Debug complete")
    print("="*70)


if __name__ == '__main__':
    debug_game_12()
