"""
Example usage of the PDN game generator with board visualization.
Run with: PYTHONPATH=./src python3 test_pdn_generator.py
"""
from checkers.core.pdn_utils import parse_pdn_games, parse_move, idx_conversion
from checkers.api.environment import reset, step, _state_to_board
from checkers.cli.interface import display_board

def main():
    """Run through first game in sample.pdn with visualization."""
    print("\n" + "="*60)
    print("PDN Game Visualization")
    print("="*60)
    print("\nPlaying through the first game in sample.pdn")
    print("Press Enter to see each move...")

    for game_idx, game_moves in enumerate(parse_pdn_games("sample.pdn"), 1):
        print(f"\n{'*'*60}")
        print(f"GAME {game_idx}: {len(game_moves)} moves")
        print(f"{'*'*60}")

        state = reset()
        print(state)
        board = _state_to_board(state)

        # Show initial board
        display_board(board)
        input()  # Wait for user to press Enter

        # Play through each move
        for move_num, move_str in enumerate(game_moves, 1):
            try:
                idx_list = parse_move(move_str)
                converted_idx_tuple = (
                    idx_conversion(idx_list[0]),
                    idx_conversion(idx_list[1])
                )

                next_state, reward, done, info = step(
                    state,
                    converted_idx_tuple
                )
                state = next_state

                # Convert state to board and display
                board = _state_to_board(state)
                display_board(board)

                # Show game info if done
                if done:
                    print(f"\n{'='*60}")
                    print("GAME OVER")
                    print(f"{'='*60}")
                    print(f"Winner: {info.get('winner', 'Draw')}")
                    print(f"Reward: {reward}")
                    print(f"Info: {info}")
                    print(f"{'='*60}\n")
                    break

                # Wait for user input to continue
                input()  # Press Enter for next move

            except Exception as e:
                print(f"\nError processing move {move_num} ({move_str}): {e}")
                import traceback
                traceback.print_exc()
                break

        # Ask if user wants to continue to next game
        if game_idx < len(list(parse_pdn_games("sample.pdn"))):
            response = input("\nContinue to next game? (y/n): ").strip().lower()
            if response != 'y':
                break


if __name__ == '__main__':
    main()
