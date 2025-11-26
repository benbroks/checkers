#!/usr/bin/env python3
"""
CLI version of checkers game for RL environment development.
Uses coordinate notation (e.g., 'a3 b4') for moves.
"""

from checkers.core.board import Board
from checkers.core.piece import Piece
from checkers.core.utils import get_position_with_row_col


def create_initial_pieces():
    """Create initial piece setup for checkers game."""
    pieces = []

    # Black pieces (top 3 rows, positions 0-11)
    for pos in range(12):
        pieces.append(Piece(str(pos) + "BN"))

    # White pieces (bottom 3 rows, positions 20-31)
    for pos in range(20, 32):
        pieces.append(Piece(str(pos) + "WN"))

    return pieces


def position_to_coords(position):
    """Convert position (0-31) to (row, col) coordinates."""
    row = position // 4
    # Determine if this row starts with even or odd column
    if row % 2 == 0:
        col = (position % 4) * 2 + 1
    else:
        col = (position % 4) * 2
    return row, col


def coords_to_position(row, col):
    """Convert (row, col) coordinates to position (0-31)."""
    return get_position_with_row_col(row, col)


def parse_coordinate(coord):
    """
    Parse coordinate like 'a3' to (row, col).
    Letters a-h map to columns 0-7.
    Numbers 1-8 map to rows 0-7.
    """
    if len(coord) != 2:
        return None

    col_letter = coord[0].lower()
    row_number = coord[1]

    if col_letter not in 'abcdefgh' or row_number not in '12345678':
        return None

    col = ord(col_letter) - ord('a')
    row = int(row_number) - 1

    return row, col


def display_board(board):
    """Display the board as a markdown table with pieces."""
    pieces_dict = {}

    # Build dictionary of positions and their pieces
    for piece in board.get_pieces():
        pos = int(piece.get_position())
        color = piece.get_color()
        is_king = piece.is_king()

        if is_king:
            pieces_dict[pos] = f"{color}K"
        else:
            pieces_dict[pos] = color

    print("\n")
    print("     a     b     c     d     e     f     g     h  ")
    print("   +-----+-----+-----+-----+-----+-----+-----+-----+")

    for row in range(8):
        row_display = f" {row + 1} |"

        for col in range(8):
            # Checkerboard pattern - only dark squares are playable
            if (row + col) % 2 == 1:
                # This is a dark (playable) square
                position = coords_to_position(row, col)
                if position in pieces_dict:
                    content = pieces_dict[position].center(4)
                else:
                    content = " · ".center(4)
            else:
                # Light square (not playable)
                content = "    "

            row_display += f" {content}|"

        print(row_display)
        print("   +-----+-----+-----+-----+-----+-----+-----+-----+")

    print()


def get_valid_moves_for_piece(piece, board):
    """Get valid moves for a piece."""
    return piece.get_moves(board)


def check_forced_jumps(pieces, board, current_color):
    """Check if any piece has a forced jump (eating move)."""
    for piece in pieces:
        if piece.get_color() == current_color:
            moves = piece.get_moves(board)
            for move in moves:
                if move["eats_piece"]:
                    return True
    return False


def get_piece_at_position(pieces, position):
    """Find piece at given position."""
    for i, piece in enumerate(pieces):
        if int(piece.get_position()) == position:
            return piece, i
    return None, None


def main():
    """Main game loop."""
    # Initialize game
    pieces = create_initial_pieces()
    board = Board(pieces, "W")  # White moves up
    current_turn = "W"  # White starts

    print("=" * 50)
    print("CHECKERS - CLI Version")
    print("=" * 50)
    print("\nEnter moves as: <from> <to>")
    print("Example: a3 b4")
    print("Type 'quit' to exit\n")

    while True:
        # Display board
        display_board(board)

        # Check for winner
        winner = board.get_winner()
        if winner:
            print(f"\n{'=' * 50}")
            print(f"{'White' if winner == 'W' else 'Black'} wins!")
            print(f"{'=' * 50}\n")
            break

        # Show current turn
        current_player = "White" if current_turn == "W" else "Black"
        print(f"{current_player}'s turn")

        # Check for forced jumps
        has_forced_jump = check_forced_jumps(board.get_pieces(), board, current_turn)
        if has_forced_jump:
            print("** Forced jump! You must capture an opponent's piece. **")

        # Get move input
        move_input = input("Enter move: ").strip().lower()

        if move_input == 'quit':
            print("Thanks for playing!")
            break

        # Parse move
        parts = move_input.split()
        if len(parts) != 2:
            print("Invalid format. Use: a3 b4")
            continue

        from_coord = parse_coordinate(parts[0])
        to_coord = parse_coordinate(parts[1])

        if not from_coord or not to_coord:
            print("Invalid coordinates. Use letters a-h and numbers 1-8.")
            continue

        from_row, from_col = from_coord
        to_row, to_col = to_coord

        # Check if coordinates are on dark squares
        if (from_row + from_col) % 2 == 0:
            print(f"Position {parts[0]} is not a playable square (must be dark square).")
            continue
        if (to_row + to_col) % 2 == 0:
            print(f"Position {parts[1]} is not a playable square (must be dark square).")
            continue

        from_position = coords_to_position(from_row, from_col)
        to_position = coords_to_position(to_row, to_col)

        # Find piece at from_position
        piece, piece_index = get_piece_at_position(board.get_pieces(), from_position)

        if not piece:
            print(f"No piece at {parts[0]}.")
            continue

        if piece.get_color() != current_turn:
            print(f"That's not your piece! It's {current_player}'s turn.")
            continue

        # Get valid moves for this piece
        valid_moves = get_valid_moves_for_piece(piece, board)

        # Filter for forced jumps if necessary
        if has_forced_jump:
            eating_moves = [m for m in valid_moves if m["eats_piece"]]
            if eating_moves:
                valid_moves = eating_moves
            else:
                print("This piece cannot make a jump. Choose a piece that can capture.")
                continue

        # Check if move is valid
        target_move = None
        for move in valid_moves:
            if int(move["position"]) == to_position:
                target_move = move
                break

        if not target_move:
            print(f"Invalid move. {parts[0]} cannot move to {parts[1]}.")
            if valid_moves:
                print("Valid moves for this piece:")
                for move in valid_moves:
                    pos = int(move["position"])
                    r, c = position_to_coords(pos)
                    coord = chr(ord('a') + c) + str(r + 1)
                    move_type = " (capture)" if move["eats_piece"] else ""
                    print(f"  - {coord}{move_type}")
            continue

        # Execute move
        board.move_piece(piece_index, to_position)

        # Check for multi-jump
        piece_after_move, _ = get_piece_at_position(board.get_pieces(), to_position)
        if piece_after_move and piece_after_move.get_has_eaten():
            # Check if more jumps are available
            next_moves = get_valid_moves_for_piece(piece_after_move, board)
            eating_moves = [m for m in next_moves if m["eats_piece"]]

            if eating_moves:
                print(f"\n** Multi-jump available! {current_player} must continue jumping. **")
                continue  # Same player's turn

        # Switch turns
        current_turn = "B" if current_turn == "W" else "W"


if __name__ == "__main__":
    # for i in range(32):
    #     print(i, position_to_coords(i))
    main()
