from checkers.core.utils import get_position_with_row_col

class Board:
    def __init__(self, pieces, color_up, visit_counts=None):
        # Example: [Piece('12WND'), Piece('14BNU'), Piece('24WYD')]
        self.pieces = pieces
        self.color_up = color_up # Defines which of the colors is moving up.
        self.current_turn = "W" # White always starts
        self.visit_counts = visit_counts if visit_counts is not None else {}

    def reset(self):
        """Reset the board to initial game state with all pieces in starting positions."""
        from checkers.core.piece import Piece

        pieces = []

        # Black pieces (top 3 rows, positions 0-11)
        for pos in range(12):
            pieces.append(Piece(str(pos) + "BN"))

        # White pieces (bottom 3 rows, positions 20-31)
        for pos in range(20, 32):
            pieces.append(Piece(str(pos) + "WN"))

        self.pieces = pieces
        self.current_turn = "W"  # Reset to white's turn
        self.visit_counts = {}  # Clear visit history
    
    def get_color_up(self):
        return self.color_up

    def get_current_turn(self):
        return self.current_turn

    def set_current_turn(self, color):
        self.current_turn = color

    def get_pieces(self):
        return self.pieces

    def get_piece_by_index(self, index):
        return self.pieces[index]

    def get_visit_counts(self):
        return self.visit_counts

    def set_visit_counts(self, visit_counts):
        self.visit_counts = visit_counts

    def has_piece(self, position):
        # Receives position (e.g.: 28), returns True if there's a piece in that position
        string_pos = str(position)

        for piece in self.pieces:
            if piece.get_position() == string_pos:
                return True

        return False
    
    def get_row_number(self, position):
        # Receives position (e.g.: 1), returns the row this position is on the board.
        return position // 4
    
    def get_col_number(self, position):
        # There are four dark squares on each row where pieces can be placed.
        # The remainder of (position / 4) can be used to determine which of the four squares has the position.
        # We also take into account that odd rows on the board have a offset of 1 column.
        current_row = self.get_row_number(position)
        remainder = position % 4
        if current_row % 2 == 0:
            return remainder * 2 + 1
        else:
            return remainder * 2
    
    def get_row(self, row_number):
        # Receives a row number, returns a set with all pieces contained in it.
        # [0, 1, 2, 3] represents the first row of the board. All rows contain four squares.
        # row_pos needs to contain strings on it because Piece.get_position() returns a number in type string.

        row_pos = [0, 1, 2, 3]
        row_pos = list(map((lambda pos: str(pos + (4 * row_number))), row_pos))
        row = []

        for piece in self.pieces:
            if piece.get_position() in row_pos:
                row.append(piece)

        return set(row)
    
    def get_pieces_by_coords(self, *coords):
        # Receives a variable number of (row, column) pairs.
        # Returns a ordered list of same length with a Piece if found, otherwise None.
        row_memory = dict() # Used to not have to keep calling get_row().
        results = []

        for coord_pair in coords:
            if coord_pair[0] in row_memory:
                current_row = row_memory[coord_pair[0]]
            else:
                current_row = self.get_row(coord_pair[0])
                row_memory[coord_pair[0]] = current_row
            
            for piece in current_row:
                if self.get_col_number(int(piece.get_position())) == coord_pair[1]:
                    results.append(piece)
                    break
            else:
                # This runs if 'break' isn't called on the for loop above.
                results.append(None)
        
        return results
    
    def move_piece(self, moved_index, new_position):
        def is_eat_movement(current_position):
            # If the difference in the rows of the current and next positions isn't 1, i.e. if the piece isn't moving one square, 
            # then the piece is eating another piece.
            return abs(self.get_row_number(current_position) - self.get_row_number(new_position)) != 1

        def get_eaten_index(current_position):
            current_coords = [self.get_row_number(current_position), self.get_col_number(current_position)]
            new_coords = [self.get_row_number(new_position), self.get_col_number(new_position)]
            eaten_coords = [current_coords[0], current_coords[1]]

            # Dividing by 2 because neither the current position or the new one is desired, but the one in the middle.
            # Getting the difference between the new coordinates and current coordinates helps getting the direction.
            eaten_coords[0] += (new_coords[0] - current_coords[0]) // 2
            eaten_coords[1] += (new_coords[1] - current_coords[1]) // 2

            # Converting to string to compare later.
            eaten_position = str(get_position_with_row_col(eaten_coords[0], eaten_coords[1]))

            for index, piece in enumerate(self.pieces):
                if piece.get_position() == eaten_position:
                    return index

        def is_king_movement(piece):
            # Receives the piece moving and returns True if the move turns that piece into a king.
            if piece.is_king() == True:
                return False
            
            end_row = self.get_row_number(new_position)
            piece_color = piece.get_color()
            king_row = 0 if self.color_up == piece_color else 7

            return end_row == king_row

        piece_to_move = self.pieces[moved_index]

        # Delete piece from the board if this move eats another piece
        if is_eat_movement(int(piece_to_move.get_position())):

            get_eaten = get_eaten_index(int(piece_to_move.get_position()))
            self.pieces.pop(get_eaten)
            piece_to_move.set_has_eaten(True)
        else:
            piece_to_move.set_has_eaten(False)

        # Turn piece into a king if it reaches the other side of the board
        piece_was_kinged = False
        if is_king_movement(piece_to_move):
            piece_to_move.set_is_king(True)
            piece_was_kinged = True

        # Actually move
        piece_to_move.set_position(new_position)

        # Handle turn switching - check for multi-jump scenario
        if piece_to_move.get_has_eaten():
            # If piece was just kinged, turn ends immediately
            if piece_was_kinged:
                self.current_turn = "B" if self.current_turn == "W" else "W"
            else:
                # Check if this piece can make another jump
                next_moves = piece_to_move.get_moves(self)
                has_more_jumps = any(move["eats_piece"] for move in next_moves)

                if has_more_jumps:
                    # Multi-jump available - keep same player's turn
                    pass
                else:
                    # No more jumps - switch turns
                    self.current_turn = "B" if self.current_turn == "W" else "W"
        else:
            # Normal move (no capture) - switch turns
            self.current_turn = "B" if self.current_turn == "W" else "W"

        # Update visit counts for the new position
        from checkers.core.state_utils import create_position_hash
        position_hash = create_position_hash(self)
        self.visit_counts[position_hash] = self.visit_counts.get(position_hash, 0) + 1
    
    def get_winner(self):
        # Returns the winning color or None if no player has won yet
        current_color = self.pieces[0].get_color()

        for piece in self.pieces:
            if piece.get_color() != current_color:
                break
        else:
            return current_color

        return None

    def legal_moves(self):
        """
        Get all legal moves for the current player.
        Returns a list of tuples: [(from_position, to_position), ...]
        Enforces forced jump rule: if any capture is available, only capture moves are returned.
        For multi-jumps, only returns the first jump; call again after executing to get next jump.
        """
        legal_moves_list = []
        has_capture = False

        # Get all pieces for current player
        current_player_pieces = [
            (i, piece) for i, piece in enumerate(self.pieces)
            if piece.get_color() == self.current_turn
        ]

        # First pass: check if any piece has a capture move
        for piece_index, piece in current_player_pieces:
            moves = piece.get_moves(self)
            if any(move["eats_piece"] for move in moves):
                has_capture = True
                break

        # Second pass: collect legal moves
        for piece_index, piece in current_player_pieces:
            from_position = int(piece.get_position())
            moves = piece.get_moves(self)

            for move in moves:
                to_position = int(move["position"])

                # Forced jump rule: if captures exist, only include capture moves
                if has_capture:
                    if move["eats_piece"]:
                        legal_moves_list.append((from_position, to_position))
                else:
                    legal_moves_list.append((from_position, to_position))

        return legal_moves_list