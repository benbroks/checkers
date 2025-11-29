import re

# PDN to State
def parse_pdn_games(filepath: str, split_multi_jumps: bool = True):
    """
    Generator that yields individual games from a PDN file.

    Each game is separated by metadata headers (lines starting with '[').
    The generator yields one game at a time as a list of move strings.

    Example usage:
        >>> for game_moves in parse_pdn_games('games.pdn'):
        ...     print(f"Game with {len(game_moves)} moves")
        ...     print(f"First move: {game_moves[0]}")

    Args:
        filepath: Path to the PDN file
        split_multi_jumps: If True, split multi-jump captures into separate moves.
                          e.g., "26x17x10x1" becomes ["26x17", "17x10", "10x1"]
                          Default: True

    Yields:
        list[str]: List of moves for each game
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Split content into individual games
    # Games are separated by metadata headers starting with '['
    games = []
    current_game_lines = []

    for line in content.split('\n'):
        stripped = line.strip()

        # If we hit a new game header and we have accumulated game moves, save the game
        if stripped.startswith('[Event') and current_game_lines:
            games.append('\n'.join(current_game_lines))
            current_game_lines = []

        current_game_lines.append(line)

    # Don't forget the last game
    if current_game_lines:
        games.append('\n'.join(current_game_lines))

    # Parse and yield each game
    for game_text in games:
        moves = parse_pdn_string(game_text, split_multi_jumps=split_multi_jumps)
        if moves:  # Only yield if there are actual moves
            yield moves


def parse_pdn_games_from_string(pdn_string: str, split_multi_jumps: bool = True):
    """
    Generator that yields individual games from a PDN string.

    Similar to parse_pdn_games but works with a string directly.

    Example usage:
        >>> pdn = "[Event \"Game 1\"]\\n1. 11-15 24-20\\n\\n[Event \"Game 2\"]\\n1. 9-13 22-18"
        >>> for game_moves in parse_pdn_games_from_string(pdn):
        ...     print(game_moves)

    Args:
        pdn_string: PDN formatted string containing one or more games
        split_multi_jumps: If True, split multi-jump captures into separate moves.
                          e.g., "26x17x10x1" becomes ["26x17", "17x10", "10x1"]
                          Default: True

    Yields:
        list[str]: List of moves for each game
    """
    games = []
    current_game_lines = []

    for line in pdn_string.split('\n'):
        stripped = line.strip()

        # If we hit a new game header and we have accumulated game moves, save the game
        if stripped.startswith('[Event') and current_game_lines:
            games.append('\n'.join(current_game_lines))
            current_game_lines = []

        current_game_lines.append(line)

    # Don't forget the last game
    if current_game_lines:
        games.append('\n'.join(current_game_lines))

    # Parse and yield each game
    for game_text in games:
        moves = parse_pdn_string(game_text, split_multi_jumps=split_multi_jumps)
        if moves:  # Only yield if there are actual moves
            yield moves


def parse_pdn_file(filepath: str, split_multi_jumps: bool = True) -> list[str]:
    """
    Parse a PDN (Portable Draughts Notation) file and extract moves.

    PDN format includes metadata headers in brackets (e.g., [Event "..."])
    followed by the game moves. Move numbers are prefixed with integers and periods.
    Moves are separated by spaces and can span multiple lines.

    Example input:
        1. 11-15 24-20 2. 8-11 28-24 3. 9-13 22-18 4. 15x22 25x18

    Example output (with split_multi_jumps=True):
        ["11-15", "24-20", "8-11", "28-24", "9-13", "22-18", "15x22", "25x18"]

    Example with multi-jump (26x17x10x1):
        split_multi_jumps=True  -> ["26x17", "17x10", "10x1"]
        split_multi_jumps=False -> ["26x17x10x1"]

    Args:
        filepath: Path to the PDN file
        split_multi_jumps: If True, split multi-jump captures into separate moves.
                          Default: True

    Returns:
        list[str]: List of moves in string format (e.g., "11-15", "15x22")
    """
    with open(filepath, 'r') as f:
        content = f.read()

    return parse_pdn_string(content, split_multi_jumps=split_multi_jumps)


def parse_pdn_string(pdn_string: str, split_multi_jumps: bool = True) -> list[str]:
    """
    Parse a PDN string and extract moves.

    Similar to parse_pdn_file but works with a string directly.

    Args:
        pdn_string: PDN formatted string
        split_multi_jumps: If True, split multi-jump captures into separate moves.
                          e.g., "26x17x10x1" becomes ["26x17", "17x10", "10x1"]
                          Default: True

    Returns:
        list[str]: List of moves in string format
    """
    # Remove metadata headers
    lines = [line for line in pdn_string.split('\n') if not line.strip().startswith('[')]

    # Join all game lines
    game_text = ' '.join(lines)

    # Remove curly bracket comments
    game_text = re.sub(r'\{[^}]*\}', '', game_text)

    # Split by whitespace
    tokens = game_text.split()

    moves = []
    for token in tokens:
        # Skip move numbers
        if token.endswith('.') and token[:-1].isdigit():
            continue

        # Skip result indicators
        if token in ['0-1', '1-0', '1/2-1/2', '*']:
            continue

        # Valid moves contain '-' or 'x'
        if '-' in token or 'x' in token:
            if split_multi_jumps and 'x' in token:
                # Split multi-jump captures
                positions = token.split('x')
                if len(positions) > 2:
                    # Multi-jump: create separate moves
                    for i in range(len(positions) - 1):
                        moves.append(f"{positions[i]}x{positions[i+1]}")
                else:
                    # Single jump
                    moves.append(token)
            else:
                moves.append(token)

    return moves


def idx_conversion(input_idx: int) -> int:
    """
    This works internal to pdn & vice versa.
    """
    return 32 - input_idx


def parse_move(move_str: str) -> tuple[int, int]:
    split_char = None
    if '-' in move_str:
        split_char = '-'
    elif 'x' in move_str:
        split_char = 'x'
    else:
        raise Exception("no split char!")
    split_idx = move_str.split(split_char)
    return (int(split_idx[0]), int(split_idx[1]))
