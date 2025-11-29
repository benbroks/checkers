from checkers.core.pdn_utils import parse_pdn_file, parse_pdn_string, parse_pdn_games


def test_parse_pdn_file():
    """Test parsing the sample PDN file."""
    moves = parse_pdn_file('sample.pdn')

    # Check total number of moves
    assert len(moves) == 44

    # Check first few moves
    assert moves[0] == '11-15'
    assert moves[1] == '24-20'
    assert moves[2] == '8-11'
    assert moves[3] == '28-24'

    # Check some captures
    assert moves[6] == '15x22'
    assert moves[7] == '25x18'

    # Check multi-jump capture at the end
    assert moves[-1] == '26x17x10x1'


def test_parse_pdn_string_simple():
    """Test parsing a simple PDN string."""
    pdn = """[Event "Test Game"]
[Date "2024-01-01"]
1. 11-15 24-20 2. 8-11 28-24 3. 9-13 22-18 0-1
"""
    moves = parse_pdn_string(pdn)

    assert len(moves) == 6
    assert moves == ['11-15', '24-20', '8-11', '28-24', '9-13', '22-18']


def test_parse_pdn_string_with_captures():
    """Test parsing PDN with captures."""
    pdn = "1. 11-15 24-20 2. 15x22 25x18 1-0"
    moves = parse_pdn_string(pdn)

    assert len(moves) == 4
    assert moves[2] == '15x22'
    assert moves[3] == '25x18'


def test_parse_pdn_string_multi_jump():
    """Test parsing PDN with multi-jump captures."""
    pdn = "1. 11-15 24-20 2. 15x24x31 1-0"
    moves = parse_pdn_string(pdn)

    assert len(moves) == 3
    assert moves[2] == '15x24x31'


def test_parse_pdn_string_multiline():
    """Test parsing PDN spread across multiple lines."""
    pdn = """1. 11-15 24-20 2. 8-11 28-24
3. 9-13 22-18 4. 15x22 25x18
5. 4-8 26-22 0-1"""
    moves = parse_pdn_string(pdn)

    assert len(moves) == 10
    assert moves[0] == '11-15'
    assert moves[-1] == '26-22'


def test_parse_pdn_empty():
    """Test parsing empty PDN string."""
    pdn = """[Event "Test"]
[Result "*"]
*"""
    moves = parse_pdn_string(pdn)

    assert len(moves) == 0


def test_parse_pdn_result_variations():
    """Test that different result indicators are properly filtered."""
    pdn1 = "1. 11-15 24-20 1-0"
    pdn2 = "1. 11-15 24-20 0-1"
    pdn3 = "1. 11-15 24-20 1/2-1/2"
    pdn4 = "1. 11-15 24-20 *"

    for pdn in [pdn1, pdn2, pdn3, pdn4]:
        moves = parse_pdn_string(pdn)
        assert len(moves) == 2
        assert '1-0' not in moves
        assert '0-1' not in moves
        assert '1/2-1/2' not in moves
        assert '*' not in moves
