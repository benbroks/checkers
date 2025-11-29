"""
Quick manual test script for PDN parser.
Run with: PYTHONPATH=./src python3 test_pdn_manual.py
"""
from checkers.core.state_utils import parse_pdn_file, parse_pdn_string


def test_parse_file():
    print("Testing parse_pdn_file()...")
    moves = parse_pdn_file('sample.pdn')

    print(f"✓ Parsed {len(moves)} moves")

    # Verify first few moves
    assert moves[0] == '11-15', f"Expected '11-15', got '{moves[0]}'"
    assert moves[1] == '24-20', f"Expected '24-20', got '{moves[1]}'"
    print("✓ First moves correct")

    # Verify captures
    assert moves[6] == '15x22', f"Expected '15x22', got '{moves[6]}'"
    assert moves[7] == '25x18', f"Expected '25x18', got '{moves[7]}'"
    print("✓ Capture moves correct")

    # Verify multi-jump
    assert moves[-1] == '26x17x10x1', f"Expected '26x17x10x1', got '{moves[-1]}'"
    print("✓ Multi-jump capture correct")

    print(f"✓ All file parsing tests passed!\n")


def test_parse_string():
    print("Testing parse_pdn_string()...")

    # Test simple string
    pdn1 = "1. 11-15 24-20 2. 8-11 28-24 0-1"
    moves1 = parse_pdn_string(pdn1)
    assert len(moves1) == 4
    assert moves1 == ['11-15', '24-20', '8-11', '28-24']
    print("✓ Simple string parsing works")

    # Test with metadata
    pdn2 = """[Event "Test"]
[Date "2024-01-01"]
1. 11-15 24-20 2. 15x22 25x18 1-0"""
    moves2 = parse_pdn_string(pdn2)
    assert len(moves2) == 4
    assert '15x22' in moves2
    print("✓ Metadata filtering works")

    # Test multiline
    pdn3 = """1. 11-15 24-20
2. 8-11 28-24
3. 9-13 22-18 *"""
    moves3 = parse_pdn_string(pdn3)
    assert len(moves3) == 6
    print("✓ Multiline parsing works")

    # Test result indicators are filtered
    pdn4 = "1. 11-15 24-20 1-0"
    moves4 = parse_pdn_string(pdn4)
    assert '1-0' not in moves4
    assert '0-1' not in moves4
    print("✓ Result indicators filtered correctly")

    print(f"✓ All string parsing tests passed!\n")


if __name__ == '__main__':
    print("=" * 50)
    print("PDN Parser Tests")
    print("=" * 50 + "\n")

    try:
        test_parse_file()
        test_parse_string()

        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        exit(1)
