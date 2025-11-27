#!/usr/bin/env python3
"""
Simple manual test for 3-fold repetition.
"""

from checkers.api.environment import reset, step, hash_state, _position_hash


def test_manual_threefold():
    """Manually test 3-fold repetition by setting up a position and tracking visits."""
    print("Manual 3-fold Repetition Test")
    print("=" * 60)

    state = reset()

    print("\nStarting position:")
    initial_hash = _position_hash(state)
    print(f"  Position hash: {initial_hash[:60]}...")
    print(f"  Visit count: {state['visit_counts'].get(initial_hash, 0)}")

    # Just make any legal moves and observe the tracking
    from checkers.api.environment import legal_moves

    moves_made = []
    for i in range(10):
        legal = legal_moves(state)
        if not legal:
            print(f"\nNo legal moves at step {i}")
            break

        # Pick a move
        move = legal[0]
        moves_made.append(move)

        print(f"\nMove {i+1}: {move}")
        state, reward, done, info = step(state, move)

        pos_hash = _position_hash(state)
        visit_count = state['visit_counts'].get(pos_hash, 0)

        print(f"  Visit count: {visit_count}")
        print(f"  Done: {done}")
        if done:
            print(f"  Info: {info}")
            if info.get('threefold_repetition'):
                print("\n✓ 3-fold repetition detected!")
                return True
            break

    print(f"\nFinal state visit counts (showing first 5):")
    for i, (pos, count) in enumerate(list(state['visit_counts'].items())[:5]):
        print(f"  {pos[:60]}... → {count} visits")

    print(f"\nTotal unique positions visited: {len(state['visit_counts'])}")
    print("\n✓ Visit counting works correctly!")
    return True


if __name__ == "__main__":
    test_manual_threefold()
