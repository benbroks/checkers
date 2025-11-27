from typing import Any
from checkers.api.environment import hash_state, reset, legal_moves, step
import random
import math
import json
import os
import pdb

# exploration constant
EXPLORATION_CONSTANT = 1.4


def save_mcts_data(
    N_s,
    N_s_a,
    R_s_a,
    filepath: str = "mcts_data.json"
) -> None:
    """
    Save MCTS dictionaries to disk.

    Args:
        filepath: Path to save the data (default: "mcts_data.json")

    Example:
        >>> save_mcts_data("my_mcts_checkpoint.json")
    """
    # Convert tuple keys to string keys for JSON serialization
    R_s_a_serializable = {
        f"{state}||{action}": value
        for (state, action), value in R_s_a.items()
    }
    N_s_a_serializable = {
        f"{state}||{action}": value
        for (state, action), value in N_s_a.items()
    }

    data = {
        "R_s_a": R_s_a_serializable,
        "N_s": N_s,
        "N_s_a": N_s_a_serializable,
        "exploration_constant": EXPLORATION_CONSTANT
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved MCTS data to {filepath}")
    print(f"  - {len(N_s)} unique states")
    print(f"  - {len(N_s_a)} state-action pairs")


def load_mcts_data(filepath: str = "mcts_data.json") -> tuple[
    dict[str, int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    """
    Load MCTS dictionaries from disk.

    Args:
        filepath: Path to load the data from (default: "mcts_data.json")

    Example:
        >>> load_mcts_data("my_mcts_checkpoint.json")
    """

    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        # R(s, a) = Raw utility of playing move a in state s ( q is this divided by N_s_a)
        R_s_a: dict[tuple[str, str], int] = {}

        # N(s) = number of times state s has been visited
        N_s: dict[str, int] = {}

        # N(s, a) = number of times move a has been taken from state s
        N_s_a: dict[tuple[str, str], int] = {}
        return N_s, N_s_a, R_s_a

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert string keys back to tuple keys
    R_s_a = {
        tuple(key.split("||")): value
        for key, value in data["R_s_a"].items()
    }
    N_s_a = {
        tuple(key.split("||")): value
        for key, value in data["N_s_a"].items()
    }
    N_s = data["N_s"]

    if "exploration_constant" in data:
        EXPLORATION_CONSTANT = data["exploration_constant"]

    print(f"✓ Loaded MCTS data from {filepath}")
    print(f"  - {len(N_s)} unique states")
    print(f"  - {len(N_s_a)} state-action pairs")
    print(f"  - Exploration constant: {EXPLORATION_CONSTANT}")
    return N_s, N_s_a, R_s_a


def random_playthrough(state, first_action) -> tuple[
    list[tuple[Any, str]],
    int,
    str
]:
    path = []
    done = False
    first_round = True
    while not done:
        if first_round:
            action = first_action
            first_round = False
        else:
            moves = legal_moves(state)
            action = random.choice(moves)
        path.append((state, action))
        next_state, reward, done, info = step(state, action)
        state = next_state
    return path, reward, info['winner']


def action_score(
    state,
    action,
    N_s,
    N_s_a,
    R_s_a
) -> float:
    hashed_state = hash_state(state)
    num_actions_from_state = N_s_a[(hashed_state, str(action))]
    q_val = R_s_a[(hashed_state, str(action))] / num_actions_from_state
    num_visits = N_s[hashed_state]
    exploration_factor = math.sqrt(
        math.log(num_visits) / num_actions_from_state
    )
    return q_val + EXPLORATION_CONSTANT * exploration_factor

def num_visits(
    state,
    action,
    N_s_a
) -> int:
    hashed_state = hash_state(state)
    tuple_key = (hashed_state, str(action))
    if tuple_key in N_s_a:
        return N_s_a[tuple_key]
    else:
        return 0



def update_mcts_maps(
    N_s,
    N_s_a,
    R_s_a,
    path,
    reward,
    winner
) -> tuple[
    dict[str, int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    for sample_state, sample_action in path:
        hashed_sample_state = hash_state(sample_state)

        if hashed_sample_state not in N_s:
            N_s[hashed_sample_state] = 1
        else:
            N_s[hashed_sample_state] += 1

        if (hashed_sample_state, str(sample_action)) not in N_s_a:
            N_s_a[(hashed_sample_state, str(sample_action))] = 1
        else:
            N_s_a[(hashed_sample_state, str(sample_action))] += 1

        current_player = sample_state['current_turn']
        if (hashed_sample_state, str(sample_action)) not in R_s_a:
            R_s_a[(hashed_sample_state, str(sample_action))] = 0

        # Note that a tie & loss are considered to have 0 reward here
        if reward != 0 and current_player == winner:
            R_s_a[(hashed_sample_state, str(sample_action))] += 1
        
        # if R_s_a[(hashed_sample_state, str(sample_action))] == 1:
        #     print(hashed_sample_state, str(sample_action))
        #     print(f"N_s: {N_s[hashed_sample_state]}")
        #     print(f"N_s_a: {N_s_a[(hashed_sample_state, str(sample_action))]}")
        #     print(f"R_s_a: {R_s_a[(hashed_sample_state, str(sample_action))]}")
    return N_s, N_s_a, R_s_a


def mcts_exploit(
    state,
    N_s,
    N_s_a,
    R_s_a
) -> Any | None:
    possible_moves = legal_moves(state)
    best_action, best_score = None, float(-1)
    for m in possible_moves:
        candidate_score = action_score(
            state,
            m,
            N_s,
            N_s_a,
            R_s_a
        )
        if candidate_score > best_score:
            best_action = m
            best_score = candidate_score
    return best_action

def mcts_most_traveled(
    state,
    N_s_a
) -> Any | None:
    possible_moves = legal_moves(state)
    best_action, best_score = None, -1
    for m in possible_moves:
        candidate_score = num_visits(
            state,
            m,
            N_s_a
        )
        if candidate_score > best_score:
            best_action = m
            best_score = candidate_score
    return best_action


def single_turn_mcts_player(
    N_s,
    N_s_a,
    state
):
    """
    To be used for pure-exploit testing. Not for "training".
    """
    hashed_state = hash_state(state)
    possible_moves = legal_moves(state)
    have_taken_actions = [
        (hashed_state, str(move)) in N_s_a
        for move in possible_moves
    ]
    taken_one_action = any(have_taken_actions)
    if hashed_state in N_s and taken_one_action:
        best_action = mcts_most_traveled(
            state,
            N_s_a
        )
        next_state, reward, done, info = step(state, best_action)
    else:
        action_to_take = random.choice(possible_moves)
        next_state, reward, done, info = step(state, action_to_take)
    return next_state, reward, done, info


def double_mcts_simulation(
    N_s,
    N_s_a,
    R_s_a
) -> tuple[
    dict[str, int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    state = reset()
    path = []
    finished_sim = False
    num_turns = 0
    while not finished_sim:
        num_turns += 1
        hashed_state = hash_state(state)
        possible_moves = legal_moves(state)
        if num_turns > 500:
            print(state)
            print(possible_moves)
        have_taken_actions = [
            (hashed_state, str(move)) in N_s_a
            for move in possible_moves
        ]
        taken_all_actions = all(have_taken_actions)
        if hashed_state in N_s and taken_all_actions:
            best_action = mcts_exploit(
                state,
                N_s,
                N_s_a,
                R_s_a
            )
            path.append((state, best_action))
            next_state, reward, done, info = step(state, best_action)
            state = next_state
            if done:
                finished_sim = True
                winner = info['winner']
        else:
            unexplored_actions = [
                a for have_taken, a
                in zip(have_taken_actions, possible_moves)
                if not have_taken
            ]
            action_to_take = random.choice(unexplored_actions)
            random_path, reward, winner = random_playthrough(
                state,
                action_to_take
            )
            path = path + random_path
            finished_sim = True
    return update_mcts_maps(N_s, N_s_a, R_s_a, path, reward, winner)


def one_way_mcts_simulation(
    N_s,
    N_s_a,
    R_s_a,
    mcts_player
) -> tuple[
    dict[str, int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    state = reset()
    path = []
    finished_sim = False

    while not finished_sim:
        
        current_player = state['current_turn']
        if current_player == mcts_player:
            hashed_state = hash_state(state)
            possible_moves = legal_moves(state)
            have_taken_actions = [
                (hashed_state, str(move)) in N_s_a
                for move in possible_moves
            ]
            taken_all_actions = all(have_taken_actions)
            if hashed_state in N_s and taken_all_actions:
                best_action = mcts_exploit(
                    state,
                    N_s,
                    N_s_a,
                    R_s_a
                )
                path.append((state, best_action))
                next_state, reward, done, info = step(state, best_action)
                state = next_state
                if done:
                    finished_sim = True
                    winner = info['winner']
            else:
                # COMPLETELY RANDOM (over unexplored actions)
                unexplored_actions = [
                    a for have_taken, a
                    in zip(have_taken_actions, possible_moves)
                    if not have_taken
                ]
                chosen_action = random.choice(unexplored_actions)
                path.append((state, chosen_action))
                next_state, reward, done, info = step(state, chosen_action)
                state = next_state
                if done:
                    finished_sim = True
                    winner = info['winner']
        else:
            # COMPLETELY RANDOM
            chosen_action = random.choice(legal_moves(state))
            next_state, reward, done, info = step(state, chosen_action)
            state = next_state
            if done:
                finished_sim = True
                winner = info['winner']

    return update_mcts_maps(N_s, N_s_a, R_s_a, path, reward, winner)


def main():
    # Run training when script is executed directly
    import time
    random.seed(time.time() % 10000)
    INJECT_RANDOMNESS = False
    N_s, N_s_a, R_s_a = load_mcts_data("mcts_data.json")
    num_sims = 2500
    for i in range(num_sims):
        # RANDOM
        if INJECT_RANDOMNESS:
            if i % 2 == 0:
                one_way_mcts_simulation(
                    N_s,
                    N_s_a,
                    R_s_a,
                    mcts_player="W"
                )
            else:
                one_way_mcts_simulation(
                    N_s,
                    N_s_a,
                    R_s_a,
                    mcts_player="B"
                )
        else:
            double_mcts_simulation(
                N_s,
                N_s_a,
                R_s_a
            )

        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{num_sims}:")
            print(f"  R_s_a: {len(R_s_a)}")
            print(f"  N_s_a: {len(N_s_a)}")
            print(f"  N_s: {len(N_s)}")

    # Save final model
    save_mcts_data(
        N_s,
        N_s_a,
        R_s_a,
        "mcts_data.json"
    )
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
