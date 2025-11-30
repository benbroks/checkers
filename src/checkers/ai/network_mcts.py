import os
import json
from typing import Any
import torch
import random
from checkers.api.environment import hash_state, reset, legal_moves, step
from checkers.core.state_utils import state_to_cnn_input, action_to_cnn_output
from checkers.ai.uct_mcts import random_playthrough
from checkers.ai.value_network import create_model as vn_create_model
from checkers.ai.sl_action_policy import create_model as sl_create_model
from checkers.ai.cnn_player import cnn_move_softmax

MIXING_PARAMETER = 0.5
POLICY_NETWORK_PATH = "checkpoints/rl/iter_1990.pth"
VALUE_NETWORK_PATH = "checkpoints/value_network/best_model.pth"
DEVICE = torch.device("mps")


def save_network_mcts_data(
    N_s_a,
    Q_s_a,
    P_s_a,
    filepath: str = "network_mcts_data.json"
) -> None:
    """
    Save MCTS dictionaries to disk.

    Args:
        filepath: Path to save the data (default: "network_mcts_data.json")

    Example:
        >>> save_mcts_data("my_mcts_checkpoint.json")
    """
    # Convert tuple keys to string keys for JSON serialization
    N_s_a_serializable = {
        f"{state}||{action}": value
        for (state, action), value in N_s_a.items()
    }
    Q_s_a_serializable = {
        f"{state}||{action}": value
        for (state, action), value in Q_s_a.items()
    }
    P_s_a_serializable = {
        f"{state}||{action}": value
        for (state, action), value in P_s_a.items()
    }

    data = {
        "N_s_a": N_s_a_serializable,
        "Q_s_a": Q_s_a_serializable,
        "P_s_a": P_s_a_serializable
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved MCTS data to {filepath}")
    print(f"  - {len(Q_s_a)} state-action pairs")


def load_network_mcts_data(filepath: str = "network_mcts_data.json") -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    """
    Load Network MCTS dictionaries from disk.

    Args:
        filepath: Path to load the data from (default: "mcts_data.json")

    Example:
        >>> load_mcts_data("my_mcts_checkpoint.json")
    """

    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        N_s_a: dict[tuple[str, str], int] = {}
        Q_s_a: dict[tuple[str, str], int] = {}
        P_s_a: dict[tuple[str, str], int] = {}

        return N_s_a, Q_s_a, P_s_a

    with open(filepath, 'r') as f:
        data = json.load(f)

    N_s_a = {
        tuple(key.split("||")): value
        for key, value in data["N_s_a"].items()
    }
    Q_s_a = {
        tuple(key.split("||")): value
        for key, value in data["Q_s_a"].items()
    }
    P_s_a = {
        tuple(key.split("||")): value
        for key, value in data["P_s_a"].items()
    }

    print(f"✓ Loaded Network MCTS data from {filepath}")
    print(f"  - {len(N_s_a)} state-action pairs")
    return N_s_a, Q_s_a, P_s_a


def network_action_score(
    state,
    action,
    N_s_a,
    Q_s_a,
    P_s_a
) -> float:
    hashed_state = hash_state(state)
    num_actions_from_state = N_s_a[(hashed_state, str(action))]
    q_val = Q_s_a[(hashed_state, str(action))] / num_actions_from_state
    policy_network_score = P_s_a[(hashed_state, str(action))]
    u_val = policy_network_score / (1 + num_actions_from_state)
    return q_val + u_val


def network_mcts_exploit(
    state,
    N_s_a,
    Q_s_a,
    P_s_a,
    policy_model
) -> tuple[tuple[int, int], Any]:
    probs, legal_actions = cnn_move_softmax(
        policy_model,
        state,
        DEVICE
    )
    hashed_state = hash_state(state)
    best_action, best_score = legal_actions[0], float(-1)
    for idx, la in enumerate(legal_actions):
        P_s_a[(hashed_state, str(la))] = probs[idx].item()
        candidate_score = network_action_score(
            state,
            la,
            N_s_a,
            Q_s_a,
            P_s_a
        )
        if candidate_score > best_score:
            best_action = la
            best_score = candidate_score
    return best_action, P_s_a


def leaf_eval(
    state,
    action_to_take,
    value_network_model
) -> tuple[float, int]:
    cnn_input = state_to_cnn_input(state)
    tensor_input = torch.from_numpy(cnn_input).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        value_network_output = value_network_model(tensor_input).squeeze(0).item()

    _, reward, winner = random_playthrough(
        state,
        action_to_take
    )
    winner_reward = (1-MIXING_PARAMETER)*value_network_output + MIXING_PARAMETER * reward
    if winner == state['current_turn']:
        return winner_reward, winner
    else:
        return -1 * winner_reward, winner


def update_network_mcts_maps(
    N_s_a,
    Q_s_a,
    path,
    reward,
    winner
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    for sample_state, sample_action in path:
        hashed_sample_state = hash_state(sample_state)
        if (hashed_sample_state, str(sample_action)) not in N_s_a:
            N_s_a[(hashed_sample_state, str(sample_action))] = 1
        else:
            N_s_a[(hashed_sample_state, str(sample_action))] += 1
        
        current_player = sample_state['current_turn']
        if winner == current_player:
            actual_reward = reward
        else:
            actual_reward = -1*reward

        if (hashed_sample_state, str(sample_action)) not in Q_s_a:
            Q_s_a[(hashed_sample_state, str(sample_action))] = actual_reward
        else:
            Q_s_a[(hashed_sample_state, str(sample_action))] += actual_reward

    return N_s_a, Q_s_a


def double_network_mcts_simulation(
    N_s_a,
    Q_s_a,
    P_s_a,
    policy_model,
    value_network_model,
    initial_state=None
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int]
]:
    state = initial_state if initial_state is not None else reset()
    path = []
    finished_sim = False
    num_turns = 0
    while not finished_sim:
        num_turns += 1
        hashed_state = hash_state(state)
        possible_moves = legal_moves(state)
        # if num_turns > 500:
        #     print(state, possible_moves)
        have_taken_actions = [
            (hashed_state, str(move)) in N_s_a
            for move in possible_moves
        ]
        taken_all_actions = all(have_taken_actions)
        if taken_all_actions:
            best_action, P_s_a = network_mcts_exploit(
                state,
                N_s_a,
                Q_s_a,
                P_s_a,
                policy_model
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
            path.append((state, action_to_take))
            reward, winner = leaf_eval(
                state,
                action_to_take,
                value_network_model
            )
            finished_sim = True
    N_s_a, Q_s_a = update_network_mcts_maps(N_s_a, Q_s_a, path, reward, winner)
    return N_s_a, Q_s_a, P_s_a


def main():
    # Run training when script is executed directly
    import time
    random.seed(time.time() % 10000)
    # INJECT_RANDOMNESS = False
    N_s_a, Q_s_a, P_s_a = load_network_mcts_data("network_mcts_data.json")
    num_sims = 1000
    # init value model
    value_network_model = vn_create_model().to(DEVICE)
    value_network_checkpoint = torch.load(
        VALUE_NETWORK_PATH, 
        map_location=DEVICE
    )
    value_network_model.load_state_dict(value_network_checkpoint['model_state_dict'])
    value_network_model.eval()
    # init policy model
    policy_model = sl_create_model().to(DEVICE)
    policy_checkpoint = torch.load(
        POLICY_NETWORK_PATH, 
        map_location=DEVICE
    )
    policy_model.load_state_dict(policy_checkpoint['model_state_dict'])
    policy_model.eval()

    for i in range(num_sims):
        double_network_mcts_simulation(
            N_s_a,
            Q_s_a,
            P_s_a,
            policy_model,
            value_network_model
        )

        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{num_sims}:")
            print(f"  N_s_a: {len(N_s_a)}")

    # Save final model
    save_network_mcts_data(
        N_s_a,
        Q_s_a,
        P_s_a,
        "network_mcts_data.json"
    )
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()