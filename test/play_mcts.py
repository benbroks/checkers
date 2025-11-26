from checkers.api.environment import hash_state, reset, legal_moves, step
from checkers.ai.uct_mcts import load_mcts_data, single_turn_mcts_player
import random


def uct_mcts_vs_random(
    N_s,
    N_s_a,
    R_s_a,
    mcts_color: str = "W"
) -> int:
    state = reset()
    finished_sim = False
    while not finished_sim:
        if state['current_turn'] == mcts_color:
            print("mcts player")
            next_state, reward, done, info = single_turn_mcts_player(
                N_s,
                N_s_a,
                R_s_a,
                state
            )
            if done:
                return reward
            state = next_state
        else:
            print("random player")
            potential_actions = legal_moves(state)
            random_action = random.choice(potential_actions)
            next_state, reward, done, info = step(state, random_action)
            if done:
                return -1*reward
            state = next_state
    print("shouldn't hit this")
    return 0


def main():
    N_s, N_s_a, R_s_a = load_mcts_data()
    aggregate_results = {"wins": 0, "losses": 0, "ties": 0}
    for _ in range(1):
        result = uct_mcts_vs_random(
            N_s,
            N_s_a,
            R_s_a
        )
        if result == 1:
            aggregate_results["wins"] += 1
        elif result == -1:
            aggregate_results["losses"] += 1
        else:
            aggregate_results["ties"] += 1
    print(aggregate_results)

if __name__ == '__main__':
    main()
    # N_s, N_s_a, R_s_a = load_mcts_data()
    # for k in N_s_a:
    #     print(k)
    #     break
    # state = reset()
    # hashed_state = hash_state(state)
    # potential_moves = legal_moves(state)
    # print(N_s[hashed_state])
    # for a in potential_moves:
    #     print()
    #     print(a)
    #     print(
    #         N_s_a[(hashed_state, str(a))]
    #     )
    
    # print(hashed_state)
    # print(potential_moves)
