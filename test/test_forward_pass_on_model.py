from checkers.ai.sl_action_policy import create_model
from checkers.ai.rl_player import select_cnn_move_softmax
import torch

from src.checkers.api.environment import reset


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = create_model().to(device)
    temperature = 1.0
    state = reset()
    _ = select_cnn_move_softmax(
        model,
        state,
        temperature,
        device
    )

if __name__ == '__main__':
    main()