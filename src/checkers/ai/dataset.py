"""
Dataset for supervised learning from PDN files.

This module provides a PyTorch Dataset class that loads checkers games from PDN files
and extracts (state, action) pairs for supervised learning.
"""

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from checkers.core.pdn_utils import parse_pdn_games, parse_move, idx_conversion
from checkers.api.environment import reset, step
from checkers.core.state_utils import state_to_cnn_input, action_to_cnn_output



class CheckersDataset(Dataset):
    """Dataset for supervised learning from PDN files.

    This dataset loads games from PDN files and extracts (state, action) pairs
    by replaying each game through the environment.

    Each sample is a tuple of:
        - state: (4, 8, 4) tensor representing the board state
        - action: (32, 8) one-hot tensor representing the action taken

    Args:
        pdn_file_paths: List of paths to PDN files
        max_games: Maximum number of games to load (None = load all)
    """

    def __init__(self, pdn_file_paths, max_games=None):
        self.samples = []
        self._load_data(pdn_file_paths, max_games)

    def _load_data(self, pdn_file_paths, max_games):
        """
        Load PDN games and extract (state, action) pairs.

        TODO (USER IMPLEMENTS):
        1. Use parse_pdn_games() from pdn_utils to load games
        2. Replay each game using environment.step()
        3. For each move:
           - Get state before move using environment state
           - Convert to CNN input: state_to_cnn_input(state)
           - Convert action to target: action_to_cnn_output(action)
           - Store (state_tensor, action_tensor) pair

        IMPORTANT:
        - PDN uses positions 1-32, internal uses 0-31
        - Use idx_conversion() from pdn_utils.py
        - Use parse_move() to convert move strings to tuples
        - Multi-jump moves are already split by parse_pdn_games()
        """
        games_loaded = 0
        max_games = 500  # Cutoff at 500 games

        # Count total games = sum for all PDN files, for tqdm
        if isinstance(pdn_file_paths, str):
            pdn_file_paths = [pdn_file_paths]
        all_games = []
        for pdn_file_path in pdn_file_paths:
            all_games.extend(list(parse_pdn_games(pdn_file_path)))

        for game_moves in tqdm(all_games[:max_games], desc="Loading Games", unit="game"):
            state = reset()
            for move_num, move_str in enumerate(game_moves, 1):
                idx_list = parse_move(move_str)
                action = (
                    idx_conversion(idx_list[0]),
                    idx_conversion(idx_list[1])
                )
                state_tensor = state_to_cnn_input(state)
                action_tensor = action_to_cnn_output(action)
                self.samples.append((state_tensor, action_tensor))

                next_state, _, _, _ = step(
                    state,
                    action
                )
                state = next_state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample.

        Returns:
            tuple: (state_tensor, action_tensor) where:
                - state_tensor: (4, 8, 4) float tensor
                - action_tensor: (32, 8) float tensor
        """
        state, action = self.samples[idx]
        return (
            torch.from_numpy(state).float(),  # (4, 8, 4)
            torch.from_numpy(action).float()  # (32, 8)
        )
