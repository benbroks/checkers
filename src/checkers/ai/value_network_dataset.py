"""
Dataset for value network supervised learning on played games.

This module provides a PyTorch Dataset class that 
dynamically generates game sequences with our RL-ed policy network.
"""

import torch
import os
import random
import pickle
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from checkers.core.pdn_utils import parse_pdn_games, parse_move, idx_conversion
from checkers.api.environment import reset, step, legal_moves
from checkers.core.state_utils import state_to_cnn_input, action_to_cnn_output
from checkers.ai.sl_action_policy import create_model
from checkers.ai.cnn_player import select_cnn_move_softmax



class CheckersValueNetworkDataset(Dataset):
    """
    Dataset for training value network using self-play games from policy network.

    The dataset generates game trajectories by having the trained policy network
    play against itself, then labels each state with the final game outcome.
    """

    def __init__(self, max_games=100, checkpoint_path='checkpoints/rl/iter_1990.pth',
                 device='cpu', dataset_cache_path='data/value_network_dataset.pkl'):
        """
        Initialize the dataset.

        Args:
            max_games: Number of self-play games to generate (default: 10000)
            checkpoint_path: Path to the trained policy network checkpoint
            device: Device to run the policy network on ('cpu', 'mps', 'cuda')
            dataset_cache_path: Path to save/load the generated dataset
        """
        self.samples = []
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dataset_cache_path = dataset_cache_path
        self.policy_model = None
        self._load_data(max_games)

    def _load_data(self, max_games):
        """
        Load or generate the dataset.

        First checks if cached dataset exists on disk. If so, loads it.
        Otherwise, loads the policy network and generates data through self-play.

        Args:
            max_games: Number of self-play games to generate
        """
        # Check if cached dataset exists
        if os.path.exists(self.dataset_cache_path):
            print(f"Loading cached dataset from {self.dataset_cache_path}...")
            with open(self.dataset_cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} samples from cache")
            return

        print(f"No cached dataset found. Generating new dataset with {max_games} games...")
        print(f"Loading policy network from {self.checkpoint_path}...")

        # Create and load policy model
        self.policy_model = create_model()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint['model_state_dict'])
        self.policy_model.to(self.device)
        self.policy_model.eval()

        print(f"Policy network loaded successfully on device: {self.device}")
        print(f"Model has {sum(p.numel() for p in self.policy_model.parameters()):,} parameters")

        # Generate self-play games
        print(f"\nGenerating {max_games} self-play games...")
        self._generate_selfplay_data(max_games)

        # Save to disk
        print(f"\nSaving dataset to {self.dataset_cache_path}...")
        os.makedirs(os.path.dirname(self.dataset_cache_path), exist_ok=True)
        with open(self.dataset_cache_path, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"Dataset saved! Total samples: {len(self.samples)}")

    def _generate_selfplay_data(self, num_games):
        """
        Generate training data by having policy network play against itself.

        For each game:
        - Play until completion
        - Select 2 random states (one where White moved, one where Black moved)
        - Label each state with game outcome from that player's perspective

        Args:
            num_games: Number of games to play
        """
        for game_idx in tqdm(range(num_games), desc="Self-play games"):
            # Play one game and collect all states
            game_states = []
            state = reset()

            max_moves = 500  # Prevent infinite games
            move_count = 0

            while move_count < max_moves:
                # Store current state and who is moving
                current_player = state['current_turn']
                game_states.append((state.copy(), current_player))

                # Get legal moves
                moves = legal_moves(state)
                if not moves:
                    # Game over - current player has no moves
                    break

                # Policy network samples move from softmax distribution
                try:
                    action = select_cnn_move_softmax(self.policy_model, state, self.device, temperature=1.0)
                except Exception:
                    # If model fails, game is a draw
                    break

                # Execute move
                state, reward, done, info = step(state, action)
                move_count += 1

                if done:
                    break

            # Determine game outcome
            if move_count >= max_moves:
                # Draw due to move limit
                outcome = 0
            else:
                # Check who won
                moves = legal_moves(state)
                if not moves:
                    # Current player (state['current_turn']) lost
                    loser = state['current_turn']
                    outcome = 1 if loser == 'B' else -1  # +1 for White win, -1 for Black win
                else:
                    # Game ended normally with reward signal
                    outcome = reward if state['current_turn'] == 'W' else -reward

            # Sample random states from the game
            if len(game_states) >= 2:
                # Separate states by player
                white_states = [s for s, player in game_states if player == 'W']
                black_states = [s for s, player in game_states if player == 'B']

                # Sample one from each if available
                if white_states:
                    white_state = random.choice(white_states)
                    white_value = outcome  # +1 if White won, -1 if Black won, 0 if draw
                    state_tensor = state_to_cnn_input(white_state)
                    self.samples.append((state_tensor, float(white_value)))

                if black_states:
                    black_state = random.choice(black_states)
                    black_value = -outcome  # Flip outcome for Black's perspective
                    state_tensor = state_to_cnn_input(black_state)
                    self.samples.append((state_tensor, float(black_value)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample.

        Returns:
            tuple: (state_tensor, value) where:
                - state_tensor: (4, 8, 4) numpy array
                - value: scalar float representing game outcome
                        (+1 = this player won, -1 = this player lost, 0 = draw)
        """
        state_tensor, value = self.samples[idx]
        # Convert to torch tensors
        state_tensor = torch.from_numpy(state_tensor).float()
        value = torch.tensor([value], dtype=torch.float32)
        return state_tensor, value
