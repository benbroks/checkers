# Checkers AI with Neural Networks and MCTS

## Overview

This project implements advanced AI techniques for checkers, inspired by the [AlphaGo paper](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf). The system combines Monte Carlo Tree Search (MCTS) with deep neural networks to create a strong checkers player that learns through self-play.

**Training data:** Expert games sourced from [Fierz Checkers Archives](https://www.fierz.ch/download.php)

## Installation

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/python-checkers.git
   cd python-checkers
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Run scripts:
   ```bash
   uv run python test/play_gui.py
   ```

## AI Components

### Monte Carlo Tree Search (MCTS)

**`src/checkers/ai/uct_mcts.py`** - Traditional UCT-based MCTS implementation that learns through self-play. The algorithm maintains visit counts (`N_s_a`), state visit counts (`N_s`), and reward statistics (`R_s_a`) to balance exploration and exploitation.

**Usage:**
```python
from checkers.ai.uct_mcts import load_mcts_data, double_mcts_simulation, mcts_most_traveled

# Load or initialize MCTS data
N_s, N_s_a, R_s_a = load_mcts_data("mcts_data.json")

# Run simulations to improve the tree
for _ in range(1000):
    N_s, N_s_a, R_s_a = double_mcts_simulation(N_s, N_s_a, R_s_a)

# Select best move based on visit counts
best_action = mcts_most_traveled(state, N_s_a)
```

### Supervised Learning - Action Policy Network

**`src/checkers/ai/sl_action_policy.py`** - A CNN that predicts move probabilities. The network takes a 4-channel 8x4 board representation (white/black men/kings) and outputs a 32x8 action probability distribution. This can be trained on expert games or self-play data to learn strong move selection.

The architecture uses convolutional layers with batch normalization and dropout for regularization, suitable for imitation learning from game records.

### Reinforcement Learning - Policy Improvement

**`src/checkers/ai/rl_player.py`** - Extends the policy network with softmax sampling for exploration during training. Instead of always selecting the highest-probability move, it samples moves according to a temperature-controlled distribution, enabling policy gradient methods like REINFORCE.

**Training approach:** Generate self-play games using `single_turn_rl_player()`, collect trajectories with log probabilities, and update the network using policy gradient updates based on game outcomes.

### Value Network

**`src/checkers/ai/value_network.py`** - A CNN that estimates position quality, outputting values in [-1, 1] representing the expected game outcome from the current player's perspective. The network uses similar architecture to the policy network but outputs a single scalar value.

**Training:** The value network can be trained on completed games, learning to predict winners from intermediate positions. This provides faster position evaluation than random playouts.

### Neural Network MCTS

**`src/checkers/ai/network_mcts.py`** - Combines MCTS with neural networks, similar to AlphaGo. Instead of random playouts, leaf nodes are evaluated using a mix of the value network and a single random playout. The policy network guides action selection, replacing uniform exploration.

**Key features:**
- Policy network (`P_s_a`) provides prior probabilities for move selection
- Value network provides position evaluation at leaf nodes
- Mixing parameter balances neural network evaluation with rollout results
- Action selection uses `Q(s,a) + U(s,a)` where `U` depends on policy priors

**Usage:**
```python
from checkers.ai.network_mcts import double_network_mcts_simulation, load_network_mcts_data

# Load MCTS data and models
N_s_a, Q_s_a, P_s_a = load_network_mcts_data("network_mcts_data.json")
policy_model = load_policy_model()
value_model = load_value_model()

# Run network-guided MCTS
for _ in range(100):
    N_s_a, Q_s_a, P_s_a = double_network_mcts_simulation(
        N_s_a, Q_s_a, P_s_a, policy_model, value_model
    )
```

This integrated approach enables the system to learn from self-play and improve over time without human game data.

## Training Pipeline

**1. Train Policy Network (Supervised Learning)**
```bash
uv run python test/train_supervised.py
```
Trains on expert PDN games. Output: `checkpoints/best_model.pth`

**2. Generate Value Network Dataset**
```bash
uv run python test/generate_value_dataset.py
```
Generates 10k self-play games (~30-40 min). Output: `data/value_network_dataset.pkl`

**3. Train Value Network**
```bash
uv run python test/train_value_network.py
```
Trains position evaluator. Output: `checkpoints/value_network/best_model.pth`

**4. Reinforcement Learning (Optional)**
```bash
uv run python test/train_reinforcement_learning.py
```
Improves policy via self-play. Output: `checkpoints/rl/iter_*.pth`

## Playing the AI

**Play against Network MCTS:**
```bash
uv run python test/play_cli_with_network_mcts.py
```
Interactive CLI game. Human plays White, AI plays Black. Enter moves as `a3 b4`.

Inspired / branched off of @lucaskenji's [repo](https://github.com/lucaskenji/python-checkers).