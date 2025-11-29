"""
RL training loop using REINFORCE algorithm.

This module provides the main training loop for reinforcement learning
with self-play. Key functions are left as placeholders for the user to implement.

PLACEHOLDER FUNCTIONS:
- train_on_experience() - USER IMPLEMENTS THIS (REINFORCE loss)
- train() main loop - USER IMPLEMENTS THIS (training iteration structure)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from checkers.ai.sl_action_policy import create_model
from checkers.ai.rl_config import RLConfig
from checkers.ai.checkpoint_manager import CheckpointManager
from checkers.ai.self_play import run_self_play_iteration
from checkers.ai.evaluate import evaluate_vs_random
from checkers.core.state_utils import action_to_cnn_output


def train_on_experience(model, optimizer, buffer, config):
    """
    Train model on self-play experiences using REINFORCE algorithm.

    Args:
        model: CheckersCNN model
        optimizer: Optimizer (AdamW)
        buffer: ExperienceBuffer with experiences
        config: RLConfig instance

    Returns:
        float: Loss value for logging
    """
    from checkers.api.environment import legal_moves
    from checkers.core.state_utils import state_to_cnn_input

    device = torch.device(config.device)

    # Get experiences - need full Experience objects to access original states
    experiences = buffer.experiences

    if len(experiences) == 0:
        return 0.0

    model.train()
    optimizer.zero_grad()

    # Prepare batch tensors
    states = torch.stack([exp.state_tensor for exp in experiences]).to(device)
    rewards = torch.tensor(
        [exp.reward for exp in experiences],
        dtype=torch.float32,
        device=device
    )

    # Forward pass
    logits = model(states)   # (batch, 32, 8)
    B = logits.size(0)
    logits_flat = logits.view(B, -1) # (B, 256)

    # For each example, compute softmax over legal moves only
    chosen_log_probs = []

    for i, exp in enumerate(experiences):
        # Get legal moves for this state
        if exp.state_dict is None:
            # Fallback: compute softmax over full action space
            action = exp.action
            a_tensor = action_to_cnn_output(action)
            row, col = np.where(a_tensor == 1)
            action_idx = int(row[0] * 8 + col[0])
            example_logits = logits_flat[i]
            log_probs_full = F.log_softmax(example_logits, dim=0)
            chosen_log_prob = log_probs_full[action_idx]
        else:
            # Get legal moves from state
            legal_actions = legal_moves(exp.state_dict)

            # Convert legal actions to flat indices
            legal_indices = []
            for legal_action in legal_actions:
                a_tensor = action_to_cnn_output(legal_action)
                row, col = np.where(a_tensor == 1)
                idx = int(row[0] * 8 + col[0])
                legal_indices.append(idx)

            # Convert taken action to flat index
            action = exp.action
            a_tensor = action_to_cnn_output(action)
            row, col = np.where(a_tensor == 1)
            action_idx = int(row[0] * 8 + col[0])

            # Get logits for this example and mask illegal moves
            example_logits = logits_flat[i]  # (256,)

            # Create mask: -inf for illegal moves, 0 for legal moves
            mask = torch.full_like(example_logits, float('-inf'))
            mask[legal_indices] = 0.0

            # Add mask to logits and compute softmax (only over legal moves)
            masked_logits = example_logits + mask
            log_probs_masked = F.log_softmax(masked_logits, dim=0)

            # Get log prob of the chosen action
            chosen_log_prob = log_probs_masked[action_idx]

        chosen_log_probs.append(chosen_log_prob)

    chosen_log_probs = torch.stack(chosen_log_probs)

    # REINFORCE loss
    loss = -(chosen_log_probs * rewards).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    return loss.item()


def train(config: RLConfig):
    """
    Main RL training loop.

    [PLACEHOLDER - USER IMPLEMENTS MAIN LOOP STRUCTURE]

    This function should implement the main training loop:
    1. Setup (device, model, optimizer, checkpoint manager, tensorboard)
    2. Load pretrained SL model as starting point
    3. For each iteration:
       a. Calculate temperature for this iteration
       b. Run self-play to collect experiences
       c. Train on experiences using REINFORCE
       d. Save checkpoint
       e. Log statistics to console and tensorboard

    The setup code is provided below, but you need to implement the main training loop.

    Implementation hints:
    - Use config.get_temperature(iteration) to get current temperature
    - Use run_self_play_iteration() to play games and collect experiences
    - Use train_on_experience() to train on collected experiences
    - Use checkpoint_manager.save_checkpoint() to save model
    - Log to tensorboard: writer.add_scalar('WinRate', win_rate, iteration)
    - Print progress every iteration (or every N iterations)

    Args:
        config: RLConfig instance

    Raises:
        NotImplementedError: This is a placeholder - user must implement
    """
    # ========================================
    # SETUP (FULLY IMPLEMENTED)
    # ========================================
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load pretrained SL model as starting point
    print(f"Loading pretrained model from {config.pretrained_model}...")
    model = create_model().to(device)
    checkpoint = torch.load(config.pretrained_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Pretrained model loaded successfully")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create checkpoint manager
    ckpt_manager = CheckpointManager(
        config.checkpoint_dir,
        keep_last_n=config.keep_last_n,
        milestone_every=config.milestone_every
    )

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)

    print(f"\nStarting RL training for {config.num_iterations} iterations...")
    print(f"Temperature: {config.temperature_start} → {config.temperature_end} over {config.temperature_decay_iterations} iterations")
    print(f"Self-play: {config.games_per_iteration} games per iteration")
    print("=" * 70)

    # training loop
    for iteration in range(config.num_iterations):
        print(f"\\nIteration {iteration+1}/{config.num_iterations}")
        print("-" * 70)

        # Get current temperature
        temperature = config.get_temperature(iteration)

        # Run self-play
        buffer, win_rate, draws = run_self_play_iteration(
            model, ckpt_manager, config, iteration
        )

        # Train on experiences
        loss = train_on_experience(model, optimizer, buffer, config)

        # Save checkpoint
        stats = {
            'win_rate': win_rate,
            'loss': loss,
            'temperature': temperature,
            'num_experiences': len(buffer),
            'draws': draws
        }
        ckpt_manager.save_checkpoint(model, optimizer, iteration, stats)

        # Log to tensorboard
        writer.add_scalar('WinRate', win_rate, iteration)
        writer.add_scalar('Loss', loss, iteration)
        writer.add_scalar('Temperature', temperature, iteration)

        # Print summary
        print(f"  Loss: {loss:.4f}")
        print(f"  Win rate: {win_rate*100:.1f}%")

        # Periodic evaluation against random baseline
        if config.eval_enabled and (iteration + 1) % config.eval_every == 0:
            print(f"\n  === Evaluation vs Random (iteration {iteration+1}) ===")
            eval_stats = evaluate_vs_random(model, config.eval_games, device)

            # Log to tensorboard
            writer.add_scalar('Eval/WinRate', eval_stats['win_rate'], iteration)
            writer.add_scalar('Eval/Wins', eval_stats['wins'], iteration)
            writer.add_scalar('Eval/Losses', eval_stats['losses'], iteration)
            writer.add_scalar('Eval/Draws', eval_stats['draws'], iteration)

            # Print summary
            print(f"  Wins: {eval_stats['wins']} ({eval_stats['win_rate']*100:.1f}%)")
            print(f"  Losses: {eval_stats['losses']}")
            print(f"  Draws: {eval_stats['draws']}")
            print(f"  ============================================\n")

    writer.close()
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"TensorBoard logs saved to: {config.log_dir}")
    print("=" * 70)
