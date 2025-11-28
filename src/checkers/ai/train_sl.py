"""
Training loop for supervised learning.

This module provides functions for training the CheckersCNN model on supervised
learning tasks using action prediction from expert games.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from checkers.ai.sl_action_policy import create_model
from checkers.ai.dataset import CheckersDataset
from checkers.ai.config import TrainingConfig


def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch.

    Args:
        model: CheckersCNN model
        train_loader: DataLoader for training data
        optimizer: Optimizer (AdamW)
        device: Device to train on (cpu/mps/cuda)
        config: TrainingConfig instance

    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (states, actions) in enumerate(train_loader):
        states = states.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(states)  # (batch, 32, 8)
        outputs_flat = outputs.view(outputs.size(0), -1)  # (batch, 256)
        targets_flat = actions.view(actions.size(0), -1).argmax(dim=1)  # (batch,)

        # Compute loss
        loss = F.cross_entropy(outputs_flat, targets_flat)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # Track statistics
        epoch_loss += loss.item()
        _, predicted = outputs_flat.max(1)
        total += targets_flat.size(0)
        correct += predicted.eq(targets_flat).sum().item()

        # Log progress
        if batch_idx % config.log_interval == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate the model.

    Args:
        model: CheckersCNN model
        val_loader: DataLoader for validation data
        device: Device to validate on (cpu/mps/cuda)

    Returns:
        tuple: (average_loss, accuracy) for validation set
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(device)
            actions = actions.to(device)

            # Forward pass
            outputs = model(states)
            outputs_flat = outputs.view(outputs.size(0), -1)
            targets_flat = actions.view(actions.size(0), -1).argmax(dim=1)

            # Compute loss
            loss = F.cross_entropy(outputs_flat, targets_flat)
            val_loss += loss.item()

            # Track accuracy
            _, predicted = outputs_flat.max(1)
            total += targets_flat.size(0)
            correct += predicted.eq(targets_flat).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Save model checkpoint.

    Args:
        model: CheckersCNN model
        optimizer: Optimizer state
        epoch: Current epoch number
        val_loss: Validation loss at this checkpoint
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, filepath)


def create_data_loaders(pdn_files, batch_size, val_split, num_workers, max_games=None):
    """Create train and validation data loaders.

    Args:
        pdn_files: List of PDN file paths
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        max_games: Maximum number of games to load (None = all)

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create full dataset
    dataset = CheckersDataset(pdn_files, max_games=max_games)

    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Set to True if using CUDA
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader


def train(config: TrainingConfig):
    """Main training function.

    Args:
        config: TrainingConfig instance with all hyperparameters
    """
    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Create model
    model = create_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create data loaders
    print(f"\nLoading data from {len(config.pdn_files)} PDN file(s)...")
    train_loader, val_loader = create_data_loaders(
        config.pdn_files,
        config.batch_size,
        config.val_split,
        config.num_workers,
        config.max_games
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_lr
    )

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for up to {config.num_epochs} epochs...")
    print(f"Early stopping patience: {config.early_stopping_patience} epochs")
    print("=" * 70)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, config)

        # Validate
        val_loss, val_acc = validate(model, val_loader, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            os.path.join(config.checkpoint_dir, 'latest_model.pth')
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config.checkpoint_dir, 'best_model.pth')
            )
            print(f"  ✓ New best model saved! (Val loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*70}")
            break

    # Training complete
    writer.close()
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"TensorBoard logs saved to: {config.log_dir}")
    print(f"{'='*70}\n")
