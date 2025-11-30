"""
Training loop for value network.

This module provides functions for training the ValueNetworkCNN model on self-play
generated games to predict game outcomes using regression.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from checkers.ai.value_network import create_model
from checkers.ai.value_network_dataset import CheckersValueNetworkDataset
from checkers.ai.value_network_config import ValueNetworkConfig


def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch.

    Args:
        model: ValueNetworkCNN model
        train_loader: DataLoader for training data
        optimizer: Optimizer (AdamW)
        device: Device to train on (cpu/mps/cuda)
        config: ValueNetworkConfig instance

    Returns:
        tuple: (average_loss, mean_abs_error, sign_accuracy) for the epoch
    """
    model.train()
    epoch_loss = 0.0
    total_mae = 0.0
    correct_sign = 0
    total = 0

    for batch_idx, (states, values) in enumerate(train_loader):
        states = states.to(device)  # (batch, 4, 8, 4)
        values = values.to(device)  # (batch, 1)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(states)  # (batch, 1) in [-1, 1]
        # INSERT_YOUR_CODE
        # Print first 20 items of predictions and values for inspection
        # print("Predictions (first 20):", predictions[:20].flatten().cpu().detach().numpy())
        # print("Targets     (first 20):", values[:20].flatten().cpu().detach().numpy())

        # MSE Loss
        loss = F.mse_loss(predictions, values)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        total_mae += torch.abs(predictions - values).sum().item()

        # Sign accuracy (same sign as target)
        pred_sign = torch.sign(predictions)
        true_sign = torch.sign(values)
        correct_sign += (pred_sign == true_sign).sum().item()
        total += values.size(0)

        # Log progress
        if batch_idx % config.log_interval == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'MAE: {torch.abs(predictions - values).mean().item():.4f}')

    avg_loss = epoch_loss / len(train_loader)
    mae = total_mae / total
    sign_acc = 100. * correct_sign / total
    return avg_loss, mae, sign_acc


def validate(model, val_loader, device):
    """Validate the model.

    Args:
        model: ValueNetworkCNN model
        val_loader: DataLoader for validation data
        device: Device to validate on (cpu/mps/cuda)

    Returns:
        tuple: (average_loss, mean_abs_error, sign_accuracy) for validation set
    """
    model.eval()
    val_loss = 0.0
    total_mae = 0.0
    correct_sign = 0
    total = 0

    with torch.no_grad():
        for states, values in val_loader:
            states = states.to(device)
            values = values.to(device)

            # Forward pass
            predictions = model(states)

            # Compute loss
            loss = F.mse_loss(predictions, values)
            val_loss += loss.item()

            # Track metrics
            total_mae += torch.abs(predictions - values).sum().item()

            # Sign accuracy
            pred_sign = torch.sign(predictions)
            true_sign = torch.sign(values)
            correct_sign += (pred_sign == true_sign).sum().item()
            total += values.size(0)

    avg_loss = val_loss / len(val_loader)
    mae = total_mae / total
    sign_acc = 100. * correct_sign / total
    return avg_loss, mae, sign_acc


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Save model checkpoint.

    Args:
        model: ValueNetworkCNN model
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


def create_data_loaders(dataset_cache_path, batch_size, val_split, num_workers):
    """Create train and validation data loaders.

    Args:
        dataset_cache_path: Path to cached dataset file
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load pre-generated dataset from cache
    dataset = CheckersValueNetworkDataset(
        max_games=10000,
        dataset_cache_path=dataset_cache_path
    )

    # Split into train and validation with fixed seed for reproducibility
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # MPS doesn't support pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader


def train(config: ValueNetworkConfig):
    """Main training function.

    Args:
        config: ValueNetworkConfig instance with all hyperparameters
    """
    # Setup device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Create model
    model = create_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create data loaders
    print(f"\nLoading dataset from {config.dataset_cache_path}...")
    train_loader, val_loader = create_data_loaders(
        config.dataset_cache_path,
        config.batch_size,
        config.val_split,
        config.num_workers
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
        train_loss, train_mae, train_sign_acc = train_epoch(
            model, train_loader, optimizer, device, config
        )

        # Validate
        val_loss, val_mae, val_sign_acc = validate(model, val_loader, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss:     {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
              f"Train Sign Acc: {train_sign_acc:.2f}%")
        print(f"  Val Loss:       {val_loss:.4f}, Val MAE:   {val_mae:.4f}, "
              f"Val Sign Acc:   {val_sign_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('SignAccuracy/train', train_sign_acc, epoch)
        writer.add_scalar('SignAccuracy/val', val_sign_acc, epoch)
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
