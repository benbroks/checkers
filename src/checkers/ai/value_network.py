import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetworkCNN(nn.Module):
    """
    Convolutional Neural Network for checkers value estimation.

    Input: 4 channels of 8x4 board state (height=8, width=4)
    Output: Single scalar value between -1 and 1
    """

    def __init__(self):
        super(ValueNetworkCNN, self).__init__()

        # Convolutional layers
        # Input: (batch, 4, 8, 4)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        # After conv layers, we still have 8x4 spatial dimensions
        # 128 channels * 8 * 4 = 4096
        self.fc1 = nn.Linear(128 * 8 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Output single value

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 4, 8, 4)

        Returns:
            Output tensor of shape (batch, 1) with values between -1 and 1
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # FC block 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC block 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC block 3 (output layer)
        x = self.fc3(x)
        x = torch.tanh(x)  # Squash output to [-1, 1]

        return x


def create_model():
    """
    Factory function to create a new ValueNetworkCNN model.

    Returns:
        ValueNetworkCNN instance
    """
    return ValueNetworkCNN()


if __name__ == "__main__":
    # Test the model
    model = create_model()
    print(model)

    # Create a sample input
    batch_size = 4
    sample_input = torch.randn(batch_size, 4, 8, 4)

    # Forward pass
    output = model(sample_input)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (batch_size, 1)")
    print(f"Output values (should be in [-1, 1]):")
    print(output)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
