import torch
import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self, num_classes=42):  # Assuming 42 outputs
        super(CustomNet, self).__init__()

        # Layer 1: C(55 × 55 × 96)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
        )
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2: C(27 × 27 × 256)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3: C(13 × 13 × 384)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        # Layer 4: C(13 × 13 × 384)
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        # Layer 5: C(13 × 13 × 256)
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)  # Flatten after pooling
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout (if needed)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.conv1(x))  # C(55 × 55 × 96)
        x = self.lrn1(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))  # C(27 × 27 × 256)
        x = self.lrn2(x)
        x = self.pool2(x)

        x = self.relu(self.conv3(x))  # C(13 × 13 × 384)
        x = self.relu(self.conv4(x))  # C(13 × 13 × 384)
        x = self.relu(self.conv5(x))  # C(13 × 13 × 256)
        x = self.pool3(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))  # F(4096)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  # F(4096)
        x = self.dropout(x)
        x = self.fc3(x)  # Output

        return x


# Example usage:
model = CustomNet(num_classes=42)  # For pose estimation with 42 outputs
