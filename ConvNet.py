############################################
### Code for building the baseline model ###
############################################

import torch.nn as nn


class ConvNet(nn.Module):
    """ConvNet with 3 convolutional layers and 2 fully connected layers"""

    def __init__(self, dropout=0, batchnorm=False):
        """Initialise the model

        :param dropout: dropout rate for the model
        :param batchnorm: boolean to determine if batchnorm is used
        """
        super(ConvNet, self).__init__()
        self.batchnorm = batchnorm

        # Layer 1: C(55 × 55 × 96)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            self.batch_norm(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Layer 2: C(27 × 27 × 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            self.batch_norm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Layer 3: C(13 × 13 × 384)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            self.batch_norm(384),
            nn.ReLU(),
        )
        # Layer 4: C(13 × 13 × 384)
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            self.batch_norm(384),
            nn.ReLU(),
        )
        # Layer 5: C(13 × 13 × 256)
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            self.batch_norm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Layer 6: F(4096)
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            self.batch_norm(4096, linear=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Layer 7: F(4096)
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            self.batch_norm(4096, linear=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Layer 8: F(28)
        self.fc3 = nn.Linear(4096, 28)

    def batch_norm(self, x, linear=False):
        """Return a batchnorm layer if batchnorm is True, otherwise return an identity layer

        :param x: input to the batchnorm layer
        :param linear: boolean to determine if the layer is linear
        """
        if self.batchnorm:
            if linear:
                return nn.BatchNorm1d(x)
            else:
                return nn.BatchNorm2d(x)
        return nn.Identity(x)

    def forward(self, x):
        """Forward pass through the network

        :param x: input tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
