import torch
import torch.nn as nn
import torch.nn.functional as F

class DefaultImageClassificationNet(nn.Module):
    """
    A default convolutional neural network for image classification.

    Args:
    - n_classes (int): Number of classes for the classification task.

    Attributes:
    - pool (nn.MaxPool2d): Max pooling layer.
    - conv1 (nn.Conv2d): First convolutional layer.
    - batch1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
    - conv2 (nn.Conv2d): Second convolutional layer.
    - batch2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
    - dropout1 (nn.Dropout): Dropout layer after the second convolutional layer.
    - conv3 (nn.Conv2d): Third convolutional layer.
    - batch3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.
    - conv4 (nn.Conv2d): Fourth convolutional layer.
    - batch4 (nn.BatchNorm2d): Batch normalization for the fourth convolutional layer.
    - dropout2 (nn.Dropout): Dropout layer after the fourth convolutional layer.
    - conv5 (nn.Conv2d): Fifth convolutional layer.
    - batch5 (nn.BatchNorm2d): Batch normalization for the fifth convolutional layer.
    - conv6 (nn.Conv2d): Sixth convolutional layer.
    - batch6 (nn.BatchNorm2d): Batch normalization for the sixth convolutional layer.
    - dropout3 (nn.Dropout): Dropout layer after the sixth convolutional layer.
    - fc1 (nn.Linear): Fully connected layer.
    - batch7 (nn.BatchNorm1d): Batch normalization for the fully connected layer.
    - dropout4 (nn.Dropout): Dropout layer after the fully connected layer.
    - out (nn.Linear): Output layer for classification.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the network.

    """

    def __init__(self, n_classes: int) -> None:
        super(DefaultImageClassificationNet, self).__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.batch4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(64, 128, 3)
        self.batch5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.batch6 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 24 * 24, 128)
        self.batch7 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, n_classes).
        """
        
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = F.relu(self.conv4(x))
        x = self.batch4(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.batch5(x)
        x = F.relu(self.conv6(x))
        x = self.batch6(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 24 * 24)

        x = F.relu(self.fc1(x))
        x = self.batch7(x)
        x = self.dropout4(x)
        x = self.out(x)

        return x
