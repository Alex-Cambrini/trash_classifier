import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        super(CustomCNN, self).__init__()

        # -------------------------
        # Feature extraction layers
        # -------------------------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # -------------------------
        # Adaptive pooling per dimension fissa
        # -------------------------
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # -------------------------
        # Fully connected layers
        # -------------------------
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # -------------------------
        # Feature extraction
        # -------------------------
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # -------------------------
        # Adaptive pooling e flatten
        # -------------------------
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # -------------------------
        # Fully connected layers
        # -------------------------
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Funzione getter esterna
def get_custom_cnn(num_classes: int) -> nn.Module:
        return CustomCNN(num_classes=num_classes)