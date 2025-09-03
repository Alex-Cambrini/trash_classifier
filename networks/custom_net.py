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
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Nuovi blocchi
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout / 2)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # -------------------------
        # Feature extraction con BatchNorm
        # -------------------------
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x))); x = self.pool5(x)
        x = F.relu(self.bn6(self.conv6(x))); x = self.pool6(x)

        # -------------------------
        # Adaptive pooling e flatten
        # -------------------------
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # -------------------------
        # Fully connected layers
        # -------------------------
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def get_custom_cnn(num_classes: int) -> nn.Module:
    return CustomCNN(num_classes=num_classes)
