import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights

def get_resnet18(num_classes: int, pretrained: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
