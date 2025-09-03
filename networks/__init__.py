import logging
import torch.nn as nn
from networks.efficientnet_b0 import get_efficientnet_b0
from networks.resnet18 import get_resnet18
from networks.custom_net import get_custom_cnn

def get_net(name: str, num_classes: int, logger: logging.Logger) -> nn.Module:
    """Restituisce il modello richiesto per nome."""
    networks = {
        "resnet18": get_resnet18,
        "efficientnet_b0": get_efficientnet_b0,
        "custom_cnn": get_custom_cnn,
    }
    if name not in networks:
        logger.error(f"Unknown network requested: {name}")
        raise ValueError(f"Unknown network: {name}")
    return networks[name](num_classes)
