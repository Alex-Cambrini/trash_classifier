from networks.efficientnet_b0 import get_efficientnet_b0
from networks.resnet18 import get_resnet18


def get_net(name, num_classes, logger):
    networks = {
        "resnet18": get_resnet18,
        "efficientnet_b0": get_efficientnet_b0,
    }
    if name not in networks:
        logger.error(f"Unknown network requested: {name}")
        raise ValueError(f"Unknown network: {name}")
    return networks[name](num_classes)
