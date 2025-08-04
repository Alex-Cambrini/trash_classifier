from .resnet18 import get_resnet18
from logger import get_logger

logger = get_logger()

def get_net(name, num_classes):
    networks = {
        "resnet18": get_resnet18,
    }
    if name not in networks:
        logger.error(f"Unknown network requested: {name}")
        raise ValueError(f"Unknown network: {name}")
    return networks[name](num_classes)
