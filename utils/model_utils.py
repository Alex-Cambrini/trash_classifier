import sys
import torch
from networks import get_net
import torch.nn as nn

def create_model(config, num_classes, logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_net(config.train_parameters.network_type, num_classes, logger).to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion, device

def verify_checkpoint_params(meta, config_params, logger):
    mismatches = [
        f"{key}: saved={meta.get(key)} current={value}"
        for key, value in config_params.items()
        if meta.get(key) != value
    ]
    if mismatches:
        logger.error(f"Discrepanze tra checkpoint e configurazione corrente: {mismatches}")
        sys.exit(1)

def get_config_params(config):
    return {
        "network_type": config.train_parameters.network_type,
        "batch_size": config.hyper_parameters.batch_size,
        "learning_rate": config.hyper_parameters.learning_rate,
        "momentum": config.hyper_parameters.momentum,
        "weight_decay": config.hyper_parameters.weight_decay,
    }