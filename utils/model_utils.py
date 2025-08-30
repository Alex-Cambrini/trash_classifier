import sys
import torch
from torch import nn
from typing import Tuple, Dict, Any
from networks import get_net

def create_model(config: Any, num_classes: int, logger: Any) -> Tuple[nn.Module, nn.CrossEntropyLoss, str]:
    """
    Crea il modello e il criterio di loss.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: nn.Module = get_net(config.train_parameters.network_type, num_classes, logger).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    return model, criterion, device

def verify_checkpoint_params(meta: Dict[str, Any], config_params: Dict[str, Any], logger: Any) -> None:
    """
    Verifica che i parametri del checkpoint corrispondano alla configurazione corrente.
    Se ci sono discrepanze, logga l'errore e termina l'esecuzione.
    """
    mismatches = [
        f"{key}: saved={meta.get(key)} current={value}"
        for key, value in config_params.items()
        if meta.get(key) != value
    ]
    if mismatches:
        logger.error(f"Discrepanze tra checkpoint e configurazione corrente: {mismatches}")
        sys.exit(1)

def get_config_params(config: Any) -> Dict[str, Any]:
    """
    Estrae i parametri rilevanti dalla configurazione.
    """
    return {
        "network_type": config.train_parameters.network_type,
        "batch_size": config.hyper_parameters.batch_size,
        "learning_rate": config.hyper_parameters.learning_rate,
        "momentum": config.hyper_parameters.momentum,
        "weight_decay": config.hyper_parameters.weight_decay,
        "scheduler_gamma": config.hyper_parameters.learning_rate_scheduler_gamma,
        "scheduler_step": config.hyper_parameters.scheduler_patience_in_val_steps,
    }
