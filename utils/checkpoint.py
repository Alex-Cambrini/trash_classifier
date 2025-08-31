import torch
from pathlib import Path
import logging


def read_checkpoint(path, logger: logging.Logger):
    """Legge un checkpoint salvato su disco e ritorna i suoi contenuti principali."""
    path_obj = Path(path)
    if not path_obj.is_file():
        logger.error(f"Checkpoint file non trovato: {path}")
        return None

    try:
        checkpoint = torch.load(path, weights_only=False)
        logger.info("Checkpoint caricato con successo")

        return {
            "model_state_dict": checkpoint.get("model_state_dict"),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
            "metrics": checkpoint.get("metrics", {}),
            "meta": checkpoint.get("meta", {}),
        }

    except Exception as e:
        logger.error(f"Errore lettura checkpoint: {e}")
        return None
