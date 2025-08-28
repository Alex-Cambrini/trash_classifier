import torch
from pathlib import Path
import logging

def read_checkpoint(path, logger: logging.Logger):
    path_obj = Path(path)
    if not path_obj.is_file():
        logger.error(f"Checkpoint file non trovato: {path}")
        return None

    try:
        checkpoint = torch.load(path)
        logger.info("Checkpoint caricato con successo")

        return {
            "model_state_dict": checkpoint.get("model_state_dict"),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "meta": checkpoint.get("meta", {})
        }

    except Exception as e:
        logger.error(f"Errore lettura checkpoint: {e}")
        return None


