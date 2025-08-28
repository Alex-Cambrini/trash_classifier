from pathlib import Path
import sys
import time
import torch


def decide_run_name(config, temp_logger) -> str:
    """Decide il nome della run in base al checkpoint o crea uno nuovo."""
    load_model = config.parameters.load_model
    model_load_path = config.parameters.model_load_path
    network_type = config.train_parameters.network_type

    if load_model and model_load_path:
        path_obj = Path(model_load_path)
        if not path_obj.is_file():
            temp_logger.error(f"Model file non trovato: {model_load_path}")
            sys.exit(1)
        checkpoint = torch.load(path_obj)
        run_name = checkpoint.get("meta", {}).get("run_name")
        if run_name is None:
            temp_logger.error(f"run_name mancante nel checkpoint: {model_load_path}")
            sys.exit(1)
    else:
        run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{network_type}"


    return run_name

