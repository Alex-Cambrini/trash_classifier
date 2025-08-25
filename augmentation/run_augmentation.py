import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from augment_dataset import run_full_augmentation
from utils.config_loader import check_and_get_configuration
import logging
from logger import get_logger

def run_augmentation():
    logger = get_logger()
    config_path = Path("./augmentation/config/config.json")
    schema_path = Path("./augmentation/config/schema.json")

    config = check_and_get_configuration(str(config_path), str(schema_path))
    if config is None:
        logger.error("Configurazione non valida, esco.")
        sys.exit(1)
    log_level = logging.DEBUG if config.debug else logging.INFO
    logger.setLevel(level=log_level)
    run_full_augmentation(config)

if __name__ == "__main__":
    run_augmentation()
