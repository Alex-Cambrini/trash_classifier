import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from augment_dataset import AugmentationRunner
from utils.config_loader import ConfigLoader
import logging
from utils.logger import get_logger
from datetime import datetime

def run_augmentation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name=f"augmentation_{timestamp}"    
    logger = get_logger(run_name=run_name, log_dir="logs/augmentation")
    config_path = Path("./augmentation/config/config.json")
    schema_path = Path("./augmentation/config/schema.json")

    loader = ConfigLoader(config_path, schema_path, logger)
    config = loader.load()

    if config is None:
        logger.error("Configurazione non valida, esco.")
        sys.exit(1)

    log_level = logging.DEBUG if config.debug else logging.INFO
    logger.setLevel(log_level)    
    runner = AugmentationRunner(config, logger, run_name)
    runner.run()

if __name__ == "__main__":
    run_augmentation()
