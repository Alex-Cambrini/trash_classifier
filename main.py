from config.config_loader import check_and_get_configuration
from data.data_loader import DataLoaderManager
from pathlib import Path
from train import Trainer
import sys
import logging
from logger import get_logger

def main():
    logger = get_logger()
    config_path = Path("config/config.json")
    schema_path = Path("config/schema.json")

    config = check_and_get_configuration(str(config_path), str(schema_path))
    if config is None:
        logger.error("Configurazione non valida, esco.")
        sys.exit(1)

    log_level = logging.DEBUG if config.parameters.debug else logging.INFO
    logger.setLevel(level=log_level)
    
    data_manager = DataLoaderManager(config)
    data_manager.load_data()
    num_classes = len(data_manager.classes)
    if data_manager.train_loader is None:
        logger.error("Errore: caricamento dati fallito. Esco.")
        sys.exit(1)
    logger.info("Dati caricati correttamente.")

    if config.parameters.train or config.parameters.test:
        trainer = Trainer(config, data_manager, num_classes)

        logger.debug(f"Parametro train: {config.parameters.train}")
        if config.parameters.train:
            logger.info(f"Numero di epoche: {config.hyper_parameters.epochs}")
            trainer.train()

        logger.debug(f"Parametro test: {config.parameters.test}")
        if config.parameters.test:
            trainer.test_model()

if __name__ == "__main__":
    main()
