from config.config_loader import check_and_get_configuration
from data.data_loader import DataLoaderManager
from pathlib import Path
from train import Trainer
import sys
import logging
from logger import get_logger

def main():
    config_path = Path("config/config.json")
    schema_path = Path("config/schema.json")

    config = check_and_get_configuration(str(config_path), str(schema_path))
    if config is None:
        print("Errore: configurazione non valida")
        sys.exit(1)

    log_level = logging.DEBUG if config.parameters.debug else logging.INFO
    logger = get_logger(level=log_level)

    logger.info("Configurazione caricata correttamente.")
    
    data_manager = DataLoaderManager(config)
    data_manager.load_data()
    num_classes = len(data_manager.classes)
    if data_manager.train_loader is None:
        logger.error("Errore: caricamento dati fallito. Esco.")
        sys.exit(1)
    logger.info("Dati caricati correttamente.")

    logger.info(f"Parametro train: {config.parameters.train}")
    if config.parameters.train:
        logger.info(f"Numero di epoche: {config.hyper_parameters.epochs}")
        trainer = Trainer(config, data_manager, num_classes)
        trainer.train()

if __name__ == "__main__":
    main()
