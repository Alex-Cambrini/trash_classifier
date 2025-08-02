from config.config_loader import check_and_get_configuration
from data.data_loader import DataLoaderManager
from pathlib import Path
from train import Trainer
import sys
import logger

def main():
    config_path = Path("config/config.json")
    schema_path = Path("config/schema.json")

    config = check_and_get_configuration(str(config_path), str(schema_path))
    if config is None:
        print("Errore: configurazione non valida")
        sys.exit(1)

    logger.set_debug(config.parameters.debug)
    logger.log("Configurazione caricata correttamente.")
    
    data_manager = DataLoaderManager(config)
    data_manager.load_data()
    num_classes = len(data_manager.classes)
    if data_manager.train_loader is None:
        logger.log("Errore: caricamento dati fallito. Esco.", level="error")
        sys.exit(1)
    logger.log("Dati caricati correttamente.", level="info")

    logger.log(f"Parametro train: {config.parameters.train}", level="info")
    if config.parameters.train:
        logger.log(f"Numero di epoche: {config.hyper_parameters.epochs}", level="info") 
        trainer = Trainer(config, data_manager, num_classes)
        trainer.train()

if __name__ == "__main__":
    main()
