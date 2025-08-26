import time
import numexpr
from pathlib import Path
import sys
import logging

from utils.config_loader import ConfigLoader
from data.data_loader import DataLoaderManager
from train import Trainer
from logger import get_logger
from utils.dataset_analysis import DatasetAnalyzer
from run_info import get_run_name

def main():
    numexpr.set_num_threads(16)

    # --- Logger temporaneo per il caricamento config ---
    temp_logger = get_logger(run_name="temp_log", level=logging.DEBUG, memory_only=True)
    temp_logger.info("Avvio script e caricamento configurazione...")

    # --- Caricamento e validazione config ---
    config_path = Path("config/config.json")
    schema_path = Path("config/schema.json")

    loader = ConfigLoader(config_path, schema_path, temp_logger)
    config = loader.load()
   
    if config is None:
        temp_logger.error("Configurazione non valida, esco.")
        sys.exit(1)

    # --- Caricamento dati ---
    data_manager = DataLoaderManager(config, temp_logger)
    data_manager.load_data()
    num_classes = len(data_manager.classes)
    if data_manager.train_loader is None:
        temp_logger.error("Errore: caricamento dati fallito. Esco.")
        sys.exit(1)
    temp_logger.info("Dati caricati correttamente.")

    # --- Analisi dataset ---
    analyzer = DatasetAnalyzer(config.input.dataset_folder, temp_logger)
    analyzer.analyze_and_report(config.dataset_parameters.min_samples_per_class)
    temp_logger.info("Analisi dataset completata.")

    # --- Addestramento o Test ---
    trainer = Trainer(config, data_manager, num_classes, temp_logger)
    logger = get_logger(run_name=get_run_name())

    if config.parameters.train and config.parameters.final_test:
        logger.info(f"Numero di epoche: {config.hyper_parameters.epochs}")
        trainer.train()
        trainer.test_model(use_current_model=True)
    elif config.parameters.train:
        trainer.train()
    elif config.parameters.test:
        trainer.test_model()


if __name__ == "__main__":
    main()
