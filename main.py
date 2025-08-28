import os
import numexpr
from pathlib import Path
import sys
import logging
from utils.logging_utils import LoggerUtils
from tester import Tester
from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoaderManager
from trainer import Trainer
from utils.logger import flush_temp_logger, get_logger
from utils.dataset_analysis import DatasetAnalyzer
from torch.utils.tensorboard import SummaryWriter
from utils.model_utils import create_model
from utils.run_info import decide_run_name

def main():
    numexpr.set_num_threads(16)

    # --- Logger temporaneo per il caricamento config ---
    temp_logger = get_logger(run_name="temp_log", memory_only=True)
    temp_logger.setLevel(logging.DEBUG)
    temp_logger.info("Avvio script e caricamento configurazione...")

    # --- Caricamento e validazione config ---
    config_path = Path("config/config.json")
    schema_path = Path("config/schema.json")

    loader = ConfigLoader(config_path, schema_path, temp_logger)
    config = loader.load()
   
    if config is None:
        temp_logger.error("Configurazione non valida, esco.")
        sys.exit(1)

    # --- Decidi nome run ---
    run_name = decide_run_name(config, temp_logger)

    log_level = logging.DEBUG if config.parameters.debug else logging.INFO

    logger = flush_temp_logger(run_name, log_level, temp_logger)


    # --- Inizializzazione SummaryWriter ---
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logger.debug(f"SummaryWriter inizializzato in: {log_dir}")

    # --- Inizializzazione LoggerUtils
    logger_utils = LoggerUtils(logger, writer)

    # --- Caricamento dati ---
    data_manager = DataLoaderManager(config, logger)


    # --- Addestramento o Test ---

    logger.info(f"config.parameters.train = {config.parameters.train}")
    logger.info(f"config.parameters.final_test = {config.parameters.final_test}")

    if config.parameters.train and config.parameters.final_test:
        logger.info("Decisione: train + test finale")

        data_manager.load_train_val()
        model, criterion, device = create_model(config, len(data_manager.classes), logger)
        trainer = Trainer(config, data_manager, run_name, logger, writer, logger_utils, model, criterion, device)
        
        # --- Analisi dataset ---
        analyzer = DatasetAnalyzer(config.input.dataset_folder, logger)
        analyzer.analyze_and_report(config.dataset_parameters.min_samples_per_class)
        logger.info("Analisi dataset completata.")

        
        logger.info(f"Numero di epoche: {config.hyper_parameters.epochs}")
        trainer.train()

        data_manager.load_test_only()
        tester = Tester(config, data_manager, logger, writer, logger_utils, model, criterion, device)

        tester.test_model(reload_checkpoint=False, epoch=trainer.current_epoch)

    elif config.parameters.train:
        logger.info("Decisione: solo train")

        data_manager.load_train_val()
        model, criterion, device = create_model(config, len(data_manager.classes), logger)
        trainer = Trainer(config, data_manager, run_name, logger, writer, logger_utils, model, criterion, device)
        # --- Analisi dataset ---
        analyzer = DatasetAnalyzer(config.input.dataset_folder, logger)
        analyzer.analyze_and_report(config.dataset_parameters.min_samples_per_class)
        logger.info("Analisi dataset completata.")

        
        logger.info(f"Numero di epoche: {config.hyper_parameters.epochs}")
        trainer.train()    

    elif config.parameters.final_test and config.parameters.load_model:
        logger.info("Decisione: solo test finale")

        data_manager.load_test_only()
        model, criterion, device = create_model(config, len(data_manager.classes), logger)
        tester = Tester(config, data_manager, logger, writer, logger_utils, model, criterion, device)
        tester.test_model(reload_checkpoint=True)
  
    else:
        logger.warning("Nessuna azione selezionata in config.parameters")


if __name__ == "__main__":
    main()
