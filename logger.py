import logging
from logging.handlers import MemoryHandler
from colorama import Fore, Style, init
from pathlib import Path

init(autoreset=True)

COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.WHITE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, Fore.WHITE)
        msg = super().format(record)
        return f"{color}{msg}{Style.RESET_ALL}"

def get_logger(run_name, level=logging.INFO, log_dir="logs", memory_only=False):
    logger = logging.getLogger(run_name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        # StreamHandler colorato per il terminale
        ch = logging.StreamHandler()
        ch.setFormatter(ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(ch)

        if memory_only:
            # Logger temporaneo in memoria
            mem_handler = MemoryHandler(capacity=10000, flushLevel=logging.CRITICAL)
            mem_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(mem_handler)
            logger.memory_handler = mem_handler  # salva riferimento per flush successivo
        else:
            # FileHandler normale
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(log_dir) / f"{run_name}.log"
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)

    # Riduci il logging di librerie rumorose
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logger
