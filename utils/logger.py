import logging
from logging.handlers import MemoryHandler
from colorama import Fore, Style, init
from pathlib import Path

init(autoreset=True)

LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.WHITE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
}

class ColorFormatter(logging.Formatter):
    """Formatter che colora i messaggi secondo il livello."""
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, Fore.WHITE)
        msg = super().format(record)
        return f"{color}{msg}{Style.RESET_ALL}"


def get_logger(run_name: str, log_dir: str = "logs", memory_only: bool = False) -> logging.Logger:
    """Crea e ritorna un logger. Usa file o memoria a seconda di memory_only."""
    logger = logging.getLogger(run_name)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setFormatter(ColorFormatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(ch)

        if memory_only:
            # Logger temporaneo in memoria
            mem_handler = MemoryHandler(capacity=10000, flushLevel=logging.CRITICAL)
            mem_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            logger.addHandler(mem_handler)
            logger.memory_handler = mem_handler  # salva riferimento per flush successivo
        else:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(log_dir) / f"{run_name}.log"
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            logger.addHandler(fh)

    # Riduci il logging di librerie rumorose
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logger


def flush_temp_logger(run_name: str, log_level: int, temp_logger: logging.Logger) -> logging.Logger:
    """Trasferisce i log temporanei in memoria sul file e ritorna il logger aggiornato."""
    logger = get_logger(run_name)
    logger.setLevel(log_level)

    # Se esiste un MemoryHandler nel logger temporaneo, flusha tutto sul FileHandler
    if hasattr(temp_logger, "memory_handler"):
        mem_handler: MemoryHandler = temp_logger.memory_handler 
        Path("logs").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(f"logs/{run_name}.log", mode="a")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # Imposta come target e flush
        mem_handler.setTarget(fh)
        mem_handler.flush()

        # Rimuove il MemoryHandler dal logger temporaneo per evitare doppie scritture
        temp_logger.removeHandler(mem_handler)
        del temp_logger.memory_handler

    return logger