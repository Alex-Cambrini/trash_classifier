import logging
from colorama import Fore, Style, init

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
    
def get_logger(level=logging.INFO, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Riduci il logging di librerie rumorose
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logger
