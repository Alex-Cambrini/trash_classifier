from datetime import datetime
from colorama import Fore, Style

_debug = False

_levels = {
    "debug": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "critical": 4
}

_colors = {
    "debug": Fore.CYAN,
    "info": Fore.WHITE,
    "warning": Fore.YELLOW,
    "error": Fore.RED,
    "critical": Fore.MAGENTA + Style.BRIGHT
}

def set_debug(value: bool):
    global _debug
    _debug = value

def log(msg, level="info"):
    level = level.lower()
    if _levels.get(level, 1) >= (0 if _debug else 1):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = _colors.get(level, Fore.WHITE)
        print(f"{color}[{timestamp}] [{level.upper()}] {msg}{Style.RESET_ALL}")
