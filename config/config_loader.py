import json
import jsonschema
from pathlib import Path
from types import SimpleNamespace
from logger import get_logger

logger = get_logger()

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    else:
        return d

def check_and_get_configuration(config_path: str, schema_path: str) -> object | None:
    config_file = Path(config_path)
    schema_file = Path(schema_path)

    if not config_file.is_file():
        logger.error(f"Errore: il file di configurazione '{config_path}' non esiste.")
        return None

    if not schema_file.is_file():
        logger.error(f"Errore: il file schema '{schema_path}' non esiste.")
        return None

    if config_file.suffix != ".json" or schema_file.suffix != ".json":
        logger.error("Errore: entrambi i file devono essere in formato .json")
        return None

    try:
        logger.info(f"Caricamento file di configurazione da {config_path}")
        with open(config_file, "r") as f:
            config_data = json.load(f)

        logger.info(f"Caricamento file schema da {schema_path}")
        with open(schema_file, "r") as f:
            schema_data = json.load(f)

        logger.info("Validazione configurazione con schema...")
        jsonschema.validate(instance=config_data, schema=schema_data)
        
        if config_data["early_stop_parameters"]["start_epoch"] > config_data["hyper_parameters"]["epochs"]:
            logger.error("start_epoch non può essere maggiore di epochs.")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Errore parsing JSON: {e}")
        return None

    except jsonschema.ValidationError as e:
        logger.error(f"Errore di validazione del file di configurazione: {e.message}")
        return None

    except jsonschema.SchemaError as e:
        logger.error(f"Errore nel file schema JSON: {e.message}")
        return None

    config_obj = _dict_to_namespace(config_data)
    logger.info("Configurazione caricata e validata correttamente.")
    return config_obj