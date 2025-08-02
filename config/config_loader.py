import json
import jsonschema
from pathlib import Path
from types import SimpleNamespace
import logger

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
        logger.log(f"Errore: il file di configurazione '{config_path}' non esiste.", "error")
        return None

    if not schema_file.is_file():
        logger.log(f"Errore: il file schema '{schema_path}' non esiste.", "error")
        return None

    if config_file.suffix != ".json" or schema_file.suffix != ".json":
        logger.log("Errore: entrambi i file devono essere in formato .json", "error")
        return None

    try:
        logger.log(f"Caricamento file di configurazione da {config_path}", "debug")
        with open(config_file, "r") as f:
            config_data = json.load(f)

        logger.log(f"Caricamento file schema da {schema_path}", "debug")
        with open(schema_file, "r") as f:
            schema_data = json.load(f)

        logger.log("Validazione configurazione con schema...", "debug")
        jsonschema.validate(instance=config_data, schema=schema_data)

    except json.JSONDecodeError as e:
        logger.log(f"Errore parsing JSON: {e}", "error")
        return None

    except jsonschema.ValidationError as e:
        logger.log(f"Errore di validazione del file di configurazione: {e.message}", "error")
        return None

    except jsonschema.SchemaError as e:
        logger.log(f"Errore nel file schema JSON: {e.message}", "error")
        return None

    config_obj = _dict_to_namespace(config_data)
    logger.log("Configurazione caricata e validata correttamente.", "info")
    return config_obj
