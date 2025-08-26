import json
import jsonschema
from pathlib import Path
from types import SimpleNamespace

class ConfigLoader:
    def __init__(self, config_path: str, schema_path: str, logger):
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.logger = logger

    def _dict_to_namespace(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_namespace(i) for i in d]
        else:
            return d

    def load(self) -> object | None:
        if not self.config_path.is_file():
            self.logger.error(f"Errore: il file di configurazione '{self.config_path}' non esiste.")
            return None

        if not self.schema_path.is_file():
            self.logger.error(f"Errore: il file schema '{self.schema_path}' non esiste.")
            return None

        if self.config_path.suffix != ".json" or self.schema_path.suffix != ".json":
            self.logger.error("Errore: entrambi i file devono essere in formato .json")
            return None

        try:
            self.logger.info(f"Caricamento file di configurazione da {self.config_path}")
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            self.logger.info(f"Caricamento file schema da {self.schema_path}")
            with open(self.schema_path, "r") as f:
                schema_data = json.load(f)

            self.logger.info("Validazione configurazione con schema...")
            jsonschema.validate(instance=config_data, schema=schema_data)

        except json.JSONDecodeError as e:
            self.logger.error(f"Errore parsing JSON: {e}")
            return None

        except jsonschema.ValidationError as e:
            self.logger.error(f"Errore di validazione del file di configurazione: {e.message}")
            return None

        except jsonschema.SchemaError as e:
            self.logger.error(f"Errore nel file schema JSON: {e.message}")
            return None

        config_obj = self._dict_to_namespace(config_data)
        self.logger.info("Configurazione caricata e validata correttamente.")
        return config_obj
