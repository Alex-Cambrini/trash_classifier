import logging
import os
import sys
from typing import Dict

class DatasetAnalyzer:
    """
    Analizza un dataset organizzato in cartelle per classe.
    Mantiene controllo minimo campioni per classe e log dello sbilanciamento.
    Compatibile con oversampling: analisi fatta solo sul dataset originale.
    """
    def __init__(self, dataset_path: str, logger: logging.Logger):
        self.dataset_path = dataset_path
        self.class_counts = {}
        self.logger = logger

    def analyze_and_report(self, min_samples: int) -> Dict[str, int]:
        """Esegue l'analisi del dataset e controlla min_samples/log ratio."""
        self._analyze()
        self._report(min_samples)
        return self.class_counts

    def _analyze(self) -> None:
        """Conta il numero di campioni per ciascuna classe."""
        if not os.path.exists(self.dataset_path):
            self.logger.error(f"Path non trovato: {self.dataset_path}")
            raise FileNotFoundError(f"Path non trovato: {self.dataset_path}")

        self.class_counts = {}
        for class_name in os.listdir(self.dataset_path):
            class_dir = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_dir):
                num_files = len([
                    f for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ])
                self.class_counts[class_name] = num_files

        self.logger.debug(f"Dataset analizzato: {len(self.class_counts)} classi trovate.")

    def _report(self, min_samples: int) -> None:
        """
        Controlla che ogni classe abbia almeno min_samples campioni
        e logga informazioni sullo sbilanciamento del dataset.
        """
        if not self.class_counts:
            self.logger.error("Prima esegui _analyze() per popolare i dati.")
            raise ValueError("Prima esegui _analyze() per popolare i dati.")

        total_samples = sum(self.class_counts.values())
        for class_name, count in self.class_counts.items():
            perc = (count / total_samples) * 100 if total_samples > 0 else 0
            self.logger.info(f"  Classe '{class_name}': {count} campioni ({perc:.2f}%)")
            if count < min_samples:
                self.logger.error(f"Classe '{class_name}' ha troppo pochi campioni ({count} < {min_samples})!")
                sys.exit(-1)

        max_count = max(self.class_counts.values())
        min_count = min(self.class_counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')

        if ratio <= 10:
            self.logger.info("Bilanciamento classi OK.")
        elif ratio <= 50:
            self.logger.warning("Attenzione: dataset sbilanciato (rapporto max/min > 10).")
        else:
            self.logger.error("Dataset troppo sbilanciato, training non consigliato!")
            sys.exit(-1)