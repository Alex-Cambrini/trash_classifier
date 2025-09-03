import os
import random
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler


class DataLoaderManager:
    """
    Gestisce il caricamento e la preparazione dei DataLoader,
    con logica separata per chiarezza e riutilizzabilità.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None
        self.seed = self.config.hyper_parameters.seed
        self.generator = torch.Generator().manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


    def load_train_val(self) -> None:
        """Crea solo train_loader e val_loader"""
        train_dataset, val_dataset, _ = self._prepare_datasets()
        train_sampler = self._create_train_sampler(train_dataset)
        batch_size = self.config.hyper_parameters.batch_size
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        if len(self.train_loader.dataset) == 0 or len(self.val_loader.dataset) == 0:
            self.logger.error("Train o validation dataset vuoto!")
            sys.exit(1)

        self.logger.debug("Train e Val DataLoader creati con successo.")

    def load_test_only(self) -> None:
        """Crea test_loader e lo salva in self.test_loader"""
        _, _, test_dataset = self._prepare_datasets()
        batch_size = self.config.hyper_parameters.batch_size
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        if len(self.test_loader.dataset) == 0:
            self.logger.warning("Test dataset vuoto!")
        else:
            self.logger.debug("Test DataLoader creato con successo.")

    def _get_transforms(self, mean: list = None, std: list = None) -> transforms.Compose:
        """Restituisce la pipeline di trasformazioni con normalizzazione personalizzata."""
        if mean is None or std is None:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # default ImageNet
        self.logger.debug(f"Uso trasformazioni con mean={mean} std={std}")
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def _prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepara i dataset di training, validazione e test con normalizzazione personalizzata."""
        dataset_folder = self.config.input.dataset_folder
        
        # 1. Controlla che la cartella del dataset esista
        if not os.path.exists(dataset_folder):
            self.logger.error(f"La cartella dataset '{dataset_folder}' non esiste.")
            sys.exit(1)

        # 2. Carica l'intero dataset senza normalizzazione
        raw_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        raw_dataset = datasets.ImageFolder(dataset_folder, transform=raw_transform)
        self.classes = raw_dataset.classes
        total_size = len(raw_dataset)

        # 3. Calcola dimensioni split e divide casualmente il dataset
        train_size = int(total_size * self.config.train_parameters.train_split)
        val_size = int(total_size * self.config.train_parameters.val_split)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            raw_dataset, [train_size, val_size, test_size], generator=self.generator
        )

        # 4. Calcola media e deviazione standard solo sul training set
        mean, std = self._compute_mean_std(train_dataset)

        # 5. Crea trasformazioni con normalizzazione personalizzata
        transform = self._get_transforms(mean.tolist(), std.tolist())

        # 6. Applica le trasformazioni a tutti gli split
        train_dataset.dataset.transform = transform
        val_dataset.dataset.transform = transform
        test_dataset.dataset.transform = transform

        # 7. Restituisce i tre dataset pronti
        return train_dataset, val_dataset, test_dataset

    def _create_train_sampler(self, train_dataset: Dataset) -> Sampler:
        """
        Crea un WeightedRandomSampler per bilanciare le classi nel set di training.
        Questo metodo implementa l'oversampling per le classi minoritarie.
        """
        # 1. Recupera le etichette di classe per tutti i campioni del training set
        targets = [train_dataset.dataset.samples[i][1] for i in train_dataset.indices]

        # 2. Conta quanti campioni ci sono per ogni classe
        class_counts = np.bincount(targets)

        # 3. Calcola i pesi inversi, assegnando un peso maggiore alle classi minoritarie
        class_weights = 1.0 / class_counts

        # 4. Assegna il peso corretto a ogni singolo campione
        samples_weight = class_weights[targets]
        samples_weight = torch.from_numpy(samples_weight).double()

        self.logger.debug(
            "WeightedRandomSampler creato per bilanciare le classi nel training."
        )
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def _compute_mean_std(self, subset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcola mean e std su un subset (es. solo il training set)."""
        loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)
        mean = 0.0
        std = 0.0
        nb_samples = 0
        for data, _ in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
        mean /= nb_samples
        std /= nb_samples
        self.logger.debug(f"Calcolata mean (train set): {mean}, std: {std}")
        return mean, std
