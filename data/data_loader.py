import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
from logger import get_logger
from utils.data_augmentation import create_augmented_dataset

logger = get_logger()

class DataLoaderManager:
    """
    Gestisce il caricamento e la preparazione dei DataLoader,
    con logica separata per chiarezza e riutilizzabilità.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None
        self.seed = self.cfg.hyper_parameters.seed
        self.generator = torch.Generator().manual_seed(self.seed)

    def get_transforms(self):
        """Restituisce la pipeline di trasformazioni standard."""
        logger.debug("Uso trasformazioni standard")
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _prepare_datasets(self):
        """
        Gestisce l'augmentazione, carica l'intero dataset e lo divide
        in train, validazione e test.
        """
        # 1. Decide quale cartella usare (originale o aumentata)
        if self.cfg.input.use_augmentation:
            logger.debug("Augmentazione attiva: preparo il dataset aumentato su disco")
            create_augmented_dataset(
                self.cfg.input.dataset_folder,
                self.cfg.input.dataset_folder_augmented,
                self.cfg.input.num_augmented_per_image
            )
            dataset_folder = self.cfg.input.dataset_folder_augmented
        else:
            logger.debug("Augmentazione disattivata: uso il dataset originale")
            dataset_folder = self.cfg.input.dataset_folder

        # 2. Carica l'intero dataset e calcola le dimensioni degli split
        transform = self.get_transforms()
        dataset = datasets.ImageFolder(dataset_folder, transform=transform)
        self.classes = dataset.classes
        total_size = len(dataset)
        train_size = int(total_size * self.cfg.train_parameters.train_split)
        val_size = int(total_size * self.cfg.train_parameters.val_split)
        test_size = total_size - train_size - val_size
        logger.debug(f"Split dataset: train={train_size}, val={val_size}, test={test_size}")

        # 3. Splitta il dataset in modo casuale e ritorna le tre parti
        return random_split(dataset, [train_size, val_size, test_size], generator=self.generator)

    def _create_train_sampler(self, train_dataset):
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

        logger.info("WeightedRandomSampler creato per bilanciare le classi nel training.")
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def load_data(self):
        """
        Metodo principale che prepara i DataLoader per training, validazione e test.
        """
        # 1. Prepara e splitta il dataset
        train_dataset, val_dataset, test_dataset = self._prepare_datasets()

        # 2. Crea il campionatore pesato solo per il training
        train_sampler = self._create_train_sampler(train_dataset)

        # 3. Crea i DataLoader finali
        batch_size = self.cfg.hyper_parameters.batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        logger.info("DataLoader creati con successo.")