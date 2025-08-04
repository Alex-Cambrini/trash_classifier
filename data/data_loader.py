import os
import sys
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from logger import get_logger
from utils.data_augmentation import create_augmented_dataset

logger = get_logger()

class DataLoaderManager:
    """
    Gestisce il caricamento e la preparazione dei DataLoader per
    training, validazione e test, inclusa l'eventuale augmentazione dati.
    """

    def __init__(self, cfg):
        """
        Inizializza il manager con la configurazione specificata.

        Args:
            cfg: Oggetto di configurazione contenente parametri di input, training e iperparametri.
        """
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None

    def get_transforms(self):
        """
        Restituisce la pipeline di trasformazioni da applicare sempre in RAM.

        Returns:
            transforms.Compose: trasformazioni standard di ridimensionamento, normalizzazione e conversione in tensor.
        """
        logger.debug("Uso trasformazioni standard")
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        """
        Carica il dataset, applica augmentazione se configurata,
        divide in train, val e test, e crea i DataLoader.

        Genera i DataLoader self.train_loader, self.val_loader e self.test_loader.
        Imposta anche self.classes con le classi del dataset.
        """
        if self.cfg.input.use_augmentation:
            logger.debug("Augmentazione attiva: preparo il dataset aumentato su disco")
            create_augmented_dataset(
                self.cfg.input.dataset_folder,
                self.cfg.input.dataset_folder_augmented,
                self.cfg.input.num_augmented_per_image)
            dataset_folder = self.cfg.input.dataset_folder_augmented
        else:
            logger.debug("Augmentazione disattivata: uso il dataset originale")
            dataset_folder = self.cfg.input.dataset_folder

        transform = self.get_transforms()
        dataset = datasets.ImageFolder(dataset_folder, transform=transform)
        total_size = len(dataset)

        logger.debug(f"Dataset totale dimensione: {total_size}")
        logger.debug(f"Classi trovate: {dataset.classes}")

        train_ratio = self.cfg.train_parameters.train_split
        val_ratio = self.cfg.train_parameters.val_split
        test_ratio = self.cfg.train_parameters.test_split

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "La somma degli split deve essere 1"

        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        logger.debug(f"Split dataset: train={train_size}, val={val_size}, test={test_size}")

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        self.classes = dataset.classes
        batch_size = self.cfg.hyper_parameters.batch_size

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        logger.info("DataLoader creati con successo.")
