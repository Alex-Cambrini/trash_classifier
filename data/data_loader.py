import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from logger import get_logger

logger = get_logger()

class DataLoaderManager:
    def __init__(self, cfg):
        """
        Inizializza il manager con la configurazione.
        """
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None

    def get_transforms(self):
        """
        Trasformazioni base: resize, tensor, normalizzazione.
        """
        logger.debug("Definisco trasformazioni immagini")
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        """
        Carica immagini da dataset_folder e divide in train/val/test.
        Crea i DataLoader.
        """
        logger.debug(f"Caricamento dataset da: {self.cfg.input.dataset_folder}")

        transform = self.get_transforms()
        dataset = datasets.ImageFolder(self.cfg.input.dataset_folder, transform=transform)
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
        logger.debug(f"Batch size: {batch_size}")

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        logger.info("DataLoader creati con successo.")
