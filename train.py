import torch
import os
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from networks import get_net
from pathlib import Path
from logger import get_logger

logger = get_logger()
class Trainer:
    def __init__(self, config, data_manager, num_classes):
        self.config = config

        #Inizio parametri di configurazione
        #output
        self.model_save_dir = config.output.model_save_dir

        #parameters
        self.load_model = config.parameters.load_model
        self.model_load_path = config.parameters.model_load_path

        #hyper parameters
        self.epochs = config.hyper_parameters.epochs
        self.batch_size = config.hyper_parameters.batch_size
        self.weight_decay = config.hyper_parameters.weight_decay
        self.learning_rate = config.hyper_parameters.learning_rate
        self.momentum = config.hyper_parameters.momentum

        #train parameters
        self.accuracy_eval_every = config.train_parameters.accuracy_evaluation_epochs
        self.accuracy_target = config.train_parameters.accuracy_target
        self.network_type = config.train_parameters.network_type

        #early stop parameters
        self.start_epoch = config.early_stop_parameters.start_epoch
        self.loss_eval_every = config.early_stop_parameters.loss_evaluation_epochs
        self.patience = config.early_stop_parameters.patience
        self.improvement_rate = config.early_stop_parameters.improvement_rate
        #Fine parametri di configurazione

        self.data_manager = data_manager
        self.num_classes = num_classes

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training su {self.device}")

        self.model = get_net(self.network_type, self.num_classes)
        self.model.to(self.device)
        logger.debug(f"Network selected: {self.network_type}")

        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = data_manager.train_loader
        self.val_loader = data_manager.val_loader
        self.test_loader = data_manager.test_loader

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.best_model_state = None
        self.early_stop = False
        self.target_accuracy_reached = False

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(self.train_loader, desc="Training epoch")
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, loader):
        logger.debug("Inizio validazione")
        result = self._evaluate(loader)
        logger.debug("Fine validazione")
        return result

    def test_model(self):
        logger.info("Inizio testing")
        test_loss, test_acc = self._evaluate(self.test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        logger.info("Fine testing")
        return test_loss, test_acc

    def train(self):
        logger.info("Inizio training...")
        logger.info(f"Batch Size: {self.batch_size}")
        self._load_and_resume_training()

        for epoch in range(self.current_epoch, self.epochs):
            if self._should_stop_training():
                break
            
            logger.info(f"Inizio epoca {epoch + 1}/{self.epochs}")
            epoch_loss, epoch_acc = self.train_one_epoch()
            self._log_epoch(epoch_loss, epoch_acc, epoch)

            self.current_epoch = epoch + 1

            if self.current_epoch >= self.start_epoch and (
                (self.current_epoch - self.start_epoch) % self.loss_eval_every == 0 or self.current_epoch == self.epochs
            ):
                val_loss, _ = self.validate(self.val_loader)
                prev_best = self.best_val_loss
                self._check_early_stopping(self.current_epoch, val_loss)

                if self.best_val_loss < prev_best:  # migliora la loss, salva il modello migliore
                    self._save_model_state(os.path.join(self.model_save_dir, "model_best.pth"), self.current_epoch, self.best_model_state)
                    logger.info(f"Miglior modello salvato all'epoca {self.current_epoch}")

            if self.current_epoch % self.accuracy_eval_every == 0 or self.current_epoch == self.epochs:
                self._check_accuracy_target()

        # salva modello finale comunque (stato corrente)
        self._save_model_state(os.path.join(self.model_save_dir, "model_final.pth"), self.current_epoch)
        logger.info("Fine training")



    def _check_early_stopping(self, epoch, val_loss):
        logger.debug(f"Controllo early stopping alle epoca {epoch}")
        if self.best_val_loss - val_loss >= self.improvement_rate:
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.no_improve_count = 0
            logger.debug(f"Loss di validazione migliorata: {val_loss:.4f}")
        else:
            self.no_improve_count += 1
            logger.debug(f"Nessun miglioramento ({self.no_improve_count}/{self.patience})")
            if self.no_improve_count >= self.patience:
                logger.info("Early stopping attivato")
                self.early_stop = True

    def _check_accuracy_target(self):
        _, train_acc = self._evaluate(self.train_loader)
        _, val_acc = self._evaluate(self.val_loader)

        train_acc *= 100
        val_acc *= 100

        logger.info(f"Train Accuracy: {train_acc:.2f}% - Val Accuracy: {val_acc:.2f}%")

        if train_acc > self.accuracy_target and val_acc > self.accuracy_target:
            logger.info("Target accuracy raggiunta. Interrompo il training.")
            self.target_accuracy_reached = True

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return total_loss / total, correct / total
        
    def _save_model_state(self, path: str, epoch: int, model_state=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            state_to_save = model_state if model_state is not None else self.model.state_dict()
            torch.save({
                'model_state_dict': state_to_save,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch + 1,
            }, path)
            logger.info(f"Stato del modello e dell'ottimizzatore salvato in: {path}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio dello stato del modello: {e}")


    def _load_model_state(self, path: str):
        path_obj = Path(path)
        if not path_obj.is_file():
            logger.error(f"Model state file not found: {path}")
            return False
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch')
            if self.current_epoch is None:
                logger.error("Checkpoint non contiene 'epoch'. Potrebbe causare problemi nel resume del training.")
            logger.info(f"Model and optimizer state loaded from: {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            return False
  
    def _load_and_resume_training(self):
        self.current_epoch = 0  # default se non carichi modello
        logger.debug(f"Parametro load_model: {self.load_model}")
        if self.load_model and self.model_load_path and self._load_model_state(self.model_load_path):
            logger.info("Modello caricato correttamente, riprendo il training")
        else:
            logger.info("Nessun modello caricato, inizio training da zero")

    def _should_stop_training(self):
        if self.early_stop:
            logger.info("Training interrotto per early stopping.")
            return True
        if self.target_accuracy_reached:
            logger.info("Training interrotto per raggiungimento accuracy target.")
            return True
        return False
    
    def _log_epoch(self, loss, acc, epoch):
        logger.info(f"Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")
