import logging
import sys
import time
import numpy as np
import torch
import os
import copy
import torch.nn as nn
import torch.optim as optim
from run_info import set_run_name, get_run_name
from tqdm import tqdm
from logging_utils import LoggerUtils
from networks import get_net
from pathlib import Path
from logger import get_logger
from torch.utils.tensorboard import SummaryWriter
from metrics import (
    compute_confusion_matrix_metric,
    compute_loss,
    compute_accuracy,
    compute_per_class_accuracy,
    compute_precision_recall_f1
)


class Trainer:
    def __init__(self, config, data_manager, num_classes, temp_logger):
        self.config = config
        self.data_manager = data_manager
        self.num_classes = num_classes
        self.network_type = self.config.train_parameters.network_type
        self.temp_logger = temp_logger

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # --- Configurazioni principali ---
        self._init_model()
        self._init_optim_scheduler()
        self._init_params(config)

        # --- Decido run_name ---
        self._decide_run_name()

        #--- Logger definitivo ---
        self._flush_temp_logger()
       



        self._init_writer()
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)

        # --- Stato training ---
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.best_model_state = None
        self.early_stop = False
        self.target_accuracy_reached = False
        self.current_epoch = 0


    # --- Setup model ---
    def _init_model(self):
        self.model = get_net(self.network_type, self.num_classes, self.temp_logger)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = self.data_manager.train_loader
        self.val_loader = self.data_manager.val_loader
        self.test_loader = self.data_manager.test_loader

    # --- Inizializzazione SummaryWriter ---
    def _init_writer(self):
        log_dir = os.path.join("runs", get_run_name())
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.debug(f"SummaryWriter inizializzato in: {log_dir}")

    # --- Setup optimizer & scheduler ---
    def _init_optim_scheduler(self):
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.hyper_parameters.learning_rate,
            momentum=self.config.hyper_parameters.momentum,
            weight_decay=self.config.hyper_parameters.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.hyper_parameters.learning_rate_scheduler_gamma,
            patience=self.config.hyper_parameters.learning_rate_scheduler_step
        )

    # --- Parametri di training & early stop ---
    def _init_params(self, config):

        # Parameters
        self.debug = config.parameters.debug

        # Hyperparameters
        self.epochs = config.hyper_parameters.epochs
        self.batch_size = config.hyper_parameters.batch_size

        # Train parameters
        self.accuracy_eval_every = config.train_parameters.accuracy_evaluation_epochs
        self.accuracy_target = config.train_parameters.accuracy_target

        # Early stop parameters
        self.start_epoch = config.early_stop_parameters.start_epoch
        self.loss_eval_every = config.early_stop_parameters.loss_evaluation_epochs
        self.patience = config.early_stop_parameters.patience
        self.improvement_rate = config.early_stop_parameters.improvement_rate

        # Output & checkpoint
        self.model_save_dir = config.output.model_save_dir
        self.load_model = config.parameters.load_model
        self.model_load_path = config.parameters.model_load_path
    
    # --- Set run name ---
    def _decide_run_name(self):
        if self.load_model and self.model_load_path:
            path_obj = Path(self.model_load_path)
            if not path_obj.is_file():
                self.temp_logger.error(f"Model file non trovato: {self.model_load_path}")
                sys.exit(1)
            checkpoint = torch.load(path_obj)
            run_name = checkpoint.get("run_name")
            if run_name is None:
                self.temp_logger.error(f"run_name mancante nel checkpoint: {self.model_load_path}")
                sys.exit(1)
            set_run_name(run_name)
        else:
            set_run_name(f"{time.strftime('%Y%m%d_%H%M%S')}_{self.network_type}")
  
    
    # --- Training epoch singolo ---
    def train_one_epoch(self, global_step):
        self.model.train()
        running_loss = 0.0
        total = 0
        all_labels = []

        loop = tqdm(self.train_loader, desc="Training epoch")
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            all_labels.append(labels)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
            global_step += 1

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        all_labels = torch.cat(all_labels)

        # --- Calcolo metriche sul train set usando self._evaluate ---
        with torch.no_grad():
            self.logger.info("Inizio valutazione sul train set...")
            train_metrics_full = self._evaluate(self.train_loader)
            train_metrics_full['loss'] = epoch_loss  # Sovrascrivi loss con media reale epoca
            train_metrics_full['all_labels'] = all_labels 
            self.logger.info("Fine valutazione sul train set")

        return train_metrics_full, global_step

    # --- Funzione centrale di training ---
    def train(self):
        self.logger.info("Inizio training...")
        self.logger.debug(f"Training su {self.device}")

        # verifico se devo caricare un modello
        self._load_and_resume_training()

        self.temp_logger.debug(f"Network selected: {self.network_type}")
        self.logger_utils = LoggerUtils(self.logger, self.writer, self.optimizer)

        global_step = 0

        for epoch in range(self.current_epoch, self.epochs):
            if self._should_stop_training():
                break

            self.current_epoch = epoch + 1
            self.logger.info(f"Inizio epoca {self.current_epoch}/{self.epochs}")

            # --- MONITOR DISTRIBUZIONE CLASSI ---
            try:
                all_labels = []
                for _, labels in self.train_loader:
                    all_labels.append(labels)
                all_labels = torch.cat(all_labels)
                self.logger.info(f"Distribuzione classi train (epoca {self.current_epoch}): {torch.bincount(all_labels)}")
            except Exception as e:
                self.logger.warning(f"Impossibile monitorare distribuzione classi: {e}")
            # --- FINE MONITOR ---

            # Training per un'epoca
            train_metrics, global_step = self.train_one_epoch(global_step)

            val_metrics = None
            if self.current_epoch >= self.start_epoch and (
                (self.current_epoch - self.start_epoch) % self.loss_eval_every == 0 or self.current_epoch == self.epochs
            ):
                val_metrics = self._evaluate(self.val_loader)
                self.scheduler.step(val_metrics['loss'])
                self._check_early_stopping(val_metrics)
                if self.best_val_loss == val_metrics['loss']:
                    self._save_model_state("model_best.pth", self.current_epoch, self.best_model_state)

            # Controllo target accuracy usando metriche già calcolate
            if self.current_epoch % self.accuracy_eval_every == 0 or self.current_epoch == self.epochs:
                self._check_accuracy_target(train_metrics, val_metrics)

            # Log centralizzato
            self.logger_utils.log_terminal(self.current_epoch, train_metrics, val_metrics)
            self.logger_utils.log_tensorboard(self.current_epoch, train_metrics, val_metrics)

        self._save_model_state("model_final.pth", self.current_epoch)
        self.logger.info("Fine training")

    # --- Early stopping ---
    def _check_early_stopping(self, val_metrics):
        val_loss = val_metrics['loss']
        if self.best_val_loss - val_loss >= self.improvement_rate:
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.no_improve_count = 0
            self.logger.info(f"Loss migliorata: {val_loss:.4f}")
        else:
            self.no_improve_count += 1
            self.logger.warning(f"Nessun miglioramento ({self.no_improve_count}/{self.patience})")
            if self.no_improve_count >= self.patience:
                self.logger.info("Early stopping attivato")
                self.early_stop = True

    # --- Controllo target accuracy ---
    def _check_accuracy_target(self, train_metrics, val_metrics):
        if train_metrics is None or val_metrics is None:
            return

        train_acc = train_metrics['accuracy'] * 100
        val_acc = val_metrics['accuracy'] * 100
        self.logger.info(f"Train Accuracy: {train_acc:.2f}% - Val Accuracy: {val_acc:.2f}%")

        if train_acc > self.accuracy_target and val_acc > self.accuracy_target:
            self.logger.info("Target accuracy raggiunta. Interrompo il training.")
            self.target_accuracy_reached = True

    # --- Valutazione completa di un loader ---
    def _evaluate(self, loader):
        loss = compute_loss(self.model, loader, self.criterion, self.device)
        acc = compute_accuracy(self.model, loader, self.device)
        per_class_acc = compute_per_class_accuracy(self.model, loader, self.device, self.num_classes)
        precision, recall, f1 = compute_precision_recall_f1(self.model, loader, self.device, self.num_classes)
        cm = compute_confusion_matrix_metric(self.model, loader, self.device, self.num_classes)

        return {
            "loss": loss,
            "accuracy": acc,
            "per_class_accuracy": per_class_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }

    # --- Test ---
    def test_model(self, use_current_model=False):
        self.logger.info("Inizio testing")

        if not use_current_model:
            if not self._load_model_state(self.model_load_path):
                self.logger.error("Impossibile caricare modello per testing. Esco.")
                return None
            
        test_metrics = self._evaluate(self.test_loader)

        # scrive su writer esistente (dal train o appena creato)
        self.writer.add_scalar("Loss/test_final", test_metrics['loss'], self.current_epoch)
        self.writer.add_scalar("Accuracy/test_final", test_metrics['accuracy'], self.current_epoch)

        self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info("Fine testing")
        return test_metrics

    # --- Salvataggio modello ---
    def _save_model_state(self, filename, epoch, model_state=None):
        os.makedirs(self.model_save_dir, exist_ok=True)
        run_name = get_run_name()
        path = os.path.join(self.model_save_dir, f"{run_name}_{filename}_epoch{epoch}.pth")
        
        # Stato del modello
        state_to_save = model_state if model_state is not None else self.model.state_dict()
        
        # Parametri principali da salvare
        meta = {
            "run_name": run_name,
            "network_type": self.network_type,
            "epoch": epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.config.hyper_parameters.learning_rate,
            "momentum": self.config.hyper_parameters.momentum,
            "weight_decay": self.config.hyper_parameters.weight_decay,
            "scheduler_gamma": self.config.hyper_parameters.learning_rate_scheduler_gamma,
            "scheduler_step": self.config.hyper_parameters.learning_rate_scheduler_step,
            "accuracy_target": self.accuracy_target,
            "loss_eval_every": self.loss_eval_every,
            "patience": self.patience,
            "improvement_rate": self.improvement_rate
        }

        torch.save({
            'model_state_dict': state_to_save,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'meta': meta
        }, path)
        
        self.logger.info(f"Modello salvato in: {path} con parametri: {meta}")

    # --- Caricamento modello ---
    def _load_model_state(self, path):
        path_obj = Path(path)
        if not path_obj.is_file():
            self.logger.error(f"Model state file not found: {path}")
            return False
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.logger.info(f"Modello caricato da: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Errore caricamento modello: {e}")
            return False

    def _load_and_resume_training(self):
        """Load model checkpoint and resume training, verifying key parameters."""
        self.current_epoch = 0
        if not (self.load_model and self.model_load_path):
            self.logger.info("Nuovo Modello Inizializzato")
            return

        path_obj = Path(self.model_load_path)
        if not path_obj.is_file():
            self.logger.error(f"Model state file not found: {self.model_load_path}")
            sys.exit(-1)

        try:
            checkpoint = torch.load(self.model_load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self._verify_checkpoint_params(checkpoint.get('meta', {}))
            self.logger.info(f"Riprendo training dal modello {self.model_load_path}")
            self.logger.debug(f"Epoch di partenza: {self.current_epoch}")
        except Exception as e:
            self.logger.error(f"Errore caricamento modello: {e}")
            sys.exit(-1)



    def _verify_checkpoint_params(self, meta):
        """Compare saved checkpoint parameters with current configuration."""
        check_params = {
            "network_type": self.network_type,
            "batch_size": self.batch_size,
            "learning_rate": self.config.hyper_parameters.learning_rate,
            "momentum": self.config.hyper_parameters.momentum,
            "weight_decay": self.config.hyper_parameters.weight_decay,
        }

        mismatches = [
            f"{key}: saved={meta.get(key)} current={value}"
            for key, value in check_params.items()
            if meta.get(key) != value
        ]
        if mismatches:
            self.logger.error(f"Discrepanze tra checkpoint e configurazione corrente: {mismatches}")
            sys.exit(1)

    # --- Controllo stop ---
    def _should_stop_training(self):
        if self.early_stop:
            self.logger.info("Training interrotto per early stopping.")
            return True
        if self.target_accuracy_reached:
            self.logger.info("Training interrotto per raggiungimento accuracy target.")
            return True
        return False
    
    def _flush_temp_logger(self):
        log_level = logging.DEBUG if self.debug else logging.INFO
        self.logger = get_logger(level=log_level, run_name=get_run_name())

        # Se esiste un MemoryHandler nel logger temporaneo, flusha tutto sul FileHandler
        if hasattr(self.temp_logger, "memory_handler"):
            mem_handler = self.temp_logger.memory_handler
            Path("logs").mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(f"logs/{get_run_name()}.log", mode="a")
            fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

            # Imposta come target e flush
            mem_handler.setTarget(fh)
            mem_handler.flush()

            # Rimuove il MemoryHandler dal logger temporaneo per evitare doppie scritture
            self.temp_logger.removeHandler(mem_handler)
            del self.temp_logger.memory_handler
        
        self.logger.info(f"Run name definitivo: {get_run_name()}")
        self.logger.info("Configurazione caricata correttamente.")
