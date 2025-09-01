import sys
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import os
import copy
import logging
import torch.optim as optim
from tqdm import tqdm
from utils.logging_utils import LoggerUtils
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint import read_checkpoint
from utils.evaluation import EvaluationUtils
from utils.metrics import Metrics
from utils.model_utils import get_config_params, verify_checkpoint_params


class Trainer(EvaluationUtils):
    def __init__(
        self,
        config: Any,
        data_manager: Any,
        run_name: str,
        logger: logging.Logger,
        writer: SummaryWriter,
        logger_utils: LoggerUtils,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.config = config
        self.data_manager = data_manager
        self.run_name = run_name
        self.logger = logger
        self.writer = writer
        self.logger_utils = logger_utils
        self.model = model
        self.criterion = criterion
        self.device = device

        self.num_classes = len(data_manager.classes)

        # Inizializzazione della classi Metrics e EvaluationUtils
        self.metrics = Metrics(
            model=self.model, device=self.device, num_classes=self.num_classes
        )
        self.evaluation_utils = EvaluationUtils(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            metrics=self.metrics,
        )

        self.network_type = self.config.train_parameters.network_type

        # --- Configurazioni principali ---
        self.train_loader = data_manager.train_loader
        self.val_loader = data_manager.val_loader
        self._init_params(config)
        self._init_optim_scheduler()

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)

        # --- Stato training ---
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.best_model_state = None
        self.early_stop = False
        self.target_accuracy_reached = False
        self.current_epoch = 0
        self.global_step = None

    def train_one_epoch(self) -> Dict[str, Any]:
        """Esegue il training di un'epoca e ritorna metriche"""
        self.model.train()
        running_loss = 0.0
        total = 0
        class_counts = torch.zeros(self.num_classes, device="cpu")

        loop = tqdm(self.train_loader, desc="Training epoch")
        train_labels_list = []

        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            train_labels_list.append(labels.cpu())

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("Loss/train_step", loss.item(), self.global_step)
            self.global_step += 1

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

            # Aggiornamento distribuzione classi incrementale
            class_counts += torch.bincount(labels.cpu(), minlength=self.num_classes)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        train_labels = torch.cat(train_labels_list)

        self.logger.info(
            f"Distribuzione classi train (epoca {self.current_epoch}): {class_counts.int()}"
        )

        # --- Calcolo metriche solo se necessario ---
        train_metrics_full = {"loss": epoch_loss, "train_labels": train_labels}

        if (self.current_epoch % self.accuracy_eval_every == 0) or (
            self.current_epoch == self.epochs
        ):
            with torch.no_grad():
                self.logger.info("Inizio Valutazione Rapida (Train Set)...")
                metrics = self.evaluation_utils.evaluate_light(self.train_loader)
                self.logger.info("Fine Valutazione Rapida (Train Set)")
                self.logger.debug("Aggiornamento Delle Metriche Di Train")
                train_metrics_full.update(metrics)
                self.logger.info("Metriche Aggiornate")
        return train_metrics_full

    def train(self):
        """Esegue il training del modello per tutte le epoche configurate."""
        self.logger.info("Inizio training...")
        self.logger.debug(f"Training su {self.device}")
        self.global_step = 0
        self._load_and_resume_training()
        self.logger.debug(f"Network selected: {self.network_type}")

        for epoch in range(self.current_epoch, self.epochs):
            if self._should_stop_training():
                break

            self.current_epoch = epoch + 1
            self.logger.info(f"Inizio epoca {self.current_epoch}/{self.epochs}")

            # Training per un'epoca
            train_metrics = self.train_one_epoch()

            # Converti tensori in liste per train_metrics
            train_metrics = self._convert_metrics_to_list(train_metrics)

            val_metrics = None

            # Se siamo oltre l'epoca di start e è il momento di valutare secondo loss_eval_every, facciamo la validazione
            if self.current_epoch >= self.start_epoch and (
                (self.current_epoch - self.start_epoch) % self.loss_eval_every == 0
                or self.current_epoch == self.epochs
            ):
                self.logger.info("Inizio Valutazione Completa (Val Set)...")
                val_metrics = self.evaluation_utils.evaluate_full(self.val_loader)
                val_metrics = self._convert_metrics_to_list(val_metrics)

                self.logger.info(
                    f"Fine Valutazione Completa (Val Set) | Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
                )
                self.scheduler.step(val_metrics["loss"])
                self._check_early_stopping(val_metrics)
                self.logger.debug(
                    f"Scheduler step called | current val_loss={val_metrics['loss']:.4f} | "
                    f"scheduler patience={self.scheduler.patience}"
                )
                # Salvo modello se è il migliore
                if self.best_val_loss == val_metrics["loss"]:
                    self._save_model_state(
                        filename="model_best.pth",
                        epoch=self.current_epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )

                # Controllo target accuracy solo quando è il momento
                if (self.current_epoch % self.accuracy_eval_every == 0) or (
                    self.current_epoch == self.epochs
                ):
                    self._check_accuracy_target(train_metrics, val_metrics)

                metrics_dict = {"train": train_metrics, "val": val_metrics}
            else:
                val_metrics = None
                self.logger.debug(
                    f"Epoca {self.current_epoch}: skip validazione (start_epoch={self.start_epoch}, loss_eval_every={self.loss_eval_every})"
                )
                metrics_dict = {"train": train_metrics}

            self.logger_utils.log_terminal(self.current_epoch, metrics_dict)
            self.logger_utils.log_tensorboard(self.current_epoch, metrics_dict)

        self._save_model_state(
            "model_final.pth",
            self.current_epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )
        self.logger.info("Fine training")

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> None:
        """Aggiorna lo stato di early stopping basato sulla loss di validazione."""
        val_loss = val_metrics["loss"]
        self.logger.debug(
            f"Controllo early stopping epoca {self.current_epoch} | "
            f"best_val_loss={self.best_val_loss:.4f}, current_val_loss={val_loss:.4f}, "
            f"no_improve_count={self.no_improve_count}"
        )

        # Aggiorna best model solo se miglioramento significativo
        if self.best_val_loss - val_loss >= self.improvement_rate:
            # Aggiorna best_model_state prima di cambiare best_val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.best_val_loss = val_loss
            self.no_improve_count = 0
            self.logger.info(f"Loss migliorata: {val_loss:.4f}")
        else:
            self.no_improve_count += 1
            self.logger.warning(
                f"Nessun miglioramento ({self.no_improve_count}/{self.patience})"
            )
            if self.no_improve_count >= self.patience:
                self.logger.info("Early stopping attivato")
                self.early_stop = True

    def _check_accuracy_target(
        self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]
    ) -> None:
        """Controlla se il target di accuracy è stato raggiunto e aggiorna lo stato."""
        if train_metrics is None or val_metrics is None:
            self.logger.warning(
                "Metriche di train o validazione mancanti, non è possibile controllare target accuracy."
            )
            return

        train_acc = train_metrics["accuracy"] * 100
        val_acc = val_metrics["accuracy"] * 100

        if train_acc > self.accuracy_target and val_acc > self.accuracy_target:
            self.logger.info("Target accuracy raggiunta. Interrompo il training.")
            self.target_accuracy_reached = True

    def _save_model_state(
        self,
        filename: str,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        model_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Salva lo stato del modello e dell'optimizer con i metadati principali."""
        os.makedirs(self.model_save_dir, exist_ok=True)
        path = os.path.join(
            self.model_save_dir, f"{self.run_name}_{filename}_epoch{epoch}.pth"
        )

        # Parametri principali da salvare
        meta = {
            "run_name": self.run_name,
            "network_type": self.network_type,
            "epoch": epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.config.hyper_parameters.learning_rate,
            "momentum": self.config.hyper_parameters.momentum,
            "weight_decay": self.config.hyper_parameters.weight_decay,
            "scheduler_gamma": self.config.hyper_parameters.learning_rate_scheduler_gamma,
            "scheduler_step": self.config.hyper_parameters.scheduler_patience_in_val_steps,
            "accuracy_target": self.accuracy_target,
            "loss_eval_every": self.loss_eval_every,
            "patience": self.patience,
            "improvement_rate": self.improvement_rate,
            "global_step": self.global_step
        }

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": {"train": train_metrics, "val": val_metrics},
                "meta": meta,
            },
            path,
        )
        self.logger.info(f"Modello salvato in: {path}")
        self.logger.debug(f"Parametri modello: {meta}")

    def _load_and_resume_training(self) -> None:
        """Carica il checkpoint e riprende il training, controllando la coerenza dei parametri."""
        if not (self.load_model and self.model_load_path):
            self.logger.info("Nuovo modello inizializzato")
            return

        ckpt = read_checkpoint(self.model_load_path, self.logger)
        if ckpt is None:
            sys.exit(-1)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.meta = ckpt.get("meta")
        self.current_epoch = self.meta["epoch"]
        self.global_step = self.meta.get("global_step", 0)
        self.logger.debug(f"global step = {self.global_step}")
        config_params = get_config_params(self.config)
        verify_checkpoint_params(self.meta, config_params, self.logger)
        self.logger.info(f"Riprendo training dal checkpoint {self.model_load_path}")

    def _should_stop_training(self) -> bool:
        """Verifica se il training deve fermarsi per early stopping o target accuracy."""
        if self.early_stop:
            self.logger.info("Training interrotto per early stopping.")
            return True
        if self.target_accuracy_reached:
            self.logger.info("Training interrotto per raggiungimento accuracy target.")
            return True
        return False

    def _convert_metrics_to_list(
        self, metrics: Dict[str, Any], keys=None
    ) -> Dict[str, Any]:
        """
        Converte i tensori PyTorch di alcune metriche in liste Python.
        """
        keys = keys or ["per_class_accuracy", "precision", "recall", "f1"]
        for key in keys:
            if metrics.get(key) is not None and hasattr(metrics[key], "cpu"):
                metrics[key] = metrics[key].cpu().tolist()
        return metrics
    
    def _init_params(self, config: Any) -> None:
        """Inizializza tutti i parametri principali di training, early stopping e checkpoint."""
        # Parameters
        self.debug = config.parameters.debug

        # Hyperparameters
        self.epochs = config.hyper_parameters.epochs
        self.batch_size = config.hyper_parameters.batch_size
        self.learning_rate_scheduler_gamma = (
            self.config.hyper_parameters.learning_rate_scheduler_gamma
        )
        self.scheduler_patience_in_val_steps = (
            self.config.hyper_parameters.scheduler_patience_in_val_steps
        )
        self.learning_rate = self.config.hyper_parameters.learning_rate
        self.momentum = self.config.hyper_parameters.momentum
        self.weight_decay = self.config.hyper_parameters.weight_decay

        # Train parameters
        self.accuracy_eval_every = config.train_parameters.accuracy_evaluation_epochs
        self.accuracy_target = config.train_parameters.accuracy_target

        # Early stop parameters
        self.start_epoch = config.early_stop_parameters.start_epoch
        self.loss_eval_every = config.early_stop_parameters.val_loss_every_n_epochs
        self.patience = config.early_stop_parameters.patience
        self.improvement_rate = config.early_stop_parameters.improvement_rate

        # Output & checkpoint
        self.model_save_dir = config.output.model_save_dir
        self.load_model = config.parameters.load_model
        self.model_load_path = config.parameters.model_load_path

    def _init_optim_scheduler(self) -> None:
        """Inizializza l'optimizer SGD e lo scheduler ReduceLROnPlateau."""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # La patience di ReduceLROnPlateau conta i controlli di validazione consecutivi senza miglioramento,
        # non le epoche reali. L'early stopping invece conta le epoche reali senza miglioramento.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.learning_rate_scheduler_gamma,
            patience=self.scheduler_patience_in_val_steps,
        )