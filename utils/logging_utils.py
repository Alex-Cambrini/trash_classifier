import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from typing import Any, Dict, Optional


class LoggerUtils:
    """Utility per logging su terminale e TensorBoard."""
    def __init__(
        self,
        logger: logging.Logger,
        writer: SummaryWriter,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        self.logger = logger
        self.writer = writer
        self.optimizer = optimizer
        self.class_names = None

    def _to_numpy(self, val) -> Optional[np.ndarray]:
        """Converte, se presente, tensore, lista o singolo valore in np.ndarray."""
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        if isinstance(val, list):
            return np.array(val)
        if isinstance(val, np.ndarray):
            return val  # Se è già un array NumPy, non fare nulla
        return np.array([val])

    def log_test_final(
        self, epoch: int, metrics: Dict[str, Any], config_params: Dict[str, Any]
    ) -> None:
        """Log finale del test su terminale e TensorBoard HParams."""
        # --- Terminale ---
        self.logger.info(
            f"Test finale | Epoch {epoch} | "
            f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
        )

        # --- HParams ---
        hparam_metrics: Dict[str, float] = {
            "hparam/loss": metrics["loss"],
            "hparam/accuracy": metrics["accuracy"],
            "hparam/precision": (
                float(metrics["precision"].mean().item())
                if hasattr(metrics["precision"], "mean")
                else metrics.get("precision")
            ),
            "hparam/recall": (
                float(metrics["recall"].mean().item())
                if hasattr(metrics["recall"], "mean")
                else metrics.get("recall")
            ),
            "hparam/f1": (
                float(metrics["f1"].mean().item())
                if hasattr(metrics["f1"], "mean")
                else metrics.get("f1")
            ),
        }

        self.writer.add_hparams(config_params, hparam_metrics)

    def log_terminal(
        self, epoch: int, metrics_dict: Dict[str, Optional[Dict[str, Any]]]
    ) -> None:
        """Log principale per terminale."""
        self.logger.info(f"Epoch {epoch} Results:")
        for split, metrics in (metrics_dict or {}).items():
            if metrics is None:
                continue

            # Log principale compatto
            loss = metrics.get("loss")
            acc = metrics.get("accuracy")
            loss_str = f"Loss: {loss:.4f}" if loss is not None else "Loss: N/D"
            acc_str = f"Acc: {acc:.4f}" if acc is not None else "Acc: N/D"
            self.logger.info(f"{split.capitalize()} | {loss_str}, {acc_str}")

            # Log dettagli per classe solo in DEBUG
            for metric_name in ["per_class_accuracy", "precision", "recall", "f1"]:
                val = self._to_numpy(metrics.get(metric_name))
                if val is not None and val.size > 0:
                    val = val.flatten()
                    mean_val = np.mean(val)
                    self.logger.debug(
                        f"{split.capitalize()} {metric_name} (mean): {mean_val:.4f}"
                    )
                    self.logger.debug(
                        f"{split.capitalize()} {metric_name} (per class): {val}"
                    )

    def log_tensorboard(
        self, epoch: int, metrics_dict: Dict[str, Dict[str, Any]]
    ) -> None:
        """Log su TensorBoard per tutte le metriche e distribuzioni."""
        self.logger.debug(f"Logging metrics for epoch {epoch}: {metrics_dict.keys()}")

        for split, metrics in metrics_dict.items():
            self._log_metrics(epoch, metrics, split)
            self._log_confusion_matrix(epoch, metrics, split)
            if split == "train":  # log learning rate solo per train
                self._log_learning_rate(epoch)
            self._log_pred_distribution(epoch, metrics.get("all_preds"), split=split)

        # Log della distribuzione delle classi una sola volta, con tutti i dati disponibili
        train_labels = metrics_dict.get("train", {}).get("train_labels")
        val_labels = metrics_dict.get("val", {}).get("val_labels")

        if train_labels is not None or val_labels is not None:
            self._log_class_distribution(
                epoch, train_labels=train_labels, val_labels=val_labels
            )

    def _log_metrics(self, epoch: int, metrics: Dict[str, Any], split: str) -> None:
        """Log scalari e metriche per classe su TensorBoard."""
        # Logga le metriche scalari singolarmente (loss, accuracy)
        for metric_name in ["loss", "accuracy"]:
            val = metrics.get(metric_name)
            if val is not None:
                self.writer.add_scalar(f"{metric_name}/{split}", float(val), epoch)

        # Logga le metriche per classe su un unico grafico
        for metric_name in ["per_class_accuracy", "precision", "recall", "f1"]:
            val = metrics.get(metric_name)
            if val is not None:
                val = self._to_numpy(val)
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    class_metrics = {}
                    for i, v in enumerate(val):
                        class_metrics[self._class_name(i)] = float(v)
                    self.writer.add_scalars(
                        f"{metric_name}/{split}", class_metrics, epoch
                    )

    def _log_confusion_matrix(
        self, epoch: int, metrics: Dict[str, Any], split: str
    ) -> None:
        """Logga la confusion matrix su TensorBoard per lo split specificato."""
        if metrics is None or "confusion_matrix" not in metrics:
            return
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 6))
        class_names = [self._class_name(i) for i in range(cm.shape[0])]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{split.capitalize()} Confusion Matrix")
        self.writer.add_figure(f"ConfusionMatrix_Epoch_{epoch}/{split}", fig, epoch)
        plt.close(fig)

    def _log_learning_rate(self, epoch: int) -> None:
        """Logga il learning rate corrente su TensorBoard."""
        if self.optimizer is None:
            return
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Learning Rate", current_lr, epoch)

    def _log_class_distribution(
        self,
        epoch: int,
        train_labels: Optional[Tensor],
        val_labels: Optional[Tensor] = None,
        num_classes: Optional[int] = None,
    ):
        """Logga la distribuzione delle classi su TensorBoard per train e val."""
        if train_labels is None and val_labels is None:
            return

        # Determina numero classi
        n_classes = num_classes
        if n_classes is None:
            candidates = []
            if train_labels is not None:
                candidates.append(int(train_labels.max().item()) + 1)
            if val_labels is not None:
                candidates.append(int(val_labels.max().item()) + 1)
            n_classes = max(candidates)

        train_counts = (
            torch.bincount(train_labels.to(torch.long), minlength=n_classes)
            if train_labels is not None
            else torch.zeros(n_classes)
        )
        val_counts = (
            torch.bincount(val_labels.to(torch.long), minlength=n_classes)
            if val_labels is not None
            else torch.zeros(n_classes)
        )

        # Crea il grafico
        fig, ax = plt.subplots(figsize=(8, 5))
        indices = np.arange(n_classes)
        width = 0.35

        ax.bar(indices - width / 2, train_counts.numpy(), width, label="Train")
        ax.bar(indices + width / 2, val_counts.numpy(), width, label="Val")

        ax.set_xlabel("Classi")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribuzione classi - Epoch {epoch}")
        ax.set_xticks(indices)
        ax.set_xticklabels([self._class_name(i) for i in range(n_classes)])
        ax.legend()

        self.writer.add_figure(f"class_distribution/epoch_{epoch}", fig, epoch)
        plt.close(fig)

    def _log_pred_distribution(
        self,
        epoch: int,
        preds: Optional[Tensor],
        split: str,
        num_classes: Optional[int] = None,
    ) -> None:
        """Logga la distribuzione delle predizioni per split su TensorBoard."""
        if preds is None:
            return

        preds = preds.to(torch.long).cpu()
        n_classes = num_classes or int(preds.max().item()) + 1
        bincount = torch.bincount(preds, minlength=n_classes)
        total = bincount.sum().item()

        # Raggruppa tutte le classi in un unico grafico per Count
        count_metrics = {self._class_name(i): int(v) for i, v in enumerate(bincount)}
        self.writer.add_scalars(
            f"pred_distribution/{split}/count", count_metrics, epoch
        )

        # Raggruppa tutte le classi in un unico grafico per Freq
        freq_metrics = {
            self._class_name(i): float(v) / total for i, v in enumerate(bincount)
        }
        self.writer.add_scalars(f"pred_distribution/{split}/freq", freq_metrics, epoch)


    def _class_name(self, i: int) -> str:
        """Restituisce il nome della classe se disponibile, altrimenti 'class_i'."""
        if self.class_names is not None and i < len(self.class_names):
            return self.class_names[i]
        return f"class_{i}"
