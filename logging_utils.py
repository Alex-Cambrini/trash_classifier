import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter
from logger import get_logger


class LoggerUtils:
    def __init__(self, logger, writer: SummaryWriter, optimizer=None):
        self.logger = logger
        self.writer = writer
        self.optimizer = optimizer


    # --- Log principale per terminale ---
    def log_terminal(self, epoch, train_metrics, val_metrics=None):
        # INFO: metriche aggregate
        train_acc = train_metrics.get('accuracy', None)
        train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "N/D"
        msg = f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.4f}, Acc: {train_acc_str}"
        if val_metrics is not None:
            msg += f" | Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics.get('accuracy', 0):.4f}"
        self.logger.info(msg)

        # INFO: medie delle metriche per classe
        for metric_name in ['per_class_accuracy', 'precision', 'recall', 'f1']:
            if metric_name in train_metrics:
                mean_val = np.mean(train_metrics[metric_name])
                self.logger.info(f"Train {metric_name} (mean): {mean_val:.4f}")
                # DEBUG: valori per classe
                self.logger.debug(f"Train {metric_name} (per class): {train_metrics[metric_name]}")
            if val_metrics is not None and metric_name in val_metrics:
                mean_val = np.mean(val_metrics[metric_name])
                self.logger.info(f"Val {metric_name} (mean): {mean_val:.4f}")
                # DEBUG: valori per classe
                self.logger.debug(f"Val {metric_name} (per class): {val_metrics[metric_name]}")

    # --- Log per TensorBoard ---
    def log_tensorboard(self, epoch, train_metrics, val_metrics=None):
        self._log_metrics(epoch, train_metrics, val_metrics)
        self._log_confusion_matrix(epoch, train_metrics, val_metrics)
        self._log_learning_rate(epoch)

        # log distribuzione classi
        self._log_class_distribution(epoch, train_metrics.get('all_labels'), split="train")
        if val_metrics and 'all_labels' in val_metrics:
            self._log_class_distribution(epoch, val_metrics['all_labels'], split="val")

    # --- Helper: metriche generali + per classe ---
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        for metric_name in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'per_class_accuracy']:
            for split, metrics in zip(['train', 'val'], [train_metrics, val_metrics]):
                if metrics is None or metric_name not in metrics:
                    continue
                val = metrics[metric_name]
                self.writer.add_scalar(f"{metric_name}/{split}", np.mean(val) if isinstance(val, (list, np.ndarray)) else val, epoch)
                # Log per classe
                if isinstance(val, (list, np.ndarray)):
                    for i, v in enumerate(val):
                        self.writer.add_scalar(f"{metric_name}/{split}/class_{i}", v, epoch)

    # --- Helper: confusion matrix ---
    def _log_confusion_matrix(self, epoch, train_metrics, val_metrics):
        for split, metrics in zip(['train', 'val'], [train_metrics, val_metrics]):
            if metrics is None or 'confusion_matrix' not in metrics:
                continue
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{split.capitalize()} Confusion Matrix')
            self.writer.add_figure(f"ConfusionMatrix_Epoch_{epoch}/{split}", fig, epoch)
            plt.close(fig)

    # --- Helper: learning rate ---
    def _log_learning_rate(self, epoch):
        if self.optimizer is None:
            return
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("Learning Rate", current_lr, epoch)
        
    def _log_class_distribution(self, epoch, labels, split="train"):
        if labels is None:
            return
        bincount = torch.bincount(labels)
        for i, count in enumerate(bincount):
            self.writer.add_scalar(f"{split}/class_distribution/class_{i}", count.item(), epoch)
