import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter


class LoggerUtils:
    def __init__(self, logger, writer: SummaryWriter, optimizer=None):
        self.logger = logger
        self.writer = writer
        self.optimizer = optimizer

    def log_test_final(self, epoch, metrics, config_params):
        """
        Log finale del test:
        - Terminale
        - HParams (config + metriche principali)
        """

        self.logger.info(f"Contenuto di metrics: {metrics}")
        # --- Terminale ---
        self.logger.info(
            f"Test finale | Epoch {epoch} | "
            f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
        )

        # --- HParams: salva configurazione + metriche principali ---
        hparam_metrics = {
            "loss": float(metrics['loss']),
            "accuracy": float(metrics['accuracy'])
        }

        self.writer.add_hparams(config_params, hparam_metrics)


    # --- Log principale per terminale ---
    def log_terminal(self, epoch, metrics_dict):
        self.logger.info(f"Epoch {epoch} Results:")
        for split, metrics in (metrics_dict or {}).items():
            if metrics is None:
                continue

            # Log principale compatto
            loss = metrics.get('loss')
            acc = metrics.get('accuracy')
            loss_str = f"Loss: {loss:.4f}" if loss is not None else "Loss: N/D"
            acc_str = f"Acc: {acc:.4f}" if acc is not None else "Acc: N/D"
            self.logger.info(f"{split.capitalize()} | {loss_str}, {acc_str}")

            # Log dettagli per classe solo in DEBUG
            for metric_name in ['per_class_accuracy', 'precision', 'recall', 'f1']:
                values = metrics.get(metric_name)
                if values is not None:
                    mean_val = np.mean(values)
                    self.logger.debug(f"{split.capitalize()} {metric_name} (mean): {mean_val:.4f}")
                    self.logger.debug(f"{split.capitalize()} {metric_name} (per class): {values}")

    # --- Log per TensorBoard ---
    def log_tensorboard(self, epoch, metrics_dict, embeddings=None, labels=None):
        for split, metrics in metrics_dict.items():
                self._log_metrics(epoch, metrics, split)
                self._log_confusion_matrix(epoch, metrics, split)
                if split == "train":  # log learning rate solo per train
                    self._log_learning_rate(epoch)
                self._log_class_distribution(epoch, metrics.get('all_labels'), split=split)

        if embeddings is not None and labels is not None:
            self.writer.add_embedding(embeddings, metadata=labels, global_step=epoch)

    # --- Helper: metriche generali + per classe ---
    def _log_metrics(self, epoch, metrics, split):
        """
        Log su TensorBoard per un set di metriche.
        Lo split può essere 'train', 'val' o 'test_final'.
        """
        for metric_name in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'per_class_accuracy']:
            if metric_name not in metrics:
                continue
            val = metrics[metric_name]
            mean_val = np.mean(val) if isinstance(val, (list, np.ndarray)) else val
            self.writer.add_scalar(f"{metric_name}/{split}", mean_val, epoch)
            if isinstance(val, (list, np.ndarray)):
                for i, v in enumerate(val):
                    self.writer.add_scalar(f"{metric_name}/{split}/class_{i}", v, epoch)


    # --- Helper: confusion matrix ---
    def _log_confusion_matrix(self, epoch, metrics, split):
        if metrics is None or 'confusion_matrix' not in metrics:
            return
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


    # --- Helper: embeddings ---
    def _log_embeddings(self, epoch, features, labels=None, images=None, tag="embeddings"):
        """
        Log embedding su TensorBoard.
        features: torch.Tensor [N, D]
        labels: lista o array [N] (opzionale, usato come metadata)
        images: torch.Tensor [N, C, H, W] (opzionale, mostrato come thumbnail)
        """
        if features is None:
            return
        features = features.detach().cpu()
        metadata = labels if labels is not None else None
        label_img = images.detach().cpu() if images is not None else None

        self.writer.add_embedding(features, metadata=metadata, label_img=label_img, global_step=epoch, tag=tag)
