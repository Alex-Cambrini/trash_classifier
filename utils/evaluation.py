from typing import Any, Dict
import torch
from torch.utils.data import DataLoader

class EvaluationUtils:
    """Utility class per la valutazione di modelli PyTorch."""
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Any,
        device: torch.device,
        metrics: Any
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.metrics = metrics

    def evaluate_light(self, loader: DataLoader) -> Dict[str, Any]:
        """Calcola solo le metriche necessarie per early stopping o per raggiungere l'accuratezza target."""
        preds_labels = self.metrics._get_all_preds_labels(loader)
        loss = self.metrics.compute_loss(loader, self.criterion)
        accuracy = self.metrics.compute_accuracy(preds_labels)
        per_class_acc = self.metrics.compute_per_class_accuracy(preds_labels)
        return {
            "loss": loss,
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc
        }

    def evaluate_full(self, loader: DataLoader) -> Dict[str, Any]:
        """Calcola tutte le metriche, comprese quelle più onerose dal punto di vista computazionale."""
        # Calcolo predizioni e label
        preds_labels = self.metrics._get_all_preds_labels(loader)
        
        # Calcolo tutte le metriche base
        loss = self.metrics.compute_loss(loader, self.criterion)
        accuracy = self.metrics.compute_accuracy(preds_labels)
        per_class_acc = self.metrics.compute_per_class_accuracy(preds_labels)
        
        # Calcolo metriche complete
        precision, recall, f1 = self.metrics.compute_precision_recall_f1(preds_labels)
        cm = self.metrics.compute_confusion_matrix_metric(preds_labels)

        all_preds, all_labels = preds_labels


        return {
            "loss": loss,
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "all_preds": all_preds,
            "val_labels": all_labels  
        }