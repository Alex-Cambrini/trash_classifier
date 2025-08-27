# evaluation_utils.py
from metrics import (
    compute_loss,
    compute_accuracy,
    compute_per_class_accuracy,
    compute_precision_recall_f1,
    compute_confusion_matrix_metric,
)

class EvaluationUtils:
    # --- Valutazione “leggera” --- 
    def _evaluate_light(self, loader):
        """Compute only metrics needed for early stopping / target accuracy."""
        loss = compute_loss(self.model, loader, self.criterion, self.device)
        acc = compute_accuracy(self.model, loader, self.device)
        per_class_acc = compute_per_class_accuracy(
            self.model, loader, self.device, self.num_classes
        )

        return {
            "loss": loss,
            "accuracy": acc,
            "per_class_accuracy": per_class_acc
        }

    # --- Valutazione completa ---
    def _evaluate_full(self, loader):
        """Compute all metrics, heavy calculations included."""
        metrics = self._evaluate_light(loader)
        precision, recall, f1 = compute_precision_recall_f1(
            self.model, loader, self.device, self.num_classes
        )
        cm = compute_confusion_matrix_metric(
            self.model, loader, self.device, self.num_classes
        )

        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        })
        return metrics
