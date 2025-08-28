class EvaluationUtils:
    def __init__(self, model, criterion, device, metrics):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.metrics = metrics

    # --- Valutazione “leggera” --- 
    def _evaluate_light(self, loader):
        """Compute only metrics needed for early stopping / target accuracy."""

        preds_labels = self.metrics._get_all_preds_labels(loader)
        loss = self.metrics.compute_loss(loader, self.criterion)
        accuracy = self.metrics.compute_accuracy(preds_labels)
        per_class_acc = self.metrics.compute_per_class_accuracy(preds_labels)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc
        }

    # --- Valutazione completa ---
    def _evaluate_full(self, loader):
        """Compute all metrics, heavy calculations included."""
        # Calcolo predizioni e label
        preds_labels = self.metrics._get_all_preds_labels(loader)
        
        # Calcolo tutte le metriche base
        loss = self.metrics.compute_loss(loader, self.criterion)
        accuracy = self.metrics.compute_accuracy(preds_labels)
        per_class_acc = self.metrics.compute_per_class_accuracy(preds_labels)
        
        # Calcolo metriche complete
        precision, recall, f1 = self.metrics.compute_precision_recall_f1(preds_labels)
        cm = self.metrics.compute_confusion_matrix_metric(preds_labels)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }