import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Tuple, Any

class Metrics:
    """Calcola metriche di valutazione per modelli di classificazione."""
    def __init__(self, model: torch.nn.Module, device: torch.device, num_classes: int) -> None:
        self.model = model
        self.device = device
        self.num_classes = num_classes

    def compute_loss(self, loader: DataLoader, criterion: Any) -> float:
        """Calcola la loss media su un DataLoader."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
        return total_loss / len(loader.dataset)

    def compute_accuracy(self, preds_labels: Tuple[Tensor, Tensor]) -> float:
        """Calcola l'accuracy globale."""
        preds, labels = preds_labels
        return (preds == labels).float().mean().item()

    def compute_per_class_accuracy(self, preds_labels: Tuple[Tensor, Tensor]) -> Tensor:
        """Calcola l'accuracy per ogni classe."""
        preds, labels = preds_labels
        cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(self.num_classes)))
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        return per_class_acc

    def compute_confusion_matrix_metric(self, preds_labels: Tuple[Tensor, Tensor]) -> Tensor:
        """Restituisce la confusion matrix."""
        preds, labels = preds_labels
        cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(self.num_classes)))
        return cm

    def compute_precision_recall_f1(self, preds_labels: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Calcola precision, recall e f1 per ciascuna classe."""
        preds, labels = preds_labels
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.numpy(), preds.numpy(), labels=list(range(self.num_classes)), average=None
        )
        return precision, recall, f1
    
    def _get_all_preds_labels(self, loader: DataLoader) -> Tuple[Tensor, Tensor]:
        """Recupera predizioni e labels per l'intero DataLoader."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_preds), torch.cat(all_labels)
