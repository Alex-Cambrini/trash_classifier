import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

def compute_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

def compute_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def compute_per_class_accuracy(model, loader, device, num_classes):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                mask = labels == i
                correct[i] += (predicted[mask] == i).sum().item()
                total[i] += mask.sum().item()
    per_class_acc = (correct / total).cpu().numpy()
    return per_class_acc

def compute_confusion_matrix_metric(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm

def compute_precision_recall_f1(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(num_classes)), average=None)
    return precision, recall, f1
