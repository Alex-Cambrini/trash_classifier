import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from networks.resnet18 import get_resnet18
import logger

class Trainer:
    def __init__(self, config, data_manager, num_classes):
        self.config = config
        self.data_manager = data_manager
        self.num_classes = num_classes

        self.model = get_resnet18(num_classes)
        self.model.to("cpu")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.hyper_parameters.learning_rate,
            momentum=0.9,
            weight_decay=config.hyper_parameters.weight_decay
        )

        self.epochs = config.hyper_parameters.epochs
        self.train_loader = data_manager.train_loader
        self.val_loader = data_manager.val_loader
        self.test_loader = data_manager.test_loader

        self.start_epoch = config.early_stop_parameters.start_epoch
        self.loss_eval_every = config.early_stop_parameters.loss_evaluation_epochs
        self.patience = config.early_stop_parameters.patience
        self.improvement_rate = config.early_stop_parameters.improvement_rate

        self.accuracy_eval_every = config.train_parameters.accuracy_evaluation_epochs
        self.accuracy_target = config.train_parameters.accuracy_target

        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.best_model_state = None
        self.early_stop = False
        self.target_accuracy_reached = False

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(self.train_loader, desc="Training epoch")
        for inputs, labels in loop:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        return val_loss, val_acc

    def save_models(self):
        output_dir = self.config.output.model_save_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.log(f"Cartella creata: {output_dir}", level="info")

        final_path = os.path.join(output_dir, "model_final.pth")
        torch.save(self.model.state_dict(), final_path)
        logger.log(f"Modello finale salvato: {final_path}", level="info")

        if self.best_model_state:
            best_path = os.path.join(output_dir, "model_best.pth")
            torch.save(self.best_model_state, best_path)
            logger.log(f"Modello migliore salvato: {best_path}", level="info")

    def test_model(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= total
        test_acc = correct / total
        logger.log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}", level="info")
        return test_loss, test_acc

    def train(self):
        logger.log("Training su CPU", level="info")

        for epoch in range(self.epochs):
            if self.early_stop:
                logger.log("Training interrotto per early stopping.", level="info")
                break
            if self.target_accuracy_reached:
                logger.log("Training interrotto per raggiungimento accuracy target.", level="info")
                break

            epoch_loss, epoch_acc = self.train_one_epoch()
            logger.log(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}", level="info")
            logger.log(f"Batch Size {self.config.hyper_parameters.batch_size}", level="debug")

            # Valutazione su validation loss (early stopping)
            if epoch + 1 >= self.start_epoch and (epoch + 1) % self.loss_eval_every == 0:
                val_loss, val_acc = self.validate(self.val_loader)
                logger.log(f"Validation Accuracy: {val_acc:.4f}", level="info")

                if self.best_val_loss - val_loss >= self.improvement_rate:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    self.no_improve_count = 0
                    logger.log(f"Validation loss migliorata: {val_loss:.4f}", level="debug")
                else:
                    self.no_improve_count += 1
                    logger.log(f"Nessun miglioramento ({self.no_improve_count}/{self.patience})", level="debug")
                    if self.no_improve_count >= self.patience:
                        logger.log("Early stopping attivato", level="info")
                        self.early_stop = True

            # Valutazione accuracy target
            if (epoch + 1) % self.accuracy_eval_every == 0 or (epoch + 1) == self.epochs:
                train_acc = self.validate(self.train_loader)[1] * 100
                val_acc = self.validate(self.val_loader)[1] * 100
                logger.log(f"Train Accuracy: {train_acc:.2f}% - Val Accuracy: {val_acc:.2f}%", level="info")

                if train_acc > self.accuracy_target and val_acc > self.accuracy_target:
                    logger.log("Target accuracy raggiunta. Interrompo il training.", level="info")
                    self.target_accuracy_reached = True

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        self.save_models()
        test_loss, test_acc = self.test_model()
        logger.log(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}", level="info")
