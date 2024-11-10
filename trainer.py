"""
Author: Dimas Ahmad
Description: This file contains the functions to train and evaluate the models.
Source: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master
"""

import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
import os
import time


# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, config):
    epochs = config['epochs']
    device = config['device']
    k = config['k']
    early_stopping = EarlyStopping(config)

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    test_accuracies = np.zeros(epochs)

    best_test_loss = np.inf
    best_test_epoch = 0
    best_model_state = model.state_dict()
    time_start = time.time()

    for it in range(epochs):
        model.train()
        t0 = datetime.now()

        train_loss = []
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        train_acc = train_correct / train_total

        model.eval()
        test_loss = []
        test_correct = 0
        test_total = 0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()

        test_loss = np.mean(test_loss)
        test_acc = test_correct / test_total

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        train_accuracies[it] = train_acc
        test_accuracies[it] = test_acc

        dt = datetime.now() - t0

        if (it + 1) % config['print_freq'] == 0 or it == 0:
            print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
                      Validation Loss: {test_loss:.4f}, \
                      Train Accuracy: {train_acc:.4f}, \
                      Validation Accuracy: {test_acc:.4f}, \
                      Duration: {dt}, Best Val Epoch: {best_test_epoch}')

        if test_loss < best_test_loss:
            best_model_state = model.state_dict()
            best_test_loss = test_loss
            best_test_epoch = it
            print('best model updated')

        early_stopping(best_test_loss, test_loss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", it + 1)
            print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
                      Validation Loss: {test_loss:.4f}, \
                      Train Accuracy: {train_acc:.4f}, \
                      Validation Accuracy: {test_acc:.4f}, \
                      Duration: {dt}, Best Val Epoch: {best_test_epoch}')
            break

    model.load_state_dict(best_model_state)
    exp_path = './Experiments/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    time_stamp = datetime.today().strftime('%Y%m%d_%H%M')
    model_save_path = os.path.join(exp_path, 'model' + '_K' + str(k) + '_' + time_stamp + '.pth')
    torch.save(model, model_save_path)
    duration = time.time() - time_start
    print('Training completed in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    return model, train_losses, test_losses, train_accuracies, test_accuracies


def evaluate(model, test_loader, config):
    device = config['device']
    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    report = classification_report(all_targets, all_predictions, digits=4)
    print(report)
    return report


class EarlyStopping:
    def __init__(self, config):
        self.tolerance = config['patience']
        self.min_delta = config['min_delta']
        self.counter = 0
        self.early_stop = False

    def __call__(self, best_loss, epoch_val_loss):
        if epoch_val_loss <= (best_loss + self.min_delta):
            self.counter = 0
        elif epoch_val_loss > (best_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
