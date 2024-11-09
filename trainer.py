"""
Author: Dimas Ahmad
Description: This file contains the functions to train and evaluate the models.
Source: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master
"""

import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
import time


# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, k, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    best_model_state = model.state_dict()
    time_start = time.time()

    for it in tqdm(range(epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            best_model_state = model.state_dict()
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    model.load_state_dict(best_model_state)
    exp_path = './Experiments/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    time_stamp = datetime.today().strftime('%Y%m%d_%H%M')
    model_save_path = os.path.join(exp_path, 'model' + '_K' + k + '_' + time_stamp + '.pth')
    torch.save(model, model_save_path)
    duration = time.time() - time_start
    print('Training completed in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    return model, train_losses, test_losses


def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
