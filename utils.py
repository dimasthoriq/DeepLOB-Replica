"""
Author: Dimas Ahmad
Description: This file contains utility functions for data preprocessing functions.
Source: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master
"""

import numpy as np
import torch


def prepare_x(data):
    # Only use the first 40 features (10 levels for each price and volume of ask and bid)
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    # Only use the last 5 features which are the labels with different prediction horizons
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dy = np.array(Y)

    # Only use the data starting from the time window length T
    data_y = dy[T - 1:N]
    data_x = np.zeros((N - T + 1, T, D))

    for i in range(T, N + 1):
        data_x[i - T] = df[i - T:i, :]

    return data_x, data_y


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = torch.nn.functional.one_hot(y, num_classes=3)
    return x, y


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, k, num_classes, T):
        """
        Initialization:
        :param data: data
        :param k: prediction horizon
        :param num_classes: number of classes for classification (3 in this case: increasing, decreasing, constant)
        :param T: time window length for a single input
        """
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


def load_data(data_path):
    trainval_data = np.loadtxt(data_path + 'Train_Dst_NoAuction_DecPre_CF_7.txt')
    train_data = trainval_data[:, :int(np.floor(trainval_data.shape[1] * 0.8))]
    val_data = trainval_data[:, int(np.floor(trainval_data.shape[1] * 0.8)):]

    test_1 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_7.txt')
    test_2 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_8.txt')
    test_3 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_9.txt')
    test_data = np.hstack((test_1, test_2, test_3))

    print(train_data.shape, val_data.shape, test_data.shape)
    return train_data, val_data, test_data
