"""
Author: Dimas Ahmad
Description: This file contains the model class for the CNN-LSTM model with Inception modules.
Source: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master
"""

import torch


class DeepLOB(torch.nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

        # convolution blocks
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            #             torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
        )

        # inception modules
        self.inp1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )
        self.inp2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )
        self.inp3 = torch.nn.Sequential(
            torch.nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = torch.nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = torch.nn.Linear(64, self.y_len)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y
