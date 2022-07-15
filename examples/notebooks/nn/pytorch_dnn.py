import os.path

import pandas as pd
from json import load
import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils import scale_up, scale_down

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def chebyshev(x_grid, alpha_c, alpha_n):
    T = np.zeros((len(x_grid), n + 1))
    T[:, 0] = np.ones((len(x_grid), 1)).T
    z_grid = scale_down(x_grid, x_min, x_max)
    T[:, 1] = z_grid.T
    for i in range(1, n):
        T[:, i + 1] = 2 * z_grid * T[:, i] - T[:, i - 1]
    T_chebyshev = T @ alpha_c
    T_nn = T @ alpha_n
    return T_chebyshev, T_nn


class AndersonChebyshevDataset(Dataset):
    """Anderson chebyshev dataset."""

    def __init__(self, csv_file, order, transform=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.order = order
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anderson_config = self.df.iloc[idx, :input_size]
        anderson_config = np.array([anderson_config]).astype('float32')

        # get i-th order chebyshev polynomial
        poly_i = np.array(self.df.iloc[idx, self.order + input_size]).astype('float32')
        sample = {"anderson": anderson_config, "poly": poly_i}
        if self.transform:
            sample["anderson"] = self.transform(sample["anderson"])

        return sample


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, no_hidden_units, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, no_hidden_units[0])
        self.fc2 = nn.Linear(no_hidden_units[0], no_hidden_units[1])
        self.fc3 = nn.Linear(no_hidden_units[1], no_hidden_units[2])
        self.fc4 = nn.Linear(no_hidden_units[2], no_hidden_units[3])
        self.fc5 = nn.Linear(no_hidden_units[3], no_hidden_units[4])
        self.fc6 = nn.Linear(no_hidden_units[4], no_hidden_units[5])
        self.fc7 = nn.Linear(no_hidden_units[5], no_hidden_units[6])
        self.fc8 = nn.Linear(no_hidden_units[6], no_hidden_units[7])
        self.fc9 = nn.Linear(no_hidden_units[7], no_hidden_units[8])
        self.fc10 = nn.Linear(no_hidden_units[8], no_hidden_units[9])
        self.fc11 = nn.Linear(no_hidden_units[9], no_hidden_units[10])
        self.fc12 = nn.Linear(no_hidden_units[10], no_hidden_units[11])
        self.output = nn.Linear(no_hidden_units[11], num_classes)
        # self.sigmoid = nn.Sigmoid()             # 数值预测一般用什么
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        out = self.relu(out)
        out = self.fc9(out)
        out = self.relu(out)
        out = self.fc10(out)
        out = self.relu(out)
        out = self.fc11(out)
        out = self.relu(out)
        out = self.fc12(out)
        out = self.relu(out)
        out = self.output(out)
        return out


def train_model(batch_size, num_epochs, orders, train_file, path, **kwargs):
    for order in orders:
        model = NeuralNet(input_size, no_hidden_units, output_size).to(device)

        # loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if 'use_ckpt' in kwargs.keys() and kwargs['use_ckpt'] is True:
            # load ckpt to training continuing
            path_ckpt = os.path.join(path, f'model_{order}.ckpt')
            checkpoint = torch.load(path_ckpt)
            model.load_state_dict(checkpoint)
            model.eval()
            model.train()

        print(f'=== training model_{order}.ckpt ===')
        # dataset
        train_dataset = AndersonChebyshevDataset(csv_file=train_file, order=order, transform=transforms.ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        total_step = len(train_dataloader)
        for epoch in range(num_epochs):
            for i, sample_batched in enumerate(train_dataloader):
                andersons = sample_batched['anderson'].reshape(-1, 8).to(device)
                polys = sample_batched['poly'].to(device).reshape(-1, 1)

                # Forward pass
                outputs = model(andersons)
                loss = criterion(outputs, polys)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                             loss.item()))
        if 'test' not in kwargs.keys() or kwargs['test'] is False:
            model_path = os.path.join(path, f'model_{order}.ckpt')
            torch.save(model.state_dict(), model_path)
    return


def predict_polynomial(batch_size, orders, test_file, predict_file, **kwargs):
    total_poly = None
    for order in orders:
        test_dataset = AndersonChebyshevDataset(csv_file=test_file, order=order, transform=transforms.ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        model_path = os.path.join(f'model_{input_size}', f'model_{order}.ckpt')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        predict = None
        with torch.no_grad():
            sample_size = 0
            L2 = 0
            for sample_batched in test_dataloader:
                andersons = sample_batched['anderson'].reshape(-1, 8)
                poly = sample_batched['poly']
                outputs = model(andersons)
                L2 += (outputs - poly).norm(2)
                outputs = outputs.numpy()
                if predict is None:
                    predict = outputs
                else:
                    predict = np.concatenate((predict, outputs), axis=0)
                sample_size += poly.size(0)
        if total_poly is None:
            total_poly = predict
        else:
            total_poly = np.concatenate((total_poly, predict), axis=1)

        RMSE = L2 * (1 / math.sqrt(sample_size))
        print(f'RMSE of the network predict {order}-th on the {sample_size} test samples: {RMSE} ')
    if 'test' not in kwargs.keys() or kwargs['test'] is False:
        # initialize predict data
        df = pd.read_csv(test_file, index_col=0)
        predict_df = df.iloc[:, :input_size]
        del df
        poly_orders = [f'poly_{order}' for order in orders]
        predict_df[poly_orders] = total_poly
        predict_df.to_csv(predict_file)
    return


def show_spectral_subplot(alpha_chebyshev, alpha_nn, plot_row, plot_col):
    plt.figure(14)
    fig_1, axs_1 = plt.subplots(plot_row, plot_col, figsize=(5 * plot_col, 2.5 * plot_row), facecolor='w',
                                edgecolor='k')
    fig_1.subplots_adjust(hspace=0.5, wspace=0.5)
    axs_1 = axs_1.ravel()
    fig_1.suptitle("AW Chebyshev vs neural network")

    for i, (alpha_c, alpha_n) in enumerate(zip(alpha_chebyshev, alpha_nn)):
        x_grid = np.linspace(x_min, x_max, 1000)
        T_c, T_n = chebyshev(x_grid, alpha_c, alpha_n)
        plt.figure(14)
        if i >= plot_row * plot_col:
            break
        axs_1[i].plot(x_grid, T_c)
        axs_1[i].plot(x_grid, T_n)
        axs_1[i].set_xlabel('$\\omega$')
        axs_1[i].set_ylabel('spectral')
        axs_1[i].set_xlim([x_min, x_max])
        axs_1[i].set_ylim([0, 0.45])
    plt.show()
    return


def show_comparisons_subplot(alpha_chebyshev, alpha_nn, plot_row, plot_col):
    plt.figure(2)
    fig_2, axs_2 = plt.subplots(plot_row, plot_col, figsize=(5 * plot_col, 2.5 * plot_row), facecolor='w',
                                edgecolor='k')
    fig_2.subplots_adjust(hspace=0.7, wspace=0.5)
    axs_2 = axs_2.ravel()
    fig_2.suptitle("Comparisons Chebyshev vs neural network")

    # samplers are 1000 in paper experiments
    for i in range(plot_row * plot_col):
        plt.figure(2)
        x = alpha_chebyshev[:, i]
        y = alpha_nn[:, i]
        axs_2[i].plot(x, y, 'ro')
        axs_2[i].set_xlabel('Neural Network coefficient')
        axs_2[i].set_ylabel('Exact coefficient')
    plt.show()
    return


if __name__ == '__main__':
    configure = 'config.json'
    with open(configure) as f:
        config = load(f)
        n_b = int(config['n_b'])
        n = int(config['n'])
        x_min = int(config['x_min'])
        x_max = int(config['x_max'])

    names = "U,ef,eis_0,eis_1,eis_2,hoppings_0,hoppings_1,hoppings_2".strip().split(',') + ["poly_" + str(i) for i in
                                                                                            range(n + 1)]

    input_size = n_b + 2
    no_hidden_units = [2 * input_size, 2 * input_size, 4 * input_size, 4 * input_size, 8 * input_size, 8 * input_size,
                       8 * input_size, 8 * input_size, 4 * input_size, 4 * input_size, 2 * input_size, 2 * input_size]
    output_size = 1  # out for predict i-th order chebyshev polynomial

    # test function
    num_epochs = 100
    batch_size = 100
    learning_rate = 0.001
    orders = [x for x in range(n+1)]  # 训练第几个系数
    # sample_size are 5000 in paper
    sample_size = 5000
    train_file = f'train_{sample_size}_{n}.csv'
    test_file = os.path.join('..', 'paras.csv')
    predict_file = f'predict_{input_size}_{n}.csv'

    # # Train the model, save model ckpt in path
    # print("=== step 1: training models ===")
    # train_model(batch_size, num_epochs, orders, train_file, path=f'model_{input_size}',
    #             use_ckpt=True)
    # # Predict data, save nn polynomial in predict_file
    # print("=== step 2: predict test data ===")
    # model = NeuralNet(input_size, no_hidden_units, output_size)
    # predict_polynomial(batch_size, orders, test_file, predict_file)
    print('=== step 3: plot results ===')
    alpha_chebyshev = pd.read_csv(test_file, index_col=0).to_numpy()[:, input_size:]
    alpha_nn = pd.read_csv(predict_file, index_col=0).to_numpy()[:, input_size:]
    assert alpha_chebyshev.shape == alpha_nn.shape
    plot_row = 8
    plot_col = 4
    show_spectral_subplot(alpha_chebyshev, alpha_nn, plot_row, plot_col)
    show_comparisons_subplot(alpha_chebyshev, alpha_nn, plot_row, plot_col)