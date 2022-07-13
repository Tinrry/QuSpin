import pandas as pd

import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_b = 6  # in L = 6 Anderson model
n = 256
x_min = -25
x_max = 25
names = "U,ef,eis_0,eis_1,eis_2,hoppings_0,hoppings_1,hoppings_2".strip().split(',') + ["poly_" + str(i) for i in
                                                                                        range(n + 1)]

input_size = n_b + 2
no_hidden_units = [2 * input_size, 2 * input_size, 4 * input_size, 4 * input_size, 8 * input_size, 8 * input_size,
                   8 * input_size, 8 * input_size, 4 * input_size, 4 * input_size, 2 * input_size, 2 * input_size]
num_classes = 1  # out for predict i-th order chebyshev polynomial
num_epochs = 5
batch_size = 2
learning_rate = 0.001
orders = [0]  # 训练第几个系数
sample_size = 2
train_file = f'train_{sample_size}_{n}.csv'
test_file = f'paras.csv'
predict_file = f'predict_{input_size}_{n}.csv'


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


model = NeuralNet(input_size, no_hidden_units, num_classes).to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# Train the model
for order in orders:
    # dataset
    train_dataset = AndersonChebyshevDataset(csv_file=train_file, order=order, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, sample_batched in enumerate(train_dataloader):
            print(f"{i} {sample_batched['anderson'].size()} {sample_batched['poly'].size()}")
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
    torch.save(model.state_dict(), f'model_{order}.ckpt')

test_dataset = AndersonChebyshevDataset(csv_file=test_file, order=order, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Test model
df = pd.read_csv(test_file).iloc[:, [:input_size]]
df.to_csv(predict_file)
for order in orders:
    model = NeuralNet(input_size, no_hidden_units, num_classes)
    model.load_state_dict(torch.load(f'model_{order}.ckpt'))
    model.eval()
    predict = []
    with torch.no_grad():
        for sample_batched in test_dataloader:
            andersons = sample_batched['anderson'].reshape(-1, 8)
            poly = sample_batched['poly']
            outputs = model(andersons)
            predict.extend(outputs)

    df = pd.read_csv(predict_file)
    df[f'poly_{order}'] = predict
    df.to_csv(predict_file)

print(df.to_string())

# plot
from utils import scale_up, scale_down

alpha_chebyshev_df = pd.read_csv(test_file)

def chebyshev():
    x_grid = np.linspace(x_min, x_max, 1000)
    T = np.zeros((len(x_grid), n + 1))
    T[:, 0] = np.ones((len(x_grid), 1)).T
    z_grid = scale_down(x_grid, x_min, x_max)
    T[:, 1] = z_grid.T
    for i in range(1, n):
        T[:, i + 1] = 2 * z_grid * T[:, i] - T[:, i - 1]
        T_chebyshev = T @