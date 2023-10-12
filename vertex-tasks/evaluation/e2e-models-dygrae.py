"""
e2e-models-dygrae.py
Copyright (C) 2022, Domenico Tortorella
Copyright (C) 2022, University of Pisa

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import argparse
import math
from copy import deepcopy
from statistics import mean, stdev
from time import perf_counter

import torch
import torch.nn.functional as F
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, TwitterTennisDatasetLoader, PedalMeDatasetLoader, WikiMathsDatasetLoader
from torch_geometric_temporal.nn.recurrent import *
from torch_geometric_temporal.signal import temporal_signal_split

from stocks_loader import *


def get_dataset(name, device):
    if name == 'chickenpox':
        dataset = ChickenpoxDatasetLoader().get_dataset(lags=1)
    elif name == 'tennis':
        dataset = TwitterTennisDatasetLoader(feature_mode='encoded').get_dataset()
    elif name == 'pedalme':
        dataset = PedalMeDatasetLoader().get_dataset(lags=1)
    elif name == 'wikimath':
        dataset = WikiMathsDatasetLoader().get_dataset(lags=1)
    elif name == 'stocks':
        dataset = StocksDatasetLoader(feature_mode="encoded", target_offset=1).get_dataset()
    else:
        raise ValueError('Wrong dataset name')
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)
    train_dataset, valid_dataset = temporal_signal_split(train_dataset, train_ratio=8/9)
    return [snapshot.to(device) for snapshot in train_dataset], [snapshot.to(device) for snapshot in valid_dataset], [snapshot.to(device) for snapshot in test_dataset]


class RecurrentGCN(torch.nn.Module):
    def __init__(self, model_name, node_features, hidden_features, num_nodes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DyGrEncoder(conv_out_channels=node_features, conv_num_layers=1, conv_aggr="mean", lstm_out_channels=hidden_features, lstm_num_layers=1)
        self.linear = torch.nn.Linear(hidden_features, 1)

    def forward(self, x, edge_index, edge_weight, h=None, c=None):
        h_tilde, h, c = self.recurrent(x, edge_index, edge_weight, h, c)
        y = F.relu(h_tilde)
        y = self.linear(y)
        return h, c, y


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--model', help='model name', default='dygrae')
parser.add_argument('--units', help='reservoir units per layer', type=int, default=32)
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--epochs', help='number of epochs', type=int, default=100)
parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
parser.add_argument('--trials', help='number of trials', type=int, default=10)
args = parser.parse_args()

device = torch.device(args.device)
train_dataset, valid_dataset, test_dataset = get_dataset(args.dataset, device)
num_nodes, num_features = train_dataset[0].x.shape

train_time, train_mse, valid_mse, test_time, test_mse = [], [], [], [], []

for _ in range(args.trials):
    model = RecurrentGCN(args.model.lower(), num_features, args.units, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mse = math.inf
    tic = perf_counter()
    for _ in range(args.epochs):
        model.train()
        cost = 0
        h, c = None, None
        for time, snapshot in enumerate(train_dataset):
            h, c, y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
            cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        val_cost = 0
        for time, snapshot in enumerate(valid_dataset):
            h, c, y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
            val_cost += torch.mean((y_hat - snapshot.y) ** 2)
        val_cost = val_cost / (time + 1)
        if val_cost < best_val_mse:
            best_val_mse = val_cost
            state_dict = deepcopy(model.state_dict())
    toc = perf_counter()
    train_mse.append(cost.item())
    valid_mse.append(val_cost.item())
    train_time.append((toc - tic) * 1000)
    model.load_state_dict(state_dict)

    model.eval()
    tic = perf_counter()
    cost = 0
    h, c = None, None
    for snapshot in train_dataset:
        h, c, y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    for time, snapshot in enumerate(valid_dataset):
        h, c, y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
    for time, snapshot in enumerate(test_dataset):
        h, c, y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    toc = perf_counter()
    test_mse.append(cost.item())
    test_time.append((toc - tic) * 1000)

print(args.model,
    f'{mean(np.sqrt(train_mse)):.3f} ± {stdev(np.sqrt(train_mse)):.3f}',
    f'{mean(np.sqrt(test_mse)):.3f} ± {stdev(np.sqrt(test_mse)):.3f}',
    f'{mean([t / args.epochs for t in train_time]):.5f} ± {stdev([t / args.epochs for t in train_time]):.5f}',
    f'{mean(train_time):.5f} ± {stdev(train_time):.5f}',
    f'{mean(test_time):.5f} ± {stdev(test_time):.5f}',
    sep='\t')
