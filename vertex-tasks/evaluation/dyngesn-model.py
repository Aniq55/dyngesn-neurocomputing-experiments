"""
dyngesn-model.py
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
from statistics import mean, stdev
from time import perf_counter

import torch
from torch.nn.functional import linear

from graphesn import DynamicGraphReservoir, initializer, Readout
from graphesn.dataset import chickenpox_dataset, twitter_tennis_dataset, pedalme_dataset, wiki_maths_dataset
from graphesn.util import compute_dynamic_graph_alpha, compute_dynamic_weighted_graph_alpha
from stocks_loader import *

def prepare_data(name, device, weighted=False, lags=1):
    if name == 'chickenpox':
        data = chickenpox_dataset(target_lags=lags).to(device)
        # alpha = compute_dynamic_graph_alpha(data.edge_index)
        alpha = 5.4
        return data.edge_index, None, data.x, data.y, alpha, data.num_timesteps
    elif name == 'tennis':
        # data = twitter_tennis_dataset(feature_mode='encoded', target_offset=lags).to(device) # why encoded?
        data = twitter_tennis_dataset(feature_mode=None, target_offset=lags).to(device)
        alpha = compute_dynamic_weighted_graph_alpha(data) if weighted else compute_dynamic_graph_alpha(data)
    elif name == 'pedalme':
        data = pedalme_dataset(target_lags=lags).to(device)
        # alpha = compute_dynamic_graph_alpha(data.edge_index, data.edge_weight if weighted else None)
        alpha = 15.0
    elif name == 'wikimath':
        data = wiki_maths_dataset(target_lags=lags).to(device)
        # print(data.y)
        alpha = 57.9
        # alpha = compute_dynamic_graph_alpha(data.edge_index, data.edge_weight if weighted else None)
    elif name == 'stocks':
        data = stocks_dataset(feature_mode="encoded", target_offset=lags).to(device)
        alpha = compute_dynamic_weighted_graph_alpha(data) if weighted else compute_dynamic_graph_alpha(data)
    else:
        raise ValueError('Wrong dataset name')
    return data.edge_index, data.edge_weight if weighted else None, data.x, data.y, alpha, data.num_timesteps


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name')
parser.add_argument('--units', help='reservoir units per layer', type=int, default=32)
parser.add_argument('--device', help='device for torch computations', default='cpu')
parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, default=0.9)
parser.add_argument('--leakage', help='leakage constant', type=float, default=0.9)
parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-3])
parser.add_argument('--trials', help='number of trials', type=int, default=10)
args = parser.parse_args()

device = torch.device(args.device)
edge_index, edge_weight, x, y, alpha, T = prepare_data(args.dataset, device)
T_train, T_valid = int(T * 0.8), int(T * 0.9)
print(f'alpha = {alpha:.2f}')

train_time, train_mse, test_time, test_mse = [], [], [], []


def validate_on(weights, Xval, yval):
    return torch.mean((linear(Xval, weights[0], weights[1]) - yval) ** 2)


y_in_train = []
y_in_valid = []
y_in_test = []

y_out_train = []
y_out_valid = []
y_out_test = []

flag_ = True
for _ in range(args.trials):
    reservoir = DynamicGraphReservoir(num_layers=1, in_features=x.shape[-1], hidden_features=args.units, return_sequences=True)
    reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=args.sigma / alpha),
                                    input=initializer('uniform', scale=1),
                                    leakage=args.leakage)
    reservoir.to(device)
    readout = Readout(num_features=args.units, num_targets=1)

    tic = perf_counter()
    X = reservoir(edge_index=edge_index[:T_valid] if isinstance(edge_index, list) else edge_index, input=x[:T_valid])
    readout.fit(data=(X[:T_train].view(-1, X.shape[-1]), y[:T_train].view(-1, y.shape[-1])), regularization=args.ld,
                validate=lambda weights: validate_on(weights, X[T_train:T_valid].view(-1, X.shape[-1]), y[T_train:T_valid].view(-1, y.shape[-1])))
    toc = perf_counter()
    mse_loss = torch.mean((readout(X[:T_train].view(-1, X.shape[-1])) - y[:T_train].view(-1, y.shape[-1])) ** 2)
    if flag_:
        y_in_train.append(np.array(readout(X[:T_train].view(-1, X.shape[-1])).detach().cpu()))
        y_out_train.append(np.array(y[:T_train].view(-1, y.shape[-1]).detach().cpu()))
    train_mse.append(mse_loss.item())
    train_time.append((toc - tic) * 1000)

    tic = perf_counter()
    X = reservoir(edge_index=edge_index, input=x)
    y_hat = readout(X[T_valid:].view(-1, X.shape[-1]))
    mse_loss = torch.mean((y_hat - y[T_valid:].view(-1, y.shape[-1])) ** 2)
    if flag_:
        y_in_test.append(np.array(y[T_valid:].view(-1, y.shape[-1]).detach().cpu()))
        y_out_test.append(y_hat.detach().cpu())
    toc = perf_counter()
    test_mse.append(mse_loss.item())
    test_time.append((toc - tic) * 1000)
    
    flag_ = False

print(f'dyngesn:{args.dataset}',
    f'{mean(np.sqrt(train_mse)):.3f} ± {stdev(np.sqrt(train_mse)):.3f}',
    f'{mean(np.sqrt(test_mse)):.3f} ± {stdev(np.sqrt(test_mse)):.3f}',
    f'{mean(train_time):.5f} ± {stdev(train_time):.5f}',
    f'{mean(test_time):.5f} ± {stdev(test_time):.5f}',
    sep='\t')


dict_out = {
    "y_in_train" : y_in_train, 
    "y_in_valid": y_in_valid, 
    "y_in_test" : y_in_test,
    "y_out_train" : y_out_train,
    "y_out_valid" : y_out_valid,
    "y_out_test" : y_out_test 
}

import pickle

file_path =  './results/dyngesn_dict.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(dict_out, file)