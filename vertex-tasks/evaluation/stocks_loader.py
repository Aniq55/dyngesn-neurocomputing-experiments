import numpy as np
import pandas as pd
import pickle
import json
import urllib
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from typing import Union, Iterable, List, Optional
from graphesn.data import DynamicData
import torch

def transform_degree(x, cutoff=4):
    log_deg = np.ceil(np.log(x + 1.0))
    return np.minimum(log_deg, cutoff)


def transform_transitivity(x):
    trans = x * 10
    return np.floor(trans)


def onehot_encoding(x, unique_vals):
    E = np.zeros((len(x), len(unique_vals)))
    for i, val in enumerate(x):
        E[i, unique_vals.index(val)] = 1.0
    return E


def encode_features(X, log_degree_cutoff=4):
    X_arr = np.array(X)
    # a = transform_degree(X_arr[:, 0], log_degree_cutoff)
    # b = transform_transitivity(X_arr[:, 1])
    # A = onehot_encoding(a, range(log_degree_cutoff + 1))
    # B = onehot_encoding(b, range(11))
    # return np.concatenate((A, B), axis=1)
    X_arr = np.squeeze(X_arr, axis=0)
    return X_arr

def edge_list_from_adj(A):
    n = A.shape[0]
    
    E = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                E.append([i,j])
    return E

def perc_change(X):
    (n, T) = X.shape
    Y = (X[:,1:] - X[:,0:-1])/X[:,0:-1]
    return Y      

class StocksDatasetLoader(object): # compatible with PyTorch Temporal [?]
    """
    Parameters
    ----------
    event_id : str
        Choose to load the mention network for Roland-Garros 2017 ("rg17") or USOpen 2017 ("uo17")
    N : int <= 1000
        Number of most popular nodes to load. By default N=1000. Each snapshot contains the graph induced by these nodes.
    feature_mode : str
        None : load raw degree and transitivity node features
        "encoded" : load onehot encoded degree and transitivity node features
        "diagonal" : set identity matrix as node features
    target_offset : int
        Set the snapshot offset for the node labels to be predicted. By default node labels for the next snapshot are predicted (target_offset=1).
    """

    def __init__(self, event_id="rg17", N=None, feature_mode="encoded", target_offset=1):
        self.N = N
        self.target_offset = target_offset
        if event_id in ["rg17", "uo17"]:
            self.event_id = event_id
        else:
            raise ValueError(
                "Invalid 'event_id'! Choose 'rg17' or 'uo17' to load the Roland-Garros 2017 or the USOpen 2017 Twitter tennis dataset respectively."
            )
        if feature_mode in [None, "diagonal", "encoded"]:
            self.feature_mode = feature_mode
        else:
            raise ValueError(
                "Choose feature_mode from values [None, 'diagonal', 'encoded']."
            )
        
        self._read_local_data()
            
            
    def _read_local_data(self, perc_change_flag=False):
        
        sample_interval = 28 # sampling interval in days
        m = 5  # width of correlation window
        
        # pickle dump df
        file_path = '/home/chri6578/Documents/GG_SPP/dataset/stocks_pd.pickle'

        # Loading dataset from pickle
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            
        # Sampling

        
        d_sample = df[0:-1:sample_interval]
        df_ = pd.DataFrame(d_sample)
        X = np.array(df_).T

        E = []
        (n, T) = X.shape
        if perc_change_flag:
            X = perc_change(X)
            (n, T) = X.shape
        
        thresh = 0.5
        for i in range(int(T-m)):
            x_i = X[:, i:i+m]
            
            A_i = np.eye(n, dtype='float32')
            
            for v_i in range(n):
                for v_j in range(v_i+1, n):
                    A_i[v_i][v_j] = np.abs(np.corrcoef(x_i[v_i], x_i[v_j])[0,1]) > thresh
                    A_i[v_j][v_i] = A_i[v_i][v_j]
            
            E.append(edge_list_from_adj(A_i))
        
        X = X.T[m:]
        (T, n) = X.shape
        self.N = n
        X = X.reshape(T,n,1)
        T = len(E)
        stocks_data = {}

        for t in range(1,T-1):
            stocks_data[str(t-1)] = {
                'index': t-1,
                'edges': E[t],
                'weights' : list(np.ones((len(E[t]),))),
                'y' : X[t+1],
                'X' : np.array([X[t],X[t-1]]).T
            }
            
            # X: stock value at time t and t-1, y: stock value at time t+1

        stocks_data['time_periods'] = T-2
        
        self._dataset = stocks_data       

    def _get_edges(self):
        edge_indices = []
        self.edges = []
        for time in range(self._dataset["time_periods"]):
            E = np.array(self._dataset[str(time)]["edges"])
            if self.N != None:
                selector = np.where((E[:, 0] < self.N) & (E[:, 1] < self.N))
                E = E[selector]
                edge_indices.append(selector)
            self.edges.append(E.T)
        self.edge_indices = edge_indices

    def _get_edge_weights(self):
        edge_indices = self.edge_indices
        self.edge_weights = []
        for i, time in enumerate(range(self._dataset["time_periods"])):
            W = np.array(self._dataset[str(time)]["weights"])
            if self.N != None:
                W = W[edge_indices[i]]
            self.edge_weights.append(W)

    def _get_features(self): # something happens here.
        self.features = []
        for time in range(self._dataset["time_periods"]):
            X = np.array(self._dataset[str(time)]["X"])
            if self.N != None:
                X = X[: self.N]
            if self.feature_mode == "diagonal":
                X = np.identity(X.shape[0])
            elif self.feature_mode == "encoded":
                X = encode_features(X)
            self.features.append(X)

    def _get_targets(self):
        self.targets = []
        T = self._dataset["time_periods"]
        for time in range(T):
            # predict node degrees in advance
            snapshot_id = min(time + self.target_offset, T - 1)
            y = np.array(self._dataset[str(snapshot_id)]["y"])
            
            # # logarithmic transformation for node degrees
            # y = np.log(1.0 + y)
            # if self.N != None:
            #     y = y[: self.N]
                
            self.targets.append(y)

    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Returning the StocksDataset data iterator.

        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)*
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = DynamicGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset

def stocks_dataset(event_id: str = 'rg17', num_nodes: Optional[int] = None, feature_mode: str = 'encoded',
                target_offset: int = 1) -> DynamicData: # compatible with DynGESN
    """
    Stock Dataset
    
    :param num_nodes: Select top nodes (optional, default all)
    :param feature_mode: Can be None for raw features, or 'encoded' for one hot encoding, or 'diagonal' for identity matrix
    :param target_offset: Prediction off-set for node labels (default 1)
    :return: A single dynamic graph
    """
    loader = StocksDatasetLoader(N = num_nodes, feature_mode= feature_mode, target_offset= target_offset)
    dataset = loader.get_dataset()
    return DynamicData(edge_index=[torch.from_numpy(edge_index) for edge_index in dataset.edge_indices],
                    edge_weight=[torch.from_numpy(edge_weight.astype('float32')) for edge_weight in
                                    dataset.edge_weights],
                    x=torch.stack([torch.from_numpy(x.astype('float32')) for x in dataset.features], dim=0),
                    y=torch.stack(
                        [torch.from_numpy(y.astype('float32')).unsqueeze(dim=-1) for y in dataset.targets], dim=0))

# d = stocks_dataset(feature_mode=None, target_offset=1)




