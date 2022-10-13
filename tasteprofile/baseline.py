#import umap.umap_ as umap
import pandas as pd
import random
from dataclasses import dataclass
import time
from tqdm import tqdm
from enum import Enum
from typing import Dict

import networkx as nx
import numpy as np

import torch
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
from torch_geometric.utils import structured_negative_sampling
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch_geometric.nn import SAGEConv, GATConv, FiLMConv
from torch_geometric.nn import to_hetero, to_hetero_with_bases
from torch_geometric.loader import HGTLoader
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric
from torch.nn import Linear, CosineEmbeddingLoss
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

torch.manual_seed(0)

import gc
from sklearn.metrics.pairwise import cosine_similarity
import logging

class EmbeddingsCombineStrategy(Enum):
    CONCAT = "concat"
    PIECEWISE_PRODUCT = "piecewise_product"
    COSINE_SIMILARITY = "cosine_similarity"

criterion_dict = {
               "bce_logits": torch.nn.BCEWithLogitsLoss(), 
               "hinge": torch.nn.HingeEmbeddingLoss(),
               "cosine": torch.nn.CosineEmbeddingLoss(),
                "ranking": torch.nn.MarginRankingLoss()
}

class Criterion(Enum):
    BCE_LOSS = "bce_logits"
    HINGE_EMBEDDING_LOSS = "hinge"
    COSINE_EMBEDDING_LOSS = "cosine"
    MARGIN_RANKING_LOSS = "ranking"

@dataclass
@dataclass
class ModelConfig:
    dimensions: int = 64 # dimension of final embeddings 
    hidden_channels: int = 64 # dimension of hidden layers
    out_channels: int = 1 # output layer for link predictor
    criterion: Criterion = Criterion.BCE_LOSS # loss function
    embeddings_combine_strategy: EmbeddingsCombineStrategy =  EmbeddingsCombineStrategy.CONCAT
    customer_features: str = "default" # default or random
    batch_size: int = 4096 # Number of batches
    num_epochs: int = 1
    learning_rate: float = 0.01
    model_name: str = 'base_line'
    
    def __post_init__(self):
        self.model_name = f"{self.model_name}_hetero_{self.customer_features}_cf_{self.criterion.value}_{self.embeddings_combine_strategy.value}_{self.dimensions}_bs{self.batch_size}_ep{self.num_epochs}_lr{str(self.learning_rate).split('.')[0]}p{str(self.learning_rate).split('.')[1]}"
        
def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

def combine_embeddings(z_dict , edge_label_index, mode=EmbeddingsCombineStrategy.CONCAT):
    if mode == EmbeddingsCombineStrategy.CONCAT:
        row, col = edge_label_index
        z = torch.cat([z_dict["customer"][row], z_dict["recipe"][col]], dim=-1)
    elif mode == EmbeddingsCombineStrategy.PIECEWISE_PRODUCT:
        h_src = z_dict["customer"][edge_label_index[0]]
        h_dst = z_dict["recipe"][edge_label_index[1]]
        z = (h_src * h_dst)
    elif mode == EmbeddingsCombineStrategy.COSINE_SIMILARITY:
        h_src = z_dict["customer"][edge_label_index[0]]
        h_dst = z_dict["recipe"][edge_label_index[1]]
        cos = torch.nn.CosineSimilarity(dim=1)
        z = cos(h_src, h_dst)
        
    else:
        raise("Invalid input for combining embeddings")
    return z             

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, conv=SAGEConv):
        super().__init__()
        self.conv1 = conv((-1, -1), hidden_channels)
        self.conv2 = conv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z):
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, conv=SAGEConv):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, conv)
        self.encoder = to_hetero(self.encoder, pytorch_hetero_graph.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, 
                combine_mode=EmbeddingsCombineStrategy.CONCAT):
        z_dict = self.encoder(x_dict, edge_index_dict)
        z = combine_embeddings(z_dict, edge_label_index, mode=combine_mode)
        return self.decoder(z)

def define_data(graph, train_data):
    global pytorch_hetero_graph
    global data
    
    pytorch_hetero_graph = graph
    data = train_data
    return
# def get_negative_edges(torch_graph):
#     neg_samples = structured_negative_sampling(torch_graph['customer', 'recipe'].edge_index, num_nodes=len(torch_graph.x_dict['recipe']))
#     neg_edge_label_index = torch.stack((neg_samples[0], neg_samples[2]), dim=0)
#     return neg_edge_label_index