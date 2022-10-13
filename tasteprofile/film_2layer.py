#import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
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
class ModelConfig:
    dimensions: int = 64 # dimension of final embeddings 
    hidden_channels: int = 64 # dimension of hidden layers
    out_channels: int = 1 # output layer for link predictor
    criterion: Criterion = Criterion.BCE_LOSS # loss function
    embeddings_combine_strategy: EmbeddingsCombineStrategy =  EmbeddingsCombineStrategy.CONCAT
    customer_features: str = "default" # default or random
    model_name: str = None
    
    def __post_init__(self):
        self.model_name = f"hetero_{self.customer_features}_cf_{self.criterion.value}_{self.embeddings_combine_strategy.value}_{self.dimensions}"+self.model_name
        
def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

def facebook_infersent(z_dict , edge_label_index):
    
    # Embedding concatenation feature
    #row, col = edge_label_index
    #z_cat = torch.cat([z_dict["customer"][row], z_dict["recipe"][col]], dim=-1)
    
    # Embedding intereaction feature
    h_src = z_dict["customer"][edge_label_index[0]] 
    h_dst = z_dict["recipe"][edge_label_index[1]]
    z_int = (h_src * h_dst)
    
    # Embedding difference term
    #z_dif = torch.abs(h_src - h_dst)
    
    # Combined all features to one
    #z = torch.cat([z_cat, z_int, z_dif], dim=-1)
    
    return z_int 

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
        self.conv2 = conv((-1, -1), hidden_channels)
        #self.conv3 = conv((-1, -1), hidden_channels)
        #self.conv4 = conv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        #x = self.conv2(x, edge_index).relu()
        #x = self.conv3(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class FiLMEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, conv=FiLMConv):
        super().__init__()
        self.conv1 = conv((-1, -1), hidden_channels)
        self.conv2 = conv((-1, -1), hidden_channels)
        self.conv3 = conv((-1, -1), hidden_channels)
        self.conv4 = conv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(1*hidden_channels, hidden_channels)
        #self.lin = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z):
        z = self.lin1(z).relu()
        #z = self.lin2(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, conv=SAGEConv):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, conv)
        #self.encoder = FiLMEncoder(hidden_channels, hidden_channels, conv)
        #self.encoder = to_hetero_with_bases(self.encoder, pytorch_hetero_graph.metadata(), 1)
        self.encoder = to_hetero(self.encoder, pytorch_hetero_graph.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, 
                combine_mode=EmbeddingsCombineStrategy.CONCAT):
        z_dict = self.encoder(x_dict, edge_index_dict)
        z = facebook_infersent(z_dict, edge_label_index)
        #z = combine_embeddings(z_dict, edge_label_index, mode=combine_mode)
        return self.decoder(z)

# def get_negative_edges(torch_graph):
#     neg_samples = structured_negative_sampling(torch_graph['customer', 'recipe'].edge_index, num_nodes=len(torch_graph.x_dict['recipe']))
#     neg_edge_label_index = torch.stack((neg_samples[0], neg_samples[2]), dim=0)
#     return neg_edge_label_index
    

def train(model, optimizer, config:ModelConfig, data=None, loader=None):
    
    model.train()
    if loader is None:
        optimizer.zero_grad()
        pred = model(data.x_dict, data.edge_index_dict,
                      data['customer', 'recipe'].edge_label_index, 
                     combine_mode=config.embeddings_combine_strategy)
        
        loss = F.binary_cross_entropy_with_logits(pred, data["customer", "recipe"].edge_label.float())

        loss.backward()
        optimizer.step()
        return float(loss)
    else:
        total_examples = total_loss = 0
        for batch in tqdm(loader):
            optimizer.zero_grad()
            batch_size = batch['customer'].batch_size
            pred = model(batch.x_dict, batch.edge_index_dict,
                        batch["customer", "recipe"].edge_index, config.embeddings_combine_strategy)
            loss = F.binary_cross_entropy_with_logits(pred, batch["customer", "recipe"].edge_label.float())
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
        return total_loss / total_examples
    


@torch.no_grad()
def test(model, data, config:ModelConfig):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'recipe'].edge_label_index, 
                combine_mode=config.embeddings_combine_strategy)
    target = data['customer', 'recipe'].edge_label.float()
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return float(loss)
 
@torch.no_grad()
def predict(model, data, config):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'recipe'].edge_label_index,
                combine_mode=config.embeddings_combine_strategy)
    return pred