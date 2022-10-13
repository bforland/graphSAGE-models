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

def train_test(model, train_loader, train_data, val_data, test_data, config, learning_rate=0.01, e_threshold=10, num_epochs=50, min_acc=0.005, epoch_save=False):
    t0 = time.time()
    #model = Model(**model_params)
    

    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2, threshold=0.001, min_lr = 0.001)
    k=0
    losses = []
    for epoch in range(1, num_epochs+1):
        loss = train(model,optimizer, config=config, loader=train_loader)
        val_loss = test(model, val_data, config)
        test_loss = test(model, test_data, config)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},  '
              f'Val: {val_loss:.4f} , LR: {cur_lr}')
        losses.append({'train_loss': loss, 
                      'val_loss': val_loss,
                      'test_loss': test_loss,
                      'lr': 'curr_lr'})
        
        if epoch_save:
            if (epoch%25==0) or (epoch==1):
                print('Saving model at '+str(epoch)+' epoch')
                print(f"/dbfs/tmp/{model_config.model_name}_EPOCH{epoch}.pkl")
                with open(f"/dbfs/tmp/{model_config.model_name}_EPOCH{epoch}.pkl", "wb") as f:
                    pickle.dump(model, f)
#     test_pred = predict(model, test_data, config)
#     accuracy = accuracy_score(test_data['customer', 'recipe'].edge_label.cpu().numpy(),
#                                 test_pred.sigmoid().cpu().numpy() > 0.5 )
    return pd.DataFrame({'loss': loss,
            'val_loss': val_loss,
            'test_loss': test_loss, 
            #'accuracy': accuracy,
            'time': (time.time()-t0)/60}, index=[0]), model, losses