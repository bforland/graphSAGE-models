# Databricks notebook source
# MAGIC %sh
# MAGIC pip install networkx==2.8.6
# MAGIC pip install -U pip
# MAGIC pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# MAGIC pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# MAGIC pip install torch-geometric
# MAGIC pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# MAGIC pip install s3fs

# COMMAND ----------

import sys
import importlib
import pickle

import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import SAGEConv

# COMMAND ----------

sys.path.append("/Workspace/Repos/blake.forland@hellofresh.com/graphSAGE-models")

# COMMAND ----------

from tasteprofile_models import baseline
#from tasteprofile_models
from tasteprofile_models import utils

# COMMAND ----------

importlib.reload(baseline)
importlib.reload(utils)

# COMMAND ----------

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
from sklearn.preprocessing import RobustScaler
from torch_geometric.utils import structured_negative_sampling
torch.manual_seed(0)

def to_pytorch_graph(nx_graph: nx.Graph):
    new_graph = nx.DiGraph()
    nodes = [(node_id, node_data)  if type(node_id) == str else (f"C{int(node_id)}", node_data) for node_id, node_data in nx_graph.nodes(data=True)]
    new_graph.add_nodes_from(nodes)
    print(f"Nodes: {len(new_graph.nodes)}")
    edges = [("C" + str(y), x, {"edge_label": min(1, labels["times_ordered"])}) for x, y, labels in nx_graph.edges(data=True)]
    new_graph.add_edges_from(edges)
    print(f"Edges: {len(new_graph.edges)}")
    new_graph, rf, cf = detach_features(new_graph)
    pytorch_homo_graph = from_networkx(new_graph)
    pytorch_hetero_graph = convert_homo_to_hetero(new_graph, pytorch_homo_graph)
    pytorch_hetero_graph = T.ToUndirected()(pytorch_hetero_graph)
    del pytorch_hetero_graph["recipe", "rev_to", "customer"].edge_label
    del pytorch_hetero_graph["customer"].num_nodes
    del pytorch_hetero_graph["recipe"].num_nodes

    #pytorch_hetero_graph["customer"].x = torch.tensor(cf)
    pytorch_hetero_graph["customer"].x = get_random_customer_features(len(cf))
    pytorch_hetero_graph["recipe"].x =  torch.tensor(rf)
    return new_graph, pytorch_hetero_graph



    
def get_recipe_features(nx_graph: nx.Graph) -> np.ndarray:
    recipe_features = [
        node_data["embedding"]
        for node_id, node_data in nx_graph.nodes(data=True) 
        if node_data.get("type") == "recipe"
    ]
    feature_array = np.array(recipe_features).astype("float32")
    print(feature_array.shape)
    return feature_array

def get_customer_features(nx_graph: nx.Graph) -> np.ndarray:
    customer_features = []
    for _, node_data in nx_graph.nodes(data=True):
        if node_data.get("type") == "customer":
            c = node_data.copy()
            c.pop("type")
            customer_features.append(list(c.values()))
    feature_array = np.array(customer_features).astype("float32")
    scaler = RobustScaler()
    feature_array = scaler.fit_transform(feature_array)
    print(feature_array.shape)
    return customer_features


def detach_features(nx_graph: nx.Graph):
    recipe_features = get_recipe_features(nx_graph)
    customer_features = get_random_customer_features(num_customers=969725)
    
    for _, node_data in nx_graph.nodes(data=True):
        for k in list(node_data.keys()):
            if k != "type":
                node_data.pop(k)
    return nx_graph, recipe_features, customer_features


def convert_homo_to_hetero(nx_graph: nx.Graph, pytorch_graph):
    nodes = nx_graph.nodes(data=True)
    edges = [(nodes[a]["type"], "to", nodes[b]["type"]) for a, b in nx_graph.edges]

    node_types = ["customer", "recipe"]
    edge_types = [("customer", "to", "recipe")]


    d_node_types = dict(zip(node_types, list(range(len(node_types)))))
    d_edge_types = dict(zip(edge_types, list(range(len(edge_types)))))

    hetero_graph = pytorch_graph.to_heterogeneous(
        node_type=torch.tensor([d_node_types.get(i) for i in pytorch_graph.node_stores[0].type]),
        edge_type=torch.tensor([d_edge_types.get(i) for i in edges]),
        node_type_names=node_types,
        edge_type_names=edge_types,
    )
    return hetero_graph

def get_random_customer_features(num_customers):
    return torch.nn.init.xavier_normal_(torch.ones(num_customers, 495))
  
def sample_edges(nx_graph, prop=0.8):
    # get fewer edges <- randomly reduce to 50M edges
    all_edges = list(nx_graph.edges)
    sampled_edges = random.sample(all_edges, int(len(all_edges) * (1-prop)))
    nx_graph.remove_edges_from(sampled_edges)
    return nx_graph
  
def sample_customer_nodes(nx_graph, customer_nodes, num_nodes=10000):
    sampled_nodes = random.sample(customer_nodes, len(customer_nodes)-num_nodes)
    nx_graph.remove_nodes_from(sampled_nodes)
    return nx_graph

# COMMAND ----------

nx_graph_sp = nx.read_gpickle("/dbfs/tmp/customer_recipe_features_2022-W01_2022-W24_2022-10-11.gpickle")

# COMMAND ----------

# Also, do we have a pickle somewhere that has customer ID and most recent preset ?

# COMMAND ----------

nodes_to_remove = [node_id for node_id, node_data in nx_graph_sp.nodes(data=True) if node_data.get('type', None) is None ]

# COMMAND ----------

nx_graph_sp.remove_nodes_from(nodes_to_remove)

# COMMAND ----------

customer_ids = []
for node_id, node_data in nx_graph_sp.nodes(data=True):
    customer_ids.append(node_data.get('type'))
    if node_data.get('type') == 'customer':
        customer_ids.append(node_id)

# COMMAND ----------

set(customer_ids)

# COMMAND ----------

# For mapping node ids in networkx graph to pytorch graph
#new_graph = nx.read_gpickle("/dbfs/tmp/nx_graph_for_pytorch_hetero.gpickle")
#customer_node_map = [int(''.join(node_id[1:])) for node_id, node_data in new_graph.nodes(data=True) if node_data.get('type') == 'customer']
#recipe_node_map = [node_id for node_id, node_data in new_graph.nodes(data=True) if node_data.get('type') == 'recipe'] 

# COMMAND ----------

nx_hetero_sp, pytorch_hetero_graph = to_pytorch_graph(nx_graph_sp)

# COMMAND ----------

with open(f"/dbfs/tmp/customer_recipe_features_2022-W01_2022-W24_2022-10-11_pytorch_hetero.pkl", "wb") as f:
    pickle.dump(pytorch_hetero_graph, f)

# COMMAND ----------

with open(f"/dbfs/tmp/customer_random_recipe_scaled_metadata_2022-10-18_pytorch_hetero_64k.pkl", "rb") as f:
    pytorch_hetero_graph = pickle.load(f)

# COMMAND ----------

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.25,
    num_test=0.25,
    add_negative_train_samples=False,
    neg_sampling_ratio=0.0,
    edge_types=[('customer', 'to', 'recipe')],
    rev_edge_types=[('recipe', 'rev_to', 'customer')],
)(pytorch_hetero_graph)

# COMMAND ----------

with open(f"/dbfs/tmp/train_val_test.pkl", "wb") as f:
    pickle.dump({'train_data': train_data,
                 'val_data': val_data,
                 'test_data': test_data}, f)

# COMMAND ----------

train_loader = HGTLoader(
    train_data,
    # Sample 2000 nodes per type and per iteration for 4 iterations
    num_samples={'customer': [200, 200, 200, 200], 'recipe': [500, 500, 500, 500]},
    batch_size=128,
    input_nodes=('customer'),
    #transform=NormalizeFeatures(),
)

# COMMAND ----------

{key: [32] * 3 for key in train_data.node_types}

# COMMAND ----------

next(iter(train_loader))

# COMMAND ----------

EmbeddingsCombineStrategy = baseline.EmbeddingsCombineStrategy("piecewise_product")

# COMMAND ----------

# MAGIC %md # Model

# COMMAND ----------

#import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch_geometric.loader import HGTLoader
from torch.nn import Linear, CosineEmbeddingLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
torch.manual_seed(0)

# COMMAND ----------

from enum import Enum
from typing import Dict

class EmbeddingsCombineStrategy(Enum):
    CONCAT = "concat"
    PAIR_PRODUCT = "pair_product"
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
        self.model_name = f"hetero_{self.customer_features}_cf_{self.criterion.value}_{self.embeddings_combine_strategy.value}_{self.dimensions}"
        
def combine_embeddings(z_dict , edge_label_index, mode=EmbeddingsCombineStrategy.CONCAT):
    if mode == EmbeddingsCombineStrategy.CONCAT:
        row, col = edge_label_index
        z = torch.cat([z_dict["customer"][row], z_dict["recipe"][col]], dim=-1)
    elif mode == EmbeddingsCombineStrategy.PAIR_PRODUCT:
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

# COMMAND ----------

# model definition


device = "cuda"
criterion = CosineEmbeddingLoss()
def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, conv=SAGEConv):
        super().__init__()
        self.conv1 = conv((-1, -1), hidden_channels, normalize=True)
        self.conv2 = conv((-1, -1), hidden_channels, normalize=True)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

# Link Predictor
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z):
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

# E2E model for link prediction
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, conv=SAGEConv):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, conv)
        self.encoder = to_hetero(self.encoder, pytorch_hetero_graph.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, combine_mode=EmbeddingsCombineStrategy.CONCAT):
        z_dict = self.encoder(x_dict, edge_index_dict)
        z = combine_embeddings(z_dict, edge_label_index, mode=combine_mode)
        return self.decoder(z)

# E2E Model trained for minimizing cosine embedding loss - no link prediction 
class EmbeddingsModel(torch.nn.Module):
    def __init__(self, hidden_channels, conv=SAGEConv):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, conv)
        self.encoder = to_hetero(self.encoder, pytorch_hetero_graph.metadata(), aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        
        return z_dict
    

def train_link_predictor(model, optimizer, config:ModelConfig, data=None, loader=None):
    
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
                        batch["customer", "recipe"].edge_index,
                        config.embeddings_combine_strategy)
            loss = F.binary_cross_entropy_with_logits(pred, 
                                                      batch["customer", "recipe"].edge_label.float())
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
        return total_loss / total_examples
    
@torch.no_grad()
def test_link_predictor(model, data, config):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'recipe'].edge_label_index, config.embeddings_combine_strategy)
    target = data['customer', 'recipe'].edge_label.float()
    loss = F.binary_cross_entropy_with_logits(pred, target)
    return float(loss)


def train_embedding_similarity(model, optimizer, config:ModelConfig, data=None, loader=None):
    model.train()
    model.to(device)
    loss_method = criterion_dict[config.criterion.value]
    if loader is None:
        optimizer.zero_grad()
        embedding_dict = model(data.x_dict, 
                    data.edge_index_dict)
        target = data["customer", "recipe"].edge_label.float()
        h_src = embedding_dict["customer"][data["customer", "recipe"].edge_index[0]]
        h_dst = embedding_dict["recipe"][data["customer", "recipe"].edge_index[1]]
        loss = loss_method(h_src, h_dst, target)
        loss.backward()
        optimizer.step()
        return float(loss)
    else:
        total_examples = total_loss = 0
        for batch in tqdm(loader):
            optimizer.zero_grad()
            batch.to(device)
            batch_size = batch['customer'].batch_size
            embedding_dict = model(batch.x_dict, 
                    batch.edge_index_dict)
            target = batch["customer", "recipe"].edge_label.float()
            h_src = embedding_dict["customer"][batch["customer", "recipe"].edge_index[0]]
            h_dst = embedding_dict["recipe"][batch["customer", "recipe"].edge_index[1]]
            loss = loss_method(h_src, h_dst, target)
            loss.backward()
            optimizer.step()
            total_examples += batch_size
            total_loss += float(loss) * batch_size
        return total_loss / total_examples
    
@torch.no_grad()
def test_embedding_similarity(model, data, config):
    model.eval()
    model.to('cpu')
    loss_method = criterion_dict[config.criterion.value]
    embedding_dict = model(data.x_dict, data.edge_index_dict)
    target = data['customer', 'recipe'].edge_label.float()
    h_src = embedding_dict["customer"][data["customer", "recipe"].edge_label_index[0]]
    h_dst = embedding_dict["recipe"][data["customer", "recipe"].edge_label_index[1]]
    loss =loss_method(h_src, h_dst, target)
    return float(loss)
 
@torch.no_grad()
def predict(model, data, config):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict)
    return pred

# COMMAND ----------

def train_test(config, learning_rate=0.01, e_threshold=10, num_epochs=50, min_acc=0.005):
    t0 = time.time()
    model_params = {"hidden_channels": config.hidden_channels, "conv": SAGEConv}
    model = EmbeddingsModel(**model_params).to(device)
    

    with torch.no_grad():
        batch = next(iter(train_loader))
        batch.to(device)
        model.encoder(batch.x_dict, batch.edge_index_dict)
    
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=3,mode="triangular2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    k=0
    losses = []
    for epoch in range(1, num_epochs+1):
        loss = train_embedding_similarity(model, optimizer, config=config, loader=train_loader)
        val_loss = test_embedding_similarity(model, val_data, config)
        test_loss = test_embedding_similarity(model, test_data, config)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Val: {val_loss:.4f}, Test: {test_loss:.4f}')
        losses.append({'train_loss': loss, 
                      'val_loss': val_loss,
                      'test_loss': test_loss})
        if early_stopper.early_stop(val_loss):             
            break
        

    return pd.DataFrame({'loss': loss,
            'val_loss': val_loss,
            'test_loss': test_loss, 
            'time': (time.time()-t0)/60}, index=[0]), model, losses

# COMMAND ----------


class EmbeddingsCombineStrategy(Enum):
    CONCAT = "concat"
    PAIR_PRODUCT = "piecewise_product"
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
    batch_size: int = 4096 # Number of batches
    num_epochs: int = 1
    learning_rate: float = 0.01
    model_name: str = ''
    
    def __post_init__(self):
        self.model_name = f"{self.model_name}_hetero_{self.customer_features}_cf_{self.criterion.value}_{self.embeddings_combine_strategy.value}_{self.dimensions}_bs{self.batch_size}_ep{self.num_epochs}_lr{str(self.learning_rate).split('.')[0]}p{str(self.learning_rate).split('.')[1]}"

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

# COMMAND ----------

#def train_test(model, train_loader, train_data, val_data, test_data, config, e_threshold=10, min_acc=0.005, epoch_save=False):
def train_test(config, learning_rate=0.1, e_threshold=10, num_epochs=50, min_acc=0.005):
    t0 = time.time()
    #model = Model(**model_params)
    epoch_save = False

    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)
    optimizer = torch.optim.AdamW(model.parameters())
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01, amsgrad=False)
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2, threshold=0.001, min_lr = 0.001)
    k=0
    losses = []
    for epoch in range(1, 16):
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

# COMMAND ----------

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# COMMAND ----------

early_stopper = EarlyStopper(patience=3, min_delta=0.1)

# COMMAND ----------

train_data

# COMMAND ----------

# model training
model_params = {"hidden_channels": 64, "conv": SAGEConv}
#model_config = ModelConfig(embeddings_combine_strategy = EmbeddingsCombineStrategy.PIECEWISE_PRODUCT)                                         
model_config = ModelConfig(dimensions=64,
                           hidden_channels=64,
                           criterion=Criterion.BCE_LOSS, 
                           customer_features='random',
                          embeddings_combine_strategy=EmbeddingsCombineStrategy.PAIR_PRODUCT
                          )

#baseline.define_data(pytorch_hetero_graph, train_data)

model = Model(64, SAGEConv)

results, model, losses = train_test(config=model_config)

# COMMAND ----------

loss_train = []
loss_val = []
for i in range(len(losses)):
    loss_train.append(losses[i]['train_loss'])
    loss_val.append(losses[i]['val_loss'])

# COMMAND ----------

fig,ax = plt.subplots(figsize=(10,7))
plt.plot(np.arange(1,len(loss_train)+1),loss_train, label='Training loss')
plt.plot(np.arange(1,len(loss_train)+1),loss_val, label='Validation loss')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.legend(fontsize=20)

# COMMAND ----------

model_file_path = f"/dbfs/tmp/test_baseline.pkl"

# COMMAND ----------

with open(model_file_path, "wb") as f:
    print(model_file_path)
    pickle.dump(model, f)

# COMMAND ----------

# MAGIC %md # Test Model

# COMMAND ----------

import pandas as pd
import torch
'''
def recipes_to_rank_by_week(week, model_name, customer_node_map_df, recipe_node_map_df):
    """
    This is done to get customer & recipe node ids from the graph for which uptake probabilities need to be predicted
    Saves a version of recipes to rank with respective customer and recipe node id
    """
    # read recipes to rank
    recipes_to_rank = pd.read_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/customer_graph_input/customer_recipes_to_predict_2022-W25_2022-W29.parquet/hellofresh_week={week}")
    recipes_to_rank = recipes_to_rank.merge(customer_node_map_df, how='left', on="customer_id")
    recipes_to_rank = recipes_to_rank.merge(recipe_node_map_df, how="left", on="recipe_code")
    recipes_to_rank = recipes_to_rank.dropna()
    recipes_to_rank.to_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/recipe_recommendations/heterographsage/intermediate/recipes_to_rank_{model_name}_{week}.parquet", index=False)
    return
'''
def mean_ave_precision(df):
    print(str(df['customer_id'].nunique())+' unique customers')
    value = ave_precision(df).sum()/len(df.groupby(['customer_id']).count())
    print('Mean Average Precision :'+str(value))
    return value

def ave_precision(df):
    return df['relative_rank']/ (df['predicted_rank'] * df['order_n'])

def accuracy(df,exact=False):
    if exact:
        value = len(np.where(df['predicted_rank']==df['relative_rank'])[0])/len(df)
    else:
        value = len(np.where(df['predicted_rank']<=df['order_n'])[0])/len(df)
    print('Average Accuracy :'+str(value))
    return value

def predict_uptake_probs(edges):
    all_preds = []
    epoch=0
    for batch in batch_edge_gen(edges, num_samples=10000000):
        with torch.no_grad():
            pred = model(test_data.x_dict, test_data.edge_index_dict, batch, combine_mode=EmbeddingsCombineStrategy.PAIR_PRODUCT)
            all_preds.extend(pred.sigmoid().tolist())
            epoch+=1
            print(f"Batch num: {epoch}")
    return all_preds

def run_predict_by_test_week(week, model_name):
    print(f"Compute probs for {week}")
    week_num = week.split('-')[1].lower()
    recipes_to_rank = pd.read_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/recipe_recommendations/heterographsage/intermediate/recipes_to_rank_{model_name}_{week}.parquet")
    edges_to_predict = torch.stack([torch.tensor(recipes_to_rank.customer_node_id.values), torch.tensor(recipes_to_rank.recipe_node_id.values)], dim=0)
    edges_to_predict = edges_to_predict.long()
    pred = predict_uptake_probs(edges=edges_to_predict)
    recipes_to_rank["uptake_probability"] = pred
    recipes_to_rank[["customer_id", "product_menu_key", "uptake_probability"]].to_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/recipe_recommendations/heterographsage/predictions/{model_name}_rank_2022_{week_num}.parquet")
    return

def recipes_to_rank_by_week(week, recipe_node_map_df, customer_node_map_df, model_type="heterographsage", remove_defaults=False):
    """
    This is done to get customer & recipe node ids from the graph for which uptake probabilities need to be predicted
    Saves a version of recipes to rank with respective customer and recipe node id
    """
    recipes_to_rank = pd.read_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/customer_graph_input/customer_recipes_to_predict_2022-W25_2022-W29.parquet/hellofresh_week={week}")
    recipes_to_rank = recipes_to_rank.merge(customer_node_map_df, how='left', on="customer_id")
    recipes_to_rank = recipes_to_rank.merge(recipe_node_map_df, how="left", on="recipe_code")
    recipes_to_rank = recipes_to_rank.dropna()
    if remove_defaults:
        recipe_defaults = pd.read_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/customer_graph_input/customer_actuals_2022-W25_2022-W29_2022-10-11.parquet/hellofresh_week={week}")
        recipe_defaults['default_recipes'] = recipe_defaults['default_recipes'].map(tuple)
        # get default recipes per customer
        recipe_defaults = recipe_defaults.groupby(['customer_id', 'default_recipes'])['product_menu_key'].count().reset_index()
        recipe_defaults = recipe_defaults[["customer_id", "default_recipes"]].explode('default_recipes').reset_index(drop=True)
        # get slot number 
        recipes_to_rank['slot'] = recipes_to_rank['product_menu_key'].apply(lambda x: x.split('_')[2])
        # merge with indicator =True so that we can filter for rows only in recipes to rank (non default)
        recipes_to_rank = recipes_to_rank.merge(recipe_defaults, how='left', left_on=['customer_id','slot'], 
                                                right_on=["customer_id", "default_recipes"] ,indicator=True)
        # this will only retain rows in left dataframe 
        recipes_to_rank = recipes_to_rank.loc[recipes_to_rank._merge=='left_only',recipes_to_rank.columns!='_merge']
        recipes_to_rank.drop(columns=['default_recipes'], inplace=True)
        recipes_to_rank.dropna(inplace=True)
        recipes_to_rank.to_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/recipe_recommendations/{model_type}/intermediate/recipes_to_rank_non_defaults_{week}_100K.parquet", index=False)
    else:
        recipes_to_rank.to_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/recipe_recommendations/{model_type}/intermediate/recipes_to_rank_{week}_100K.parquet", index=False)
    return 

def data_prep_for_mAP(ranked_df, customer_actuals):
    """
    """
    # compute number of recipes ordered per week
    evaluate_df = customer_actuals.merge(ranked_df[["customer_id", "product_menu_key", "predicted_rank"]], how="left", on=["customer_id", "product_menu_key"])
    # no prior orders
    evaluate_df = evaluate_df.dropna()
    evaluate_df["relative_rank"] = evaluate_df.groupby(["customer_id"])["predicted_rank"].rank()
    return evaluate_df

def evaluate_uptake_probs(week, model_name):
    week_num = week.split('-')[1].lower()
    uptake_probs = pd.read_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/recipe_recommendations/heterographsage/predictions/{model_name}_rank_2022_{week_num}.parquet")
    uptake_probs["predicted_rank"] = uptake_probs.groupby(["customer_id"])["uptake_probability"].rank(ascending=False)
    customer_actuals = pd.read_parquet(f"s3://hf-global-ai-dev/mls_embeddings_platform/customer_graph_input/customer_actuals_2022-W25_2022-W29.parquet/hellofresh_week={week}")
    customer_actuals["order_n"] = customer_actuals.groupby(["customer_id"])["product_menu_key"].transform('count')
    evaluate_df = data_prep_for_mAP(uptake_probs, customer_actuals)
    print(f"MAP for {week}:  {mean_ave_precision(evaluate_df)}")
    print(f"Accuracy for {week}: {accuracy(evaluate_df)}")
    
    return mean_ave_precision(evaluate_df)

def batch_edge_gen(edges, num_samples=100000):
    for i in range(0, edges.shape[1], num_samples): 
        test_subset = torch.stack([edges[0, i:i+num_samples], edges[1,  i:i+num_samples]], dim=0)
        yield test_subset
        
def define_model_test_data(temp_model, data):
    global model
    global test_data
    
    test_data = data
    model = temp_model
    
    return

# COMMAND ----------

# For mapping node ids in networkx graph to pytorch graph
#new_graph = nx.read_gpickle("/dbfs/tmp/nx_graph_for_pytorch_hetero.gpickle")
customer_node_map = [int(''.join(node_id[1:])) for node_id, node_data in nx_hetero_sp.nodes(data=True) if node_data.get('type') == 'customer']
recipe_node_map = [node_id for node_id, node_data in nx_hetero_sp.nodes(data=True) if node_data.get('type') == 'recipe']
