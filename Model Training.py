# Databricks notebook source
# MAGIC %sh
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

from tasteprofile import baseline
from tasteprofile import utils

# COMMAND ----------

importlib.reload(baseline)
importlib.reload(utils)

# COMMAND ----------

# For mapping node ids in networkx graph to pytorch graph
new_graph = nx.read_gpickle("/dbfs/tmp/nx_graph_for_pytorch_hetero.gpickle")
customer_node_map = [int(''.join(node_id[1:])) for node_id, node_data in new_graph.nodes(data=True) if node_data.get('type') == 'customer']
recipe_node_map = [node_id for node_id, node_data in new_graph.nodes(data=True) if node_data.get('type') == 'recipe'] 

# COMMAND ----------

with open("/dbfs/tmp/pytorch_hetero_09_22_50M.pkl", "rb") as f:
    pytorch_hetero_graph = pickle.load(f)

# COMMAND ----------

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
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
    num_samples={key: [2000] * 4 for key in train_data.node_types},
    batch_size=5000000,
    input_nodes=('customer'),
    #transform=NormalizeFeatures(),
)

# COMMAND ----------

# model training
model_params = {"hidden_channels": 64, "conv": SAGEConv}
model_config = baseline.ModelConfig(model_name='_test')
_=baseline.define_data(pytorch_hetero_graph, train_data)
model = baseline.Model(64, SAGEConv)
results, model, losses = utils.train_test(model, train_loader, train_data, val_data, test_data, config=model_config, learning_rate=0.05, num_epochs=1)

# COMMAND ----------

model_file_path = f"/dbfs/tmp/test_baseline.pkl"

# COMMAND ----------

with open(model_file_path, "wb") as f:
    print(model_file_path)
    pickle.dump(model, f)
