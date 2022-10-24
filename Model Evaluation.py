# Databricks notebook source
# MAGIC %md # Preamble

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install networkx==2.8.6
# MAGIC pip install -U pip
# MAGIC pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# MAGIC pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# MAGIC pip install torch-geometric
# MAGIC pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# MAGIC pip install s3fs

# COMMAND ----------

# MAGIC %md ## Imports

# COMMAND ----------

import pickle
import networkx as nx
import pandas as pd
import numpy as np

import sys
import importlib

import torch

from pyspark.sql.functions import concat, col, lit, array, udf, row_number, desc, asc, substring_index, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.types import DoubleType, FloatType
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import torch_geometric.transforms as T

from sklearn.metrics.pairwise import cosine_similarity

import os, sys
import logging

import pyspark.sql.functions as F
from pyspark.sql.functions import concat, col, lit, array, udf, row_number, desc, asc

from scipy.spatial.distance import cosine 
from numpy import dot
from numpy.linalg import norm
import random

# COMMAND ----------

sys.path.append("/Workspace/Repos/blake.forland@hellofresh.com/graphSAGE-models")

# COMMAND ----------

from tasteprofile_models import baseline
#from tasteprofile import linkprediction

importlib.reload(baseline)
#importlib.reload(linkprediction)

# COMMAND ----------

# MAGIC %md ## Functions

# COMMAND ----------

# MAGIC %md ### Utility

# COMMAND ----------

def sk_cos_sim(a,b):
    return cosine_similarity(a,b)

# COMMAND ----------

def mean_ave_precision(df):
    print(str(df['customer_id'].nunique())+' unique customers')
    # value = ave_precision(df).sum()/df['customer_id'].nunique()
    value = ave_precision(df).sum()/len(df.groupby(['customer_id', 'hellofresh_week']).count())
    print('Mean Average Precision :'+str(value))
    return value

def ave_precision(df):
    return df['actual_rank']/ (df['predicted_rank'] * df['order_n'])

def accuracy(df,exact=False):
    if exact:
        value = len(np.where(df['predicted_rank']==df['actual_rank'])[0])/len(df)
    else:
        value = len(np.where(df['predicted_rank']<=df['order_n'])[0])/len(df)
    print('Average Accuracy :'+str(value))
    return value

# COMMAND ----------

# MAGIC %md ### R-R Eval

# COMMAND ----------

def recipe_recipe_prediction(
    recipe_embeddings: pd.DataFrame(),
    n_custs = 5000,
    test_weeks = ['2022-W25', '2022-W26', '2022-W27', '2022-W28', '2022-W29'],
    skip_weeks = ['2022-W30', '2022-W31', '2022-W32', '2022-W33', '2022-W34', '2022-W35', '2022-W36', '2022-W37', '2022-W38'],
    weeks = (202223, 202227),
    use_mean=False,
    use_choice=False,
    lyre=False,
     verbose=1,
):
    #recipe_cosin_pairs = gen_cossim_table(recipe_embeddings, embed_str='PC').toPandas()
    
    recipe_embeddings = recipe_embeddings.set_index('product_menu_key')
    recipe_embeddings_np = recipe_embeddings.to_numpy()
    cos_sim_score_np = sk_cos_sim(recipe_embeddings_np,recipe_embeddings_np)
    cos_sim_score_pd  = pd.DataFrame(cos_sim_score_np)
    cos_sim_score_pd.index = recipe_embeddings.index
    cos_sim_score_pd.columns = recipe_embeddings.index.tolist()
    names = cos_sim_score_pd.columns
    cos_sim_score_pd  = cos_sim_score_pd.reset_index()
    
    recipe_cosin_pairs = pd.melt(cos_sim_score_pd, id_vars='product_menu_key', value_vars=names).rename(columns={'variable': 'product_menu_key_2', 'value': 'coSim'})
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    
    print('Loading Cos Sim Scores')
    #recipe_cosin_pairs = pd.read_parquet(recipe_embeddings_scores)
    recipe_cosin_pairs = recipe_cosin_pairs[~recipe_cosin_pairs.product_menu_key.str.contains('2019')]
    recipe_cosin_pairs = recipe_cosin_pairs[~recipe_cosin_pairs.product_menu_key.str.contains('|'.join(skip_weeks))]
    
    #customer_orders_SP = spark.read.parquet(f's3://hf-global-ai-dev/mls_embeddings_platform/recipe_graph_input/full_customer_order_history.parquet')
    print('Loading Customer Order History')
    logging.info('User Choice is:'+str(use_choice))
    # loading recipe defaults
    recipe_defaults = pd.read_parquet("s3://hf-global-ai-dev/mls_embeddings_platform/customer_graph_input/customer_actuals_2022-W25_2022-W29_2022-10-11.parquet")
    
    if use_choice:
        customer_orders = pd.read_parquet(f's3://hf-global-ai-dev/mls_embeddings_platform/recipe_graph_input/full_customer_order_history_user_choice.parquet')
    else:
        customer_orders = pd.read_parquet(f's3://hf-global-ai-dev/mls_embeddings_platform/recipe_graph_input/full_customer_order_history.parquet')
    customer_orders = customer_orders[~customer_orders.product_menu_key.str.contains('2019')]
    # Michal faster way
    #customer_orders = pd.read_parquet(f's3://hf-global-ai-dev/mls_embeddings_platform/recipe_graph_input/full_customer_order_history2.parquet')

    
    customer_orders_SP = customer_orders.rename(columns={'product_menu_key': 'ordered_recipe'})
    
    print('Filtering to test weeks')
    test_customer = customer_orders_SP[customer_orders_SP.ordered_recipe.str.contains('|'.join(test_weeks))]['customer_id'].drop_duplicates()
    
    # filter for customers in recipe defaults data set
    test_customer = test_customer[test_customer.isin(recipe_defaults.customer_id.unique())]
    # Michal faster way
    #test_customer = set(customer_orders_SP[customer_orders_SP.week.between(weeks[0], weeks[1])].customer_id.drop_duplicates())
    
    print('Filtering to '+str(n_custs)+' customers')
    test_customer = set(test_customer[:n_custs])
    
    customer_orders_SP = customer_orders_SP[customer_orders_SP.customer_id.isin(test_customer)]
    
    print('Unique customers :'+str(customer_orders_SP['customer_id'].nunique()))
    
    logging.info('Filtering to train weeks')
    customer_orders_train = customer_orders_SP[~customer_orders_SP.ordered_recipe.str.contains('|'.join(test_weeks))]
    # Michal faster way
    #customer_orders_train = customer_orders_SP[~customer_orders_SP.week.between(weeks[0], weeks[1])]
    
    print('Loading Menus')
    menu_SP = pd.read_parquet(f's3://hf-global-ai-dev/mls_embeddings_platform/recipe_graph_input/full_menu_history.parquet')
    
    print('Filtering Menus')
    # could be faster using "isin"
    menu_test = menu_SP[menu_SP.product_menu_key.str.contains('|'.join(test_weeks))]
    
    # Crossjoin Menu and Order history
    # could do a inner poin
    customer_orders_train['key'] = 0
    menu_test['key'] =   0
    customer_menu = customer_orders_train.merge(menu_test, on='key', how='outer').drop(columns=['key'])
    
    print('Creating customer cosim table')
    test = customer_menu.merge(recipe_cosin_pairs.rename(columns={'product_menu_key_2': 'ordered_recipe'}), on=['product_menu_key', 'ordered_recipe'], how='inner')
    
    #test = test.drop(columns=['ordered_recipe'])
    
    # Get highest scores recipe relationship per menu item
    if use_mean:
        logging.info('Using Mean for Rank')
        test_2 = pd.DataFrame(test.groupby(['customer_id', 'product_menu_key'])['coSim'].mean()).reset_index()
    else:
        logging.info('Using Max for Rank')
        test_2 = pd.DataFrame(test.groupby(['customer_id', 'product_menu_key'])['coSim'].max()).reset_index()
    
    print('Predicting Rank')
    if lyre:
        recipe_defaults["hellofresh_week"] = recipe_defaults["product_menu_key"].apply(lambda x: x.split('_')[1])
        defaults_df = recipe_defaults[["customer_id", "hellofresh_week", "default_recipes"]].explode('default_recipes').drop_duplicates()
        defaults_df['product_menu_key'] = defaults_df["hellofresh_week"] + "_" + defaults_df["default_recipes"]
        defaults_df['drop_default'] = 1

        defaults_df.drop('default_recipes', axis=1, inplace=True)
        test_2 = test_2.merge(defaults_df, on=["customer_id", "product_menu_key"], how='left')
        test_2 = test_2.loc[test_2.drop_default.isna(), ['customer_id', 'product_menu_key', 'coSim']]
        
    test_2['predicted_rank'] = test_2.groupby(['customer_id', test_2.product_menu_key.str.split('_',expand=True)[1]])['coSim'].rank(ascending=False)
    
    #customer_orders_SP = customer_orders_SP[customer_orders_SP.customer_id.isin(test_customer)]
    
    customer_orders_test = customer_orders_SP[customer_orders_SP.ordered_recipe.str.contains('|'.join(test_weeks))]
    # Michal faster way
    # customer_orders_test = customer_orders_SP[customer_orders_SP.week.between(weeks[0], weeks[1])]
    
    
    customer_orders_test = customer_orders_test.rename(columns={'ordered_recipe': 'product_menu_key'})
    
    if lyre:
        customer_orders_test = customer_orders_test.merge(recipe_defaults,  on=["customer_id", "product_menu_key"], how='inner')
        customer_orders_test = customer_orders_test[customer_orders_test.include_in_lyre_evaluation==1]
        
    test_2 = test_2.merge(customer_orders_test,on=['customer_id','product_menu_key'],how='inner')
    
    order_n = pd.DataFrame(test_2.groupby(['customer_id',test_2.product_menu_key.str.split('_',expand=True)[1]])['product_menu_key'].count()).reset_index()
    order_n = order_n.rename(columns={'product_menu_key': 'order_n', 1: 'hellofresh_week'})
    
    # use this as the lookup instead of contains blah
    test_2['hellofresh_week'] = test_2.product_menu_key.str.split('_',expand=True)[1]
    
    test_3 = test_2.merge(order_n,on=['customer_id', 'hellofresh_week'],how='inner')
    
    # Prediction ranking order
    print('Getting Prediction order')
    test_3['actual_rank'] = test_3.groupby(['customer_id','hellofresh_week'])['predicted_rank'].rank()
    
    return test_3

# COMMAND ----------

# MAGIC %md ### Link Prediction

# COMMAND ----------

import pandas as pd
import torch

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
            pred = model(test_data.x_dict, test_data.edge_index_dict, batch)
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

# MAGIC %md # Evaluation

# COMMAND ----------

# MAGIC %md ## Test Model

# COMMAND ----------

with open(f"/dbfs/tmp/base_line_hetero_default_cf_bce_logits_piecewise_product_64_bs4096_ep20_lr0p01_MUL.pkl", "rb") as f:
    model = pickle.load(f)

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

#with open(f"/dbfs/tmp/train_val_test.pkl", "rb") as f:
#    data = pickle.load(f)
#train_data = data['train_data']
#val_data = data['val_data']
#test_data = data['test_data']

# COMMAND ----------

# For mapping node ids in networkx graph to pytorch graph
new_graph = nx.read_gpickle("/dbfs/tmp/nx_graph_for_pytorch_hetero.gpickle")
customer_node_map = [int(''.join(node_id[1:])) for node_id, node_data in new_graph.nodes(data=True) if node_data.get('type') == 'customer']
recipe_node_map = [node_id for node_id, node_data in new_graph.nodes(data=True) if node_data.get('type') == 'recipe']  

# COMMAND ----------

customer_node_map_df = pd.DataFrame(customer_node_map, columns=["customer_id"])
customer_node_map_df = customer_node_map_df.reset_index().rename(columns={'index': 'customer_node_id'})
recipe_node_map_df = pd.DataFrame(recipe_node_map, columns=["recipe_code"])
recipe_node_map_df = recipe_node_map_df.reset_index().rename(columns={"index": "recipe_node_id"})

# COMMAND ----------

test_weeks=['2022-W25', '2022-W26', '2022-W27', '2022-W28', '2022-W29']
for week in test_weeks:
    print(week)
    recipes_to_rank_by_week(week, 'test_baseline', customer_node_map_df, recipe_node_map_df)

# COMMAND ----------

test_weeks=['2022-W25', '2022-W26', '2022-W27', '2022-W28', '2022-W29']
for week in test_weeks:
    run_predict_by_test_week(week, 'test_baseline')
    evaluate_uptake_probs(week, 'test_baseline')

# COMMAND ----------


