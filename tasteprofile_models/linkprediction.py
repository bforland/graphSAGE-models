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