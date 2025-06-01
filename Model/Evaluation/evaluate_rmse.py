import os, sys; [(sys.path.append(d),)
                 for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) 
                           for i in range(len(os.getcwd().split(os.sep)))) 
                           if os.path.isfile(os.path.join(d, 'main.py'))]

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit

def top_and_bottom_k_intersection(edge_list, target, pred, k):
    """
    Count how many edges appear in both the predicted and true top-k AND bottom-k.

    Parameters:
        edge_list: list of (user_idx, business_idx) tuples
        target: 1D numpy array of true ratings
        pred: 1D numpy array of predicted ratings
        k: int, number of top and bottom elements to consider

    Returns:
        A tuple (top_k_intersection, bottom_k_intersection)
    """
    # Convert to tensors for sorting
    pred_tensor = torch.from_numpy(pred)
    target_tensor = torch.from_numpy(target)

    # Sort indices (descending for top-k)
    pred_top_indices = torch.argsort(-pred_tensor).numpy()[:k]
    true_top_indices = torch.argsort(-target_tensor).numpy()[:k]

    # Sort indices (ascending for bottom-k)
    pred_bottom_indices = torch.argsort(pred_tensor).numpy()[:k]
    true_bottom_indices = torch.argsort(target_tensor).numpy()[:k]

    # Get edge sets
    pred_top = set(tuple(edge_list[i]) for i in pred_top_indices)
    true_top = set(tuple(edge_list[i]) for i in true_top_indices)
    pred_bottom = set(tuple(edge_list[i]) for i in pred_bottom_indices)
    true_bottom = set(tuple(edge_list[i]) for i in true_bottom_indices)

    # Compute intersections
    top_k_intersection = len(pred_top & true_top)
    bottom_k_intersection = len(pred_bottom & true_bottom)

    return top_k_intersection, bottom_k_intersection

def correct_order_count(target, pred):
    """
    Count the number of correctly ordered pairs using merge sort inversion count.
    Returns (correct_count, total_pairs).
    """
    import numpy as _np

    n = len(pred)
    total_pairs = n * (n - 1) // 2
    idx = _np.argsort(-pred)
    t_sorted = target[idx]
    def merge_count(arr):
        if len(arr) <= 1:
            return 0, arr
        mid = len(arr) // 2
        left_count, left = merge_count(arr[:mid])
        right_count, right = merge_count(arr[mid:])
        merged = []
        count = left_count + right_count
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] >= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                count += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return count, merged
    discordant, _ = merge_count(list(t_sorted))
    correct = total_pairs - discordant
    return correct, total_pairs


def save_dict_to_csv_row(dictionary, filename):
    """
    Helper function to save a dictionary to a CSV file. 
        If the file exists, it appends the dictionary as a new row.
        If the file doesn't exist, it creates it with headers from the dictionary keys.
    Args:
        dictionary (dict): The dictionary to write. Keys are column headers.
        filename (str): Path to the CSV file.
    """
    import csv
    import os

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(dictionary)

def evaluate_hgn(k=20, version='baseline'):
    data_path=f'Data/Saved_Model/hetero_data-{version}.pt'
    model_path=f'Model/Saved_Model/hgn_model-{version}.pt'
    from Model.Prediction.hgn_recommendation import DEVICE, _model
    # Load full hetero data
    data = torch.load(data_path, weights_only=False)
    _model = torch.load(model_path, weights_only=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split edges: 10% validation, 10% test, no negative sampling
    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'business')],
        rev_edge_types=[('business', 'rev_rates', 'user')]
    )(data)
    test_data = test_data.to(DEVICE)

    # Extract true labels and edge index for user -> business ratings
    edge_label_index = test_data['user', 'rates', 'business'].edge_label_index
    target = test_data['user', 'rates', 'business'].edge_label.float().to(DEVICE)

    # Predict
    with torch.no_grad():
        pred = _model(test_data.x_dict, test_data.edge_index_dict, edge_label_index)
        pred = pred.clamp(0, 5)

    # Compute RMSE
    rmse = F.mse_loss(pred, target).sqrt()
    print(f'HGN Test RMSE: {rmse.item():.4f}')

    # Prepare data for ranking metrics
    edge_index_cpu = edge_label_index.cpu().numpy().T
    edge_list = [tuple(e) for e in edge_index_cpu]
    pred_arr = pred.cpu().numpy()
    target_arr = target.cpu().numpy()

    # Top-k intersection
    top_k_intersection, bottom_k_intersection = top_and_bottom_k_intersection(edge_list, target_arr, pred_arr, k)
    print(f'HGN Top-{k} intersection size: {top_k_intersection}')
    print(f'HGN Bottom-{k} intersection size: {bottom_k_intersection}')

    # Pairwise ordering accuracy
    correct_pairs, total_pairs = correct_order_count(target_arr, pred_arr)
    print(f'HGN Correctly ordered pairs: {correct_pairs}/{total_pairs} '
          f'({correct_pairs/total_pairs*100:.2f}%)')
    
    results_dict={
        'Model':'TGN',
        'Version':version,
        'RMSE':rmse.item(),
        'k':k,
        'Top-k':top_k_intersection,
        'Bottom-k':bottom_k_intersection,
        'Correct Pairs':correct_pairs,
        'Total Pairs':total_pairs,
        'Order correctness %':correct_pairs/total_pairs,
    }
    save_dict_to_csv_row(dictionary=results_dict, filename='Data/eval_results.csv')


def evaluate_tgn(k=20, version='baseline'):
    model_path=f'Model/tgn_model-{version}.pt'
    import os
    from Model.Data_Presprocessing.Prediction.tgn_recommendation import load_and_preprocess, temporal_split, MyTGN, TemporalEdgeDataset
    from torch.utils.data import DataLoader

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess temporal data
    (source, dest, edge_attr, norm_ts,
     user_df, biz_df,
     user_to_index, business_to_index,
     user_feats, biz_feats,
     avg_rating) = load_and_preprocess()

    # Construct dataset and split
    dataset = TemporalEdgeDataset(source, dest, edge_attr, norm_ts)
    train_ds, val_ds, test_ds = temporal_split(dataset)

    # Instantiate model and load state
    total_nodes = len(user_to_index) + len(business_to_index)
    model = MyTGN(total_nodes,
                  user_feat_dim=user_feats.size(1),
                  business_feat_dim=biz_feats.size(1)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    # Load only matching weights to handle dimensionality changes
    model_dict = model.state_dict()
    filtered_state = {k: v for k, v in state.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_state)
    model.load_state_dict(model_dict)
    model.eval()

    # Evaluate on test split
    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    for src, dst, eattr, ts in loader:
        src, dst, eattr, ts = src.to(DEVICE), dst.to(DEVICE), eattr.to(DEVICE), ts.to(DEVICE)
        # Concatenate edge attributes with user and business features to match training
        u_feats = user_feats[src]
        b_feats = biz_feats[dst - user_feats.size(0)]
        combined = torch.cat([eattr, u_feats, b_feats], dim=1)
        with torch.no_grad():
            pred = model(src, dst, ts, combined).squeeze()
            pred = pred.clamp(0, 5)
        target = eattr[:, 0]  # rating is first column

    # Compute RMSE
    rmse = F.mse_loss(pred, target).sqrt()
    print(f'TGN Test RMSE: {rmse.item():.4f}')

    # Prepare data for ranking metrics
    edge_list = list(zip(src.cpu().tolist(), dst.cpu().tolist()))
    pred_arr = pred.cpu().numpy()
    target_arr = target.cpu().numpy()

    # Top-k intersection
    top_k_intersection, bottom_k_intersection = top_and_bottom_k_intersection(edge_list, target_arr, pred_arr, k)
    print(f'HGN Top-{k} intersection size: {top_k_intersection}')
    print(f'HGN Bottom-{k} intersection size: {bottom_k_intersection}')

    # Pairwise ordering accuracy
    correct_pairs, total_pairs = correct_order_count(target_arr, pred_arr)
    print(f'TGN Correctly ordered pairs: {correct_pairs}/{total_pairs} '
          f'({correct_pairs/total_pairs*100:.2f}%)')

    results_dict={
        'Model':'TGN',
        'Version':version,
        'RMSE':rmse.item(),
        'k':k,
        'Top-k':top_k_intersection,
        'Bottom-k':bottom_k_intersection,
        'Correct Pairs':correct_pairs,
        'Total Pairs':total_pairs,
        'Order correctness %':correct_pairs/total_pairs,
    }
    save_dict_to_csv_row(dictionary=results_dict, filename='Data/eval_results.csv')

def main(version):
    k=10000
    evaluate_hgn(k=k,version=version)
    evaluate_tgn(k=k,version=version)

if __name__ == "__main__":
    main(version='baseline')
    main(version='business_categories')
    main(version='business_names')
    main(version='business_address')
    main(version='all')
