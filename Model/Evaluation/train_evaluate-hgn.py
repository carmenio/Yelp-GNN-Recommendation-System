import pandas as pd
import torch
import torch_geometric.transforms as T
import pickle
from torch_geometric.data import HeteroData
import numpy as np
import random
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch_geometric.transforms import RandomLinkSplit
import sys,os

# Ensure project root is on PATH
sys.path.append(os.path.abspath(os.getcwd()))
from Model.Evaluation.evaluate_rmse import top_and_bottom_k_intersection, correct_order_count, save_dict_to_csv_row, evaluate_hgn

#=============================================================
# Heterogeneous Data Model
#=============================================================
def create_hetero_data(version='baseline',embeddings_features=[]):
    review_df = pd.read_parquet('Data/preprocessed_review.parquet', engine='pyarrow')
    user_df = pd.read_parquet('Data/preprocessed_user.parquet', engine='pyarrow')
    business_df = pd.read_parquet('Data/preprocessed_business.parquet', engine='pyarrow')

    # loading business_category_index
    with open('Data/business_category_index.pickle', 'rb') as handle:
        business_category_index = pickle.load(handle)

    ### Nodes ###
    # User features
    user_features=torch.from_numpy(user_df[['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']].values).to(torch.float)  # [num_users, num_features_users]
    print(user_features.shape)

    # Business features
    business_features=torch.from_numpy(business_df[['stars', 'review_count', 'is_open', 'latitude', 'longitude']].values).to(torch.float)  # [num_businesses, num_features_businesses]

    business_features=[business_features]
    for feature in embeddings_features:
        business_features.append(torch.from_numpy(np.stack(business_df[feature].values)).float())

    business_features=torch.cat(business_features, dim=1)

    print(business_features.shape)

    # Business Categories
    # category_features = torch.eye(len(business_category_index))

    ### Edges ###
    # Review edge index
    review_edge_index=torch.stack([
        torch.tensor(review_df['user_id_index'].values),
        torch.tensor(review_df['business_id_index'].values)]
        , dim=0)
    assert review_edge_index.shape == (2, len(review_df))
    print('review_edge_index.shape:',review_edge_index.shape)

    # Review edge label
    rating=torch.from_numpy(review_df['stars'].values).to(torch.float)
    print('rating.shape:',rating.shape)

    # Friend edge index
    user_friend_index = torch.tensor([
        (user, friend)                                      # 3) Tuple of users and friends
        for user, friend_list in user_df['friends'].items() # 1) Parent-loop, across each user in user_df
        for friend in friend_list                           # 2) Child-loop, across each friend of each user
    ], dtype=torch.long).t().contiguous()                   # 4) Convert to tensor and transpose to required shape
    print('user_friend_index.shape:',user_friend_index.shape)

    # Business category edge index
    business_category_index = torch.tensor([
        (business, category)                                                 # 3) Tuple of businesses and their categories
        for business, category_list in business_df['category_index'].items() # 1) Parent-loop, across each business in business_df
        for category in category_list                                        # 2) Child-loop, across each category in category_list
    ], dtype=torch.long).t().contiguous()                                    # 4) Convert to tensor and transpose to required shape
    print('business_category_index.shape:',business_category_index.shape)

    data = HeteroData()
    data['user'].x = user_features
    data['business'].x = business_features
    data['user','rates','business'].edge_index=review_edge_index
    data['user','rates','business'].edge_label=rating
    # data['user','friends','user'].edge_index=user_friend_index
    # data['business','part_of','category'].edge_index=business_category_index

    # Add the reverse edges in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)

    # With the above transformation we also got reversed labels for the edges.
    # We are going to remove them:
    del data['business', 'rev_rates', 'user'].edge_label

    # Save the HeteroData object
    torch.save(data, f'Data/hetero_data-{version}.pt')

    print(data)

    return data

#=============================================================
# GraphSAGE model definition
#=============================================================
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['business'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)



def evaluate_hgn(test_data, model, DEVICE, k=10000, version='baseline'):

    test_data = test_data.to(DEVICE)

    # Extract true labels and edge index for user -> business ratings
    edge_label_index = test_data['user', 'rates', 'business'].edge_label_index
    target = test_data['user', 'rates', 'business'].edge_label.float().to(DEVICE)

    # Predict
    with torch.no_grad():
        pred = model(test_data.x_dict, test_data.edge_index_dict, edge_label_index)
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
        'Model':'HGN',
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



#=============================================================
# Main Function
#=============================================================
def main(version,embeddings_features):

    # Create Heterogeneous Graph Dataset
    data=create_hetero_data(version=version,embeddings_features=embeddings_features)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Define Train-test splits
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'business')],
        rev_edge_types=[('business', 'rev_rates', 'user')],
    )(data)

    # Determine Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model object with 32 hidden channels
    model = Model(hidden_channels=32, metadata=data.metadata()).to(DEVICE)

    print(model)

    # Create optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter()

    #=============================================================
    # Train Loop
    #=============================================================
    def train():
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, 
                    train_data.edge_index_dict,
                    train_data['user', 'rates', 'business'].edge_label_index
                    )
        target = train_data['user', 'rates', 'business'].edge_label
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss)

    #=============================================================
    # Test Loop
    #=============================================================
    @torch.no_grad()
    def test(data):
        data = data.to(DEVICE)
        model.eval()
        pred = model(data.x_dict,
                    data.edge_index_dict,
                    data['user','rates', 'business'].edge_label_index
                    )
        pred = pred.clamp(min=0, max=5)
        target = data['user', 'rates', 'business'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        return float(rmse)


    #=============================================================
    # Epoch Loop
    #=============================================================
    best_val = float('inf')
    for epoch in range(1, 3001):
    # for epoch in range(1, 10):
        train_data = train_data.to(DEVICE)
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)

        writer.add_scalar("Loss", loss, epoch)
        writer.add_scalar("Train rmse", train_rmse, epoch)
        writer.add_scalar("Val rsme", val_rmse, epoch)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
            f'Val: {val_rmse:.4f}')

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 200:
                print("Early stopping.")
                break

    writer.close()
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"hgn_model-{version}.pt")

    evaluate_hgn(test_data, model, DEVICE, k=10000, version=version)

if __name__ == "__main__":
    main(version='baseline',            embeddings_features=[])
    main(version='business_address',    embeddings_features=['address_embeddings'])
    main(version='business_categories', embeddings_features=['category_embeddings'])
    main(version='business_names',      embeddings_features=['name_embeddings'])
    main(version='all',                 embeddings_features=['address_embeddings','category_embeddings','name_embeddings'])