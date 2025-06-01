import argparse
import math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch_geometric
import numpy as np

from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator
from torch_geometric.nn.conv import TransformerConv

# --------------- Utility: Haversine distance ---------------

def haversine(user_latitude, user_longitude, business_latitude, business_longitude):
    """
    Compute the great-circle distance (in kilometers) between a user's location
    and a business location using the Haversine formula.
    """
    earth_radius_km = 6371.0
    lat1_rad = math.radians(user_latitude)
    lat2_rad = math.radians(business_latitude)
    delta_lat = math.radians(business_latitude - user_latitude)
    delta_lng = math.radians(business_longitude - user_longitude)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    )
    distance = 2 * earth_radius_km * math.asin(math.sqrt(a))
    return distance

# --------------- Dataset ---------------

class TemporalEdgeDataset(Dataset):
    """
    Organize temporal edges (source, destination, attributes, timestamps)
    in chronological order for training and evaluation.
    """
    def __init__(self, source_nodes, destination_nodes, edge_attributes, timestamps):
        assert (
            source_nodes.size(0)
            == destination_nodes.size(0)
            == edge_attributes.size(0)
            == timestamps.size(0)
        )
        sorted_indices = torch.argsort(timestamps)
        self.source_nodes = source_nodes[sorted_indices]
        self.destination_nodes = destination_nodes[sorted_indices]
        self.edge_attributes = edge_attributes[sorted_indices]
        self.timestamps = timestamps[sorted_indices]

    def __len__(self):
        return self.source_nodes.size(0)

    def __getitem__(self, index):
        return (
            self.source_nodes[index],
            self.destination_nodes[index],
            self.edge_attributes[index],
            self.timestamps[index],
        )

def temporal_split(dataset, validation_ratio=0.1, test_ratio=0.1):
    total = len(dataset)
    test_start = int(total * (1 - test_ratio))
    val_start = int(total * (1 - test_ratio - validation_ratio))
    return (
        Subset(dataset, range(0, val_start)),
        Subset(dataset, range(val_start, test_start)),
        Subset(dataset, range(test_start, total)),
    )

# --------------- Temporal Graph Network (TGN) Model ---------------

class MyTGN(torch.nn.Module):
    """
    Temporal Graph Network (TGN) with memory, transformer conv, and regression output.
    """
    def __init__(
        self,
        num_nodes,
        user_feat_dim,
        business_feat_dim,
        memory_dimension=100,
        time_dimension=10,
        embedding_dimension=100,
        output_dimension=1,
    ):
        super().__init__()
        # raw edge features count: rating, normalized time, useful, funny, cool, user_record_age, review_record_age = 7
        raw_msg_dim = 7 + user_feat_dim + business_feat_dim
        self.tgn_memory = TGNMemory(
            num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dimension,
            time_dim=time_dimension,
            message_module=IdentityMessage(raw_msg_dim, memory_dimension, time_dimension),
            aggregator_module=LastAggregator(),
        )
        self.graph_transformer_conv = TransformerConv(
            in_channels=memory_dimension,
            out_channels=embedding_dimension,
            heads=2,
            dropout=0.1,
        )
        conv_out_dim = embedding_dimension * self.graph_transformer_conv.heads
        self.output_layer = Linear(conv_out_dim * 2 + raw_msg_dim, output_dimension)

    def forward(self, src, dst, ts, edge_attr):
        mem_src, _ = self.tgn_memory(src)
        mem_dst, _ = self.tgn_memory(dst)
        batch_size = mem_src.size(0)
        idx = torch.arange(batch_size, device=mem_src.device)
        edge_index = torch.stack([idx, idx], dim=0)
        emb_src = self.graph_transformer_conv(mem_src, edge_index)
        emb_dst = self.graph_transformer_conv(mem_dst, edge_index)
        combined = torch.cat([emb_src, emb_dst, edge_attr], dim=1)
        return self.output_layer(combined)

    def reset_memory(self):
        self.tgn_memory.reset_state()

    def update_memory(self, src, dst, ts, edge_attr):
        with torch.no_grad():
            self.tgn_memory.update_state(src, dst, ts.long(), edge_attr)
        self.tgn_memory.detach()

# --------------- Data Loading and Preprocessing ---------------

def load_and_preprocess(data_directory="data"):
    import os
    cache_path = os.path.join(data_directory, "tgn_edges_cache.pt")
    # Load dataframes and mappings
    user_df = pd.read_parquet(f"{data_directory}/preprocessed_user.parquet")
    record_age_map = dict(zip(user_df['user_id'], user_df['record_age']))
    business_df = pd.read_parquet(f"{data_directory}/preprocessed_business.parquet")

    unique_users = sorted(user_df["user_id"].unique())
    unique_businesses = sorted(business_df["business_id"].unique())
    user_to_index = {u: i for i, u in enumerate(unique_users)}
    business_to_index = {b: i + len(unique_users) for i, b in enumerate(unique_businesses)}

    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        source, dest, edge_attr, norm_ts = data[:4]
    else:
        review_df = pd.read_parquet(f"{data_directory}/preprocessed_review.parquet")
        source_list, dest_list, ts_list = [], [], []
        rating_list, useful_list, funny_list, cool_list, age_list, review_age_list = [], [], [], [], [], []
        for _, r in tqdm(review_df.iterrows(), total=len(review_df), desc="Loading edges", leave=False):
            if r.user_id in user_to_index and r.business_id in business_to_index:
                source_list.append(user_to_index[r.user_id])
                dest_list.append(business_to_index[r.business_id])
                rating_list.append(r.stars)
                ts_val = pd.to_datetime(r.date).timestamp()
                ts_list.append(ts_val)
                useful_list.append(r.useful)
                funny_list.append(r.funny)
                cool_list.append(r.cool)
                age_list.append(record_age_map[r.user_id])
                review_age_list.append(r.record_age)

        source = torch.tensor(source_list, dtype=torch.long)
        dest = torch.tensor(dest_list, dtype=torch.long)
        raw_ts = torch.tensor(ts_list, dtype=torch.float)
        min_t, max_t = raw_ts.min(), raw_ts.max()
        denom = (max_t - min_t).item()
        if denom == 0:
            norm_ts = torch.zeros_like(raw_ts)
        else:
            norm_ts = (raw_ts - min_t) / denom

        rating_t = torch.tensor(rating_list, dtype=torch.float).unsqueeze(1)
        time_t = norm_ts.unsqueeze(1)
        useful_t = torch.tensor(useful_list, dtype=torch.float).unsqueeze(1)
        funny_t = torch.tensor(funny_list, dtype=torch.float).unsqueeze(1)
        cool_t = torch.tensor(cool_list, dtype=torch.float).unsqueeze(1)
        age_t = torch.tensor(age_list, dtype=torch.float).unsqueeze(1)
        review_age_t = torch.tensor(review_age_list, dtype=torch.float).unsqueeze(1)

        edge_attr = torch.cat([rating_t, time_t, useful_t, funny_t, cool_t, age_t, review_age_t], dim=1)
        torch.save((source, dest, edge_attr, norm_ts), cache_path)

    # Impute and construct feature tensors
    user_feat_cols = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars', 'record_age', 'friend_count', 'elite_count']
    biz_feat_cols = ['stars', 'review_count', 'is_open', 'latitude', 'longitude']
    user_df[user_feat_cols] = user_df[user_feat_cols].fillna(0)
    business_df[biz_feat_cols] = business_df[biz_feat_cols].fillna(0)
    user_feats = torch.tensor(user_df[user_feat_cols].values, dtype=torch.float)

    business_simple_features=torch.from_numpy(business_df[biz_feat_cols].values).to(torch.float)  # [num_businesses, num_features_businesses]
    business_category_embeddings=torch.from_numpy(np.stack(business_df['category_embeddings'].values)).float()
    business_name_embeddings=torch.from_numpy(np.stack(business_df['name_embeddings'].values)).float()
    business_address_embeddingss=torch.from_numpy(np.stack(business_df['address_embeddings'].values)).float()
    biz_feats=torch.cat([business_simple_features,
                         business_category_embeddings,
                         business_name_embeddings,
                         business_address_embeddingss,
                        ],
                        dim=1)

    avg_rating = edge_attr[:, 0].mean().item()
    return source, dest, edge_attr, norm_ts, user_df, business_df, user_to_index, business_to_index, user_feats, biz_feats, avg_rating

# --------------- Training Loop ---------------

def train_model(
    source, dest, edge_attr, norm_ts,
    total_nodes, user_feats, biz_feats,
    biz_df, user_to_index, business_to_index,
    epochs=5, batch_size=2000, lr=1e-3, device="cpu", avg_rating=0.0
):
    dataset = TemporalEdgeDataset(source, dest, edge_attr, norm_ts)
    train_ds, val_ds, _ = temporal_split(dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_users = user_feats.size(0)
    model = MyTGN(total_nodes, user_feats.size(1), biz_feats.size(1)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    writer = SummaryWriter()

    best_val = float('inf')
    patience = 0

    for epoch in range(epochs):
        model.train()
        model.reset_memory()
        total_loss = 0.0

        for src_b, dst_b, eattr_b, ts_b in tqdm(train_loader, leave=False):
            src_b, dst_b, eattr_b, ts_b = src_b.to(device), dst_b.to(device), eattr_b.to(device), ts_b.to(device)
            u_feats = user_feats[src_b]
            b_feats = biz_feats[dst_b - num_users]
            combined = torch.cat([eattr_b, u_feats, b_feats], dim=1)
            preds = model(src_b, dst_b, ts_b, combined).squeeze()
            loss = loss_fn(preds, eattr_b[:, 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_memory(src_b, dst_b, ts_b, combined)
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train, epoch)

        # validation
        model.eval()
        model.reset_memory()
        val_loss = 0.0
        with torch.no_grad():
            for src_b, dst_b, eattr_b, ts_b in val_loader:
                src_b, dst_b, eattr_b, ts_b = src_b.to(device), dst_b.to(device), eattr_b.to(device), ts_b.to(device)
                u_feats = user_feats[src_b]
                b_feats = biz_feats[dst_b - num_users]
                combined = torch.cat([eattr_b, u_feats, b_feats], dim=1)
                preds = model(src_b, dst_b, ts_b, combined).squeeze()
                val_loss += loss_fn(preds, eattr_b[:, 0]).item()
        avg_val = val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val, epoch)
        # Recommendations logging
        sample_uid = 'qVc8ODYU5SZjKXVBgXdI7w'
        sample_lat = 39.955505
        sample_lng = -75.155564
        sample_radius = 1000
        sample_hour = 14
        recs = recommend(
            model, sample_uid, sample_lat, sample_lng,
            sample_radius, sample_hour,
            user_to_index, business_to_index, biz_df,
            top_k=10, device=device, avg_rating=avg_rating, user_feats=user_feats, biz_feats=biz_feats
        )
        rec_text = "\n".join(
            [f"{biz_df.loc[biz_df['business_id']==bid,'name'].values[0]}: {sc:.4f}"
             for bid, sc in recs]
        )
        writer.add_text("Recommendations", rec_text, epoch)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping.")
                break

    writer.close()
    model.load_state_dict(best_state)
    return model

# --------------- Recommendation ---------------

def recommend(
    model, uid, lat, lng, radius, hour,
    user_to_index, business_to_index, biz_df,
    top_k=10, device="cpu", avg_rating=0.0, user_feats=None, biz_feats=None
):
    if uid not in user_to_index:
        return []
    uidx = user_to_index[uid]
    candidates = []
    for _, rec in biz_df.iterrows():
        dist = haversine(lat, lng, rec.latitude, rec.longitude)
        if dist <= radius:
            candidates.append((rec.business_id, business_to_index[rec.business_id], rec))

    if not candidates:
        return []

    src_b = torch.tensor([uidx] * len(candidates), dtype=torch.long, device=device)
    dst_b = torch.tensor([idx for _, idx, _ in candidates], dtype=torch.long, device=device)
    # construct edge attributes for recommendations
    hour_feat = hour / 23.0
    hour_t = torch.full((len(candidates), 1), hour_feat, dtype=torch.float, device=device)
    rating_t = torch.full((len(candidates), 1), avg_rating, device=device)
    eattr_rec = torch.cat([
        rating_t,
        hour_t,
        torch.zeros(len(candidates), 5, device=device)
    ], dim=1)
    u_feats = user_feats[src_b]
    b_feats = biz_feats[dst_b - user_feats.size(0)]
    combined = torch.cat([eattr_rec, u_feats, b_feats], dim=1)
    with torch.no_grad():
        scores = model(src_b, dst_b, hour_t.squeeze(), combined).squeeze().cpu().tolist()
    top = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(bus.business_id, score) for ((_, _, bus), score) in top]

# --------------- Script Entry ---------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (source, dest, edge_attr, norm_ts,
     user_df, biz_df,
     user_to_index, business_to_index,
     user_feats, biz_feats, avg_rating) = load_and_preprocess()
    total_nodes = len(user_to_index) + len(business_to_index)
    model = train_model(
        source, dest, edge_attr, norm_ts,
        total_nodes, user_feats, biz_feats,
        biz_df, user_to_index, business_to_index,
        epochs=50, device=device, avg_rating=avg_rating
    )
    torch.save(model.state_dict(), "Model/tgn_model-business_addresses.pt")
    # sample recommend
    uid = 'qVc8ODYU5SZjKXVBgXdI7w'
    recs = recommend(model, uid, 39.955505, -75.155564, 3, 10,
                     user_to_index, business_to_index, biz_df,
                     top_k=20, device=device, avg_rating=avg_rating, user_feats=user_feats, biz_feats=biz_feats)
    print("Top Recommendations:")
    for bid, sc in recs:
        name = biz_df.loc[biz_df['business_id'] == bid, 'name'].values[0]
        print(f"  {name}: {sc:.4f}")
