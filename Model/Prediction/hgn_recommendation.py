import os
import pandas as pd
import torch
import numpy as np
from math import radians, cos, sin, asin, sqrt
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

# --- HGN model definition (matching Model/hgn_recommendation.py) --- #
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

class HGNModel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=32):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# --- Utility: haversine distance --- #
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# --- Load data and model once at import --- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Load business & user dfs
USER_DF = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'preprocessed_user.parquet'), engine='pyarrow')
BUSINESS_DF = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'preprocessed_business.parquet'), engine='pyarrow')
# Load graph data and model
HETERO_DATA = torch.load(os.path.join(BASE_DIR, 'data', 'hetero_data.pt'), weights_only=False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = HGNModel(HETERO_DATA.metadata(), hidden_channels=32).to(DEVICE)
_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'hgn_model.pt'), map_location=DEVICE))
_model.eval()

def recommend(user_id, user_location, radius_km, day, hour, top_k=20):
    """
    Return top_k recommended businesses for a user within a geographic radius and open at given day and hour.
    Args:
        user_id: original user identifier matching 'user_id' column in preprocessed_user.parquet
        user_location: tuple of (latitude, longitude)
        radius_km: search radius in kilometers
        day: weekday name matching keys in 'hours' column (e.g. 'Monday')
        hour: current hour (0-23) for business hours filtering
        top_k: number of businesses to return (default 20)
    Returns:
        List of dicts: [{ 'business_id', 'name', 'latitude', 'longitude', 'predicted_rating' }, ...]
    """
    # Map user_id to internal index
    if 'user_id' in USER_DF.columns:
        try:
            u_idx = int(USER_DF.index[USER_DF['user_id'] == user_id][0])
        except Exception:
            raise KeyError(f"User '{user_id}' not found in preprocessed data.")
    else:
        raise KeyError("Column 'user_id' not found in user DataFrame.")

    # Filter businesses by distance and open status
    lat0, lon0 = user_location
    df = BUSINESS_DF.copy()
    # compute distance
    df['distance_km'] = df.apply(lambda row: haversine(lon0, lat0, row['longitude'], row['latitude']), axis=1)
    df = df[df['distance_km'] <= radius_km]

    # filter open status
    if 'hours' in df.columns:
        # hours stored as dict-like per row: keys are weekday names
        def open_at(h):
            if not isinstance(h, dict):
                return False
            ts = h.get(day)
            if not ts:
                return False
            start, end = ts.split('-')
            start_h = int(start.split(':')[0])
            end_h = int(end.split(':')[0])
            return start_h <= hour < end_h
        df = df[df['hours'].apply(open_at)]
    elif 'is_open' in df.columns:
        df = df[df['is_open'] == 1]

    if df.empty:
        return []
    # print(df)

    # build edge_index for prediction
    # map business_id to index
    if 'business_id' in df.columns:
        biz_idx_map = {bid: int(idx) for idx, bid in enumerate(BUSINESS_DF['business_id'])}
        candidates = df['business_id'].tolist()
        b_indices = [biz_idx_map[bid] for bid in candidates]
    else:
        # fallback: assume DataFrame index aligns with model indices
        b_indices = list(df.index.astype(int))
        candidates = df.index.astype(str).tolist()

    # prepare edge_label_index
    user_tensor = torch.tensor([u_idx] * len(b_indices), dtype=torch.long)
    biz_tensor = torch.tensor(b_indices, dtype=torch.long)
    edge_label_index = torch.stack([user_tensor, biz_tensor], dim=0)

    # predict
    with torch.no_grad():
        out = _model(HETERO_DATA.x_dict, HETERO_DATA.edge_index_dict, edge_label_index.to(DEVICE))
        scores = out.clamp(0, 5).cpu().numpy().flatten()

    # select top_k
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank in top_idx:
        bid = candidates[rank]
        row = BUSINESS_DF.loc[BUSINESS_DF['business_id'] == bid].iloc[0]
        results.append({
            'business_id': bid,
            'name': row.get('name', ''),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'predicted_rating': float(scores[rank]),
        })
    return results

top20 = recommend(
    user_id="qVc8ODYU5SZjKXVBgXdI7w",
    user_location=(39.955505, -75.155564),
    radius_km=1000,
    day="Friday",
    hour=14
)

if __name__ == "__main__":
    for i in top20:
        print(i)
