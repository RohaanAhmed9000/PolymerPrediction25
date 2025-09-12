# PyTorch Geometric (Colab / Linux)
!pip install torch==2.3.0 torchvision torchaudio
!pip install torch-geometric
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

# Load + Impute
df = pd.read_csv("train.csv")
targets = ['Tg','FFV','Tc','Density','Rg']

knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df[targets]), columns=targets)
df_knn = pd.concat([df[['SMILES']], df_knn], axis=1)

# Split imputed df
train_df, val_df = train_test_split(df_knn, test_size=0.2, random_state=42)

# SMILES → Graph
def smiles_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor(y, dtype=torch.float).unsqueeze(0)  # (5,)
    return Data(x=x, edge_index=edge_index, y=y)

class PolymerDataset(InMemoryDataset):
    def __init__(self, df, transform=None):
        super().__init__('.', transform)
        data_list = []
        for _, row in df.iterrows():
            graph = smiles_to_graph(row['SMILES'], row[targets].to_numpy(dtype=float))
            if graph is not None:
                data_list.append(graph)
        self.data, self.slices = self.collate(data_list)

# GNN Model
class GNNModel(nn.Module):
    def __init__(self, num_node_features, num_targets):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_targets)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Training Prep
train_dataset = PolymerDataset(train_df)
val_dataset   = PolymerDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(num_node_features=1, num_targets=5).to(device)  # atomic number feature
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


# Training Loop
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

# Validation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(out.cpu().numpy())

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# Metrics
r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
print("Validation R²:", r2)

for i, t in enumerate(targets):
    if np.std(y_true[:, i]) > 1e-8 and np.std(y_pred[:, i]) > 1e-8:
        r, _ = pearsonr(y_true[:, i], y_pred[:, i])
        print(f"{t}: Pearson R = {r:.3f}")
    else:
        print(f"{t}: Pearson R = constant predictions")
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np

# Suppose you already have:
# y_true: ground truth labels (numpy array, shape [n_samples, n_targets])
# y_pred: model predictions (numpy array, shape [n_samples, n_targets])

# RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Global R² (averaged over all outputs) ---
r2 = r2_score(y_true, y_pred, multioutput='uniform_average')

# Per-target correlations 
per_target_r = {}
for i, target in enumerate(targets):
    if np.std(y_true[:, i]) > 1e-8 and np.std(y_pred[:, i]) > 1e-8:
        r, _ = pearsonr(y_true[:, i], y_pred[:, i])
        per_target_r[target] = r
    else:
        per_target_r[target] = "constant_predictions"

print("\n=== PERFORMANCE METRICS ===")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²: {r2:.4f}")
print(f"Per-target Pearson R: {per_target_r}")
