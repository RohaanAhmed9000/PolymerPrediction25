
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
# import os, sys
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------
# Load dataset
# -------------------
csv_path = '.\\input\\neurips-open-polymer-prediction-2025\\train.csv'  # adjust if needed
train_df = pd.read_csv(csv_path)

# 1. split off 20% for dev_test
temp_df, dev_test = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 2. split remaining 80% into 75% train / 25% valid → 0.6 / 0.2 overall
dev_train, dev_val = train_test_split(
    temp_df,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 of the original
    random_state=42,
    shuffle=True
)

# Verify sizes
print(f"Total rows:   {len(train_df)}")
print(f"Dev train:    {len(dev_train)} ({len(dev_train)/len(train_df):.2%})")
print(f"Dev valid:    {len(dev_val)} ({len(dev_val)/len(train_df):.2%})")
print(f"Dev test:     {len(dev_test)} ({len(dev_test)/len(train_df):.2%})")
print(f"Polymer example:{dev_train['SMILES'].to_list()[:3]}")
print(f"Columns:{dev_train.columns}")

# -------------------
# Featurizer
# -------------------
from rdkit.Chem import rdFingerprintGenerator

def featurize(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,))
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros((n_bits,)))
    return np.array(fps)

# def featurize(smiles_list):
#     # Simple placeholder featurizer: length and counts
#     return np.array([[len(s), sum(c.isalpha() for c in s)] for s in smiles_list])

X_train = featurize(dev_train['SMILES'].to_list())
X_val   = featurize(dev_val['SMILES'].to_list())

# -------------------
# Handle missing targets
# -------------------
y_train_raw = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
y_val_raw   = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)  # finds 5 closest samples
y_train = imputer.fit_transform(y_train_raw)
y_val   = imputer.transform(y_val_raw)


# imputer = SimpleImputer(strategy="mean")
# y_train = imputer.fit_transform(y_train_raw)
# y_val   = imputer.transform(y_val_raw)

print("Any NaNs left in y_train?", np.isnan(y_train).any())
print("Any NaNs left in y_val?", np.isnan(y_val).any())

# -------------------
# SVR Model
# -------------------
base_svr = SVR(kernel='rbf', C=10, epsilon=0.1, gamma=0.01)
model = make_pipeline(StandardScaler(), MultiOutputRegressor(base_svr))

print("Training SVR model...")
model.fit(X_train, y_train)

# Validation performance
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2s = r2_score(y_val, y_pred, multioutput="raw_values")
print(dict(zip(["Tg","FFV","Tc","Density","Rg"], r2s)))
r2 = r2_score(y_val, y_pred, multioutput='uniform_average')

print("Validation RMSE:", rmse)
print("Validation R²:", r2)
print("Model initialized and trained successfully")