from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


# Featurizer (Morgan fingerprints)
def featurize(smiles_list, radius=2, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros((n_bits,)))  # fallback if invalid SMILES
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
    return np.array(fps)


# Data prep (assuming dev_train / dev_val already split)

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

X_train = featurize(dev_train['SMILES'].to_list())
X_val   = featurize(dev_val['SMILES'].to_list())

y_train = dev_train[targets].to_numpy()
y_val   = dev_val[targets].to_numpy()


# Elastic Net Model
base_en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
model = make_pipeline(StandardScaler(), MultiOutputRegressor(base_en))

print("Training Elastic Net model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# Enhanced Debugging & Fixes

# 1. Diagnostic check (run this first)
print("\n=== DIAGNOSTIC ANALYSIS ===")
for i, target in enumerate(targets):
    unique_preds = np.unique(y_pred[:, i])
    pred_std = np.std(y_pred[:, i])
    actual_std = np.std(y_val[:, i])

    print(f"\n{target}:")
    print(f"  Predictions: {len(unique_preds)} unique values, std={pred_std:.6f}")
    print(f"  Actual:      std={actual_std:.6f}")
    print(f"  Pred range:  [{y_pred[:, i].min():.3f}, {y_pred[:, i].max():.3f}]")
    print(f"  True range:  [{y_val[:, i].min():.3f}, {y_val[:, i].max():.3f}]")


# 2. Safe correlation calculation
print("\n=== SAFE CORRELATIONS ===")
r_values_safe = {}
for i, target in enumerate(targets):
    if np.std(y_pred[:, i]) > 1e-8 and np.std(y_val[:, i]) > 1e-8:
        r, p_value = pearsonr(y_val[:, i], y_pred[:, i])
        r_values_safe[target] = r
        print(f"{target}: r={r:.4f}, p={p_value:.4f}")
    else:
        r_values_safe[target] = "constant_predictions"
        print(f"{target}: CONSTANT PREDICTIONS (no correlation possible)")


# 3. Try different alpha values for problematic targets
print("\n=== TESTING DIFFERENT ALPHAS ===")
alphas_to_test = [0.001, 0.01, 0.1, 1.0]

for alpha in alphas_to_test:
    model_test = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000, random_state=42))
    )
    model_test.fit(X_train, y_train)
    y_pred_test = model_test.predict(X_val)

    print(f"\nAlpha = {alpha}:")
    for i, target in enumerate(targets):
        pred_std = np.std(y_pred_test[:, i])
        if pred_std > 1e-8:
            r, _ = pearsonr(y_val[:, i], y_pred_test[:, i])
            print(f"  {target}: r={r:.4f}, pred_std={pred_std:.6f}")
        else:
            print(f"  {target}: still constant")


# 4. Individual target models (if multioutput is the issue)
print("\n=== INDIVIDUAL TARGET MODELS ===")
individual_results = {}
for i, target in enumerate(targets):
    model_single = make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42))
    model_single.fit(X_train, y_train[:, i])
    y_pred_single = model_single.predict(X_val)

    if np.std(y_pred_single) > 1e-8:
        r, _ = pearsonr(y_val[:, i], y_pred_single)
        individual_results[target] = r
        print(f"{target}: r={r:.4f} (individual model)")
    else:
        individual_results[target] = "still_constant"
        print(f"{target}: still constant (individual model)")


# Performance metrics
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred, multioutput='uniform_average')


from sklearn.impute import KNNImputer

# Handle missing targets
y_train_raw = dev_train[targets].to_numpy()
y_val_raw   = dev_val[targets].to_numpy()

# Impute missing values
imputer = KNNImputer(n_neighbors=5)  # finds 5 closest samples
y_train = imputer.fit_transform(y_train_raw)
y_val   = imputer.transform(y_val_raw)

print("Any NaNs left in y_train?", np.isnan(y_train).any())
print("Any NaNs left in y_val?", np.isnan(y_val).any())

# Elastic Net Model

base_en = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)
model = make_pipeline(StandardScaler(), MultiOutputRegressor(base_en))

print("Training Elastic Net model with Morgan fingerprints...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)


# Enhanced Debugging

print("\n=== DIAGNOSTIC ANALYSIS ===")
for i, target in enumerate(targets):
    unique_preds = np.unique(y_pred[:, i])
    pred_std = np.std(y_pred[:, i])
    actual_std = np.std(y_val[:, i])

    print(f"\n{target}:")
    print(f"  Predictions: {len(unique_preds)} unique values, std={pred_std:.6f}")
    print(f"  Actual:      std={actual_std:.6f}")
    print(f"  Pred range:  [{y_pred[:, i].min():.3f}, {y_pred[:, i].max():.3f}]")
    print(f"  True range:  [{y_val[:, i].min():.3f}, {y_val[:, i].max():.3f}]")

print("\n=== SAFE CORRELATIONS ===")
r_values_safe = {}
for i, target in enumerate(targets):
    if np.std(y_pred[:, i]) > 1e-8 and np.std(y_val[:, i]) > 1e-8:
        r, p_value = pearsonr(y_val[:, i], y_pred[:, i])
        r_values_safe[target] = r
        print(f"{target}: r={r:.4f}, p={p_value:.4f}")
    else:
        r_values_safe[target] = "constant_predictions"
        print(f"{target}: CONSTANT PREDICTIONS (no correlation possible)")

# Performance metrics

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred, multioutput='uniform_average')

print("\n=== SUMMARY ===")
print("Validation RMSE:", rmse)
print("Validation R²:", r2)
print("Per-target Pearson R:", r_values_safe)
