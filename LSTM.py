import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('.\\input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


csv_path = ".\\input\\neurips-open-polymer-prediction-2025\\train.csv"


train_df = pd.read_csv(csv_path)

# 1. split off 20% for dev_test
temp_df, dev_test = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,  # for reproducibility
    shuffle=True
)


# 2. split the remaining 80% into 75% train / 25% valid → 0.6 / 0.2 overall
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


from tqdm.notebook import tqdm as notebook_tqdm
import tqdm
tqdm.tqdm = notebook_tqdm
tqdm.trange = notebook_tqdm

from torch_molecule import LSTMMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

search_parameters = {
    "output_dim": ParameterSpec(ParameterType.INTEGER, (8, 32)),
    "LSTMunits": ParameterSpec(ParameterType.INTEGER, (30, 120)),
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
}

lstm = LSTMMolecularPredictor(
    task_type="regression",
    num_task=5,
    batch_size=10,
    epochs=5,
    verbose=True
)

print("Model initialized successfully")
X_train = dev_train['SMILES'].to_list()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
X_val = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
lstm.autofit(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    search_parameters=search_parameters,
    n_trials = 10 # number of times searching the best hyper-parameters
)
