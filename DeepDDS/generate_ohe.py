import csv
import pandas as pd
import numpy as np


df = pd.read_csv('/cta/users/ebcandir/DeepDDs/data/new_labels_0_10.csv')

# save features for ohe drug1
drug1_features = pd.get_dummies(df['drug1']).astype(int)
drug1_features.to_csv('/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_drug1_features_ohe.csv', index=False)

# save features for ohe drug2
drug2_features = pd.get_dummies(df['drug2']).astype(int)
drug2_features.to_csv('/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_drug2_features_ohe.csv', index=False)

# save one hot encoded features for cells
cell_features_ohe = pd.get_dummies(df['cell']).astype(int)
cell_features_ohe.to_csv('/cta/users/ebcandir/DeepDDs/data/new_labels_0_10_cell_features_ohe.csv', index=False)