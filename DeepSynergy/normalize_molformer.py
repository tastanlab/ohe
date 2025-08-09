import numpy as np
import pandas as pd
import pickle 
import gzip

embedding_dim = 768
cell_dim = 39
drug_dim = 2 * embedding_dim
norm = 'tanh' 

# normalization for drugs
def normalize_drugs_only(X, means1=None, std1=None, feat_filt=None, norm='tanh'):
    X_drug = X[:, :drug_dim]
    X_cell = X[:, drug_dim:]

    if std1 is None:
        std1 = np.nanstd(X_drug, axis=0)
    if feat_filt is None:
        feat_filt = std1 != 0
    X_drug = X_drug[:, feat_filt]
    X_drug = np.ascontiguousarray(X_drug)
    if means1 is None:
        means1 = np.mean(X_drug, axis=0)
    X_drug = (X_drug - means1) / std1[feat_filt]

    if norm == 'norm':
        X_combined = np.concatenate([X_drug, X_cell], axis=1)
        return X_combined, means1, std1, feat_filt
    elif norm == 'tanh':
        X_drug = np.tanh(X_drug)
        X_combined = np.concatenate([X_drug, X_cell], axis=1)
        return X_combined, means1, std1, feat_filt

with gzip.open('/X_molformer.p.gz', 'rb') as file:
    X = pickle.load(file)

labels = pd.read_csv('/labels.csv', index_col=0)
labels = pd.concat([labels, labels], ignore_index=True)

#cross validation
total_folds = 5
configuration_count = 20
config_idx = 0

for test_fold in range(total_folds):
    for val_fold in range(total_folds):
        if test_fold == val_fold:
            continue

        idx_tr = np.where((labels['fold'] != test_fold) & (labels['fold'] != val_fold))
        idx_val = np.where(labels['fold'] == val_fold)
        idx_train = np.where(labels['fold'] != test_fold)
        idx_test = np.where(labels['fold'] == test_fold)

        X_tr = X[idx_tr]
        X_val = X[idx_val]
        X_train = X[idx_train]
        X_test = X[idx_test]

        y_tr = labels.iloc[idx_tr]['synergy'].values
        y_val = labels.iloc[idx_val]['synergy'].values
        y_train = labels.iloc[idx_train]['synergy'].values
        y_test = labels.iloc[idx_test]['synergy'].values

        X_tr, mean, std, feat_filt = normalize_drugs_only(X_tr, norm=norm)
        X_val, mean, std, feat_filt = normalize_drugs_only(X_val, mean, std, feat_filt=feat_filt, norm=norm)
        X_train, mean, std, feat_filt = normalize_drugs_only(X_train, mean, std, feat_filt=feat_filt, norm=norm)
        X_test, mean, std, feat_filt = normalize_drugs_only(X_test, mean, std, feat_filt=feat_filt, norm=norm)

        # save
        out_path = f'/molformer/data/data_test_fold{test_fold}_val_fold{val_fold}_{norm}.p'
        with open(out_path, 'wb') as f:
            pickle.dump((X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test), f)

        print(f"Saved: {out_path}")
        config_idx += 1
        if config_idx >= configuration_count:
            break
    if config_idx >= configuration_count:
        break
