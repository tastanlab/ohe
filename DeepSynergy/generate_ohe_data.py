'''import numpy as np
import pandas as pd
import pickle 

# Sample data
df = pd.read_csv('/cta/users/ebcandir/DeepSynergy/labels.csv', index_col=0) 

labels = pd.concat([df, df]) 


# Get all unique drugs from drug_a_name and drug_b_name
unique_drugs = pd.unique(df[['drug_a_name', 'drug_b_name']].values.ravel())

# One-hot encode drug_a_name and drug_b_name
one_hot_drug_a = pd.get_dummies(df['drug_a_name']).astype(int)
one_hot_drug_b = pd.get_dummies(df['drug_b_name']).astype(int)

# Ensure both one-hot encoded dataframes have the same columns (all unique drugs)
one_hot_drug_a = one_hot_drug_a.reindex(columns=unique_drugs, fill_value=0)
one_hot_drug_b = one_hot_drug_b.reindex(columns=unique_drugs, fill_value=0)

# One-hot encode cell_line
one_hot_cell_line = pd.get_dummies(df['cell_line']).astype(int)

one_hot_drug_ab = pd.concat([one_hot_drug_a, one_hot_drug_b],axis=0)
one_hot_drug_ba = pd.concat([one_hot_drug_b, one_hot_drug_a],axis=0) 
one_hot_cell_line = pd.concat([one_hot_cell_line, one_hot_cell_line],axis=0)

ohe_dataset = pd.concat([one_hot_drug_ab, one_hot_drug_ba, one_hot_cell_line],axis=1) 

# Save to CSV file
ohe_dataset.to_csv('/cta/users/ebcandir/DeepSynergy/ohe_dataset.csv', index=False)



# fold 0 is used for testing and fold 1 for validation (hyperparamter selection)
test_fold = 0
val_fold = 1


#indices of training data for hyperparameter selection: fold 2, 3, 4
idx_tr = np.where(np.logical_and(labels['fold']!=test_fold, labels['fold']!=val_fold))
#indices of validation data for hyperparameter selection: fold 1
idx_val = np.where(labels['fold']==val_fold)


#indices of training data for model testing: fold 1, 2, 3, 4
idx_train = np.where(labels['fold']!=test_fold)
#indices of test data for model testing: fold 0
idx_test = np.where(labels['fold']==test_fold)


ohe_tr = ohe_dataset.iloc[idx_tr].values
ohe_val = ohe_dataset.iloc[idx_val].values
ohe_train = ohe_dataset.iloc[idx_train].values
ohe_test = ohe_dataset.iloc[idx_test].values

y_tr = labels.iloc[idx_tr]['synergy'].values
y_val = labels.iloc[idx_val]['synergy'].values
y_train = labels.iloc[idx_train]['synergy'].values
y_test = labels.iloc[idx_test]['synergy'].values

pickle.dump((ohe_tr, ohe_val, ohe_train, ohe_test, y_tr, y_val, y_train, y_test), 
            open('/cta/users/ebcandir/DeepSynergy/ohe_data_test_fold%d.p'%(test_fold), 'wb'))'''

import numpy as np
import pandas as pd
import pickle

# Sample data
df = pd.read_csv('/cta/users/ebcandir/DeepSynergy/labels.csv', index_col=0) 

labels = pd.concat([df, df]) 

# Get all unique drugs from drug_a_name and drug_b_name
unique_drugs = pd.unique(df[['drug_a_name', 'drug_b_name']].values.ravel())

# One-hot encode drug_a_name and drug_b_name
one_hot_drug_a = pd.get_dummies(df['drug_a_name']).astype(int)
one_hot_drug_b = pd.get_dummies(df['drug_b_name']).astype(int)

# Ensure both one-hot encoded dataframes have the same columns (all unique drugs)
one_hot_drug_a = one_hot_drug_a.reindex(columns=unique_drugs, fill_value=0)
one_hot_drug_b = one_hot_drug_b.reindex(columns=unique_drugs, fill_value=0)

# One-hot encode cell_line
one_hot_cell_line = pd.get_dummies(df['cell_line']).astype(int)

one_hot_drug_ab = pd.concat([one_hot_drug_a, one_hot_drug_b], axis=0)
one_hot_drug_ba = pd.concat([one_hot_drug_b, one_hot_drug_a], axis=0) 
one_hot_cell_line = pd.concat([one_hot_cell_line, one_hot_cell_line], axis=0)

ohe_dataset = pd.concat([one_hot_drug_ab, one_hot_drug_ba, one_hot_cell_line], axis=1)

# Define folds
total_folds = 5
configuration_count = 20

config_idx = 0
for test_fold in range(total_folds):
    for val_fold in range(total_folds):
        if test_fold == val_fold:
            continue

        # Indices of training data for hyperparameter selection (exclude test and validation)
        idx_tr = np.where(np.logical_and(labels['fold'] != test_fold, labels['fold'] != val_fold))
        # Indices of validation data for hyperparameter selection
        idx_val = np.where(labels['fold'] == val_fold)

        # Indices of training data for model testing (exclude test)
        idx_train = np.where(labels['fold'] != test_fold)
        # Indices of test data for model testing
        idx_test = np.where(labels['fold'] == test_fold)

        # Extract data
        ohe_tr = ohe_dataset.iloc[idx_tr].values
        ohe_val = ohe_dataset.iloc[idx_val].values
        ohe_train = ohe_dataset.iloc[idx_train].values
        ohe_test = ohe_dataset.iloc[idx_test].values

        y_tr = labels.iloc[idx_tr]['synergy'].values
        y_val = labels.iloc[idx_val]['synergy'].values
        y_train = labels.iloc[idx_train]['synergy'].values
        y_test = labels.iloc[idx_test]['synergy'].values

        # Save the data for the current configuration
        pickle.dump(
            (ohe_tr, ohe_val, ohe_train, ohe_test, y_tr, y_val, y_train, y_test),
            open(f'/cta/users/ebcandir/DeepSynergy/ohe_data_test_fold{test_fold}_val_fold{val_fold}.p', 'wb')
        )

        config_idx += 1

        if config_idx >= configuration_count:
            break
    if config_idx >= configuration_count:
        break
