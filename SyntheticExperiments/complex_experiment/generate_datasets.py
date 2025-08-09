import numpy as np
import pandas as pd

# Feature generation

# Generate names for drugs and cell lines
drug_names = [f"Drug{i}" for i in range(1, 49001)]
cell_line_names = [f"CellLine{i}" for i in range(1, 24501)]

np.random.seed(42)

# Define mean and covariance for Gaussian-distributed feature vectors
mean_vector_drug = np.zeros(150)
cov_matrix_drug = np.eye(150)

mean_vector_cell = np.zeros(200)
cov_matrix_cell = np.eye(200)

# Generate feature vectors for each drug
drug_features = {
    drug: np.random.multivariate_normal(mean_vector_drug, cov_matrix_drug)
    for drug in drug_names
}

# Generate feature vectors for each cell line
cell_line_features = {
    cell_line: np.random.multivariate_normal(mean_vector_cell, cov_matrix_cell)
    for cell_line in cell_line_names
}

# Convert to df and label columns
drug_df = pd.DataFrame.from_dict(drug_features, orient='index')
drug_df.index.name = 'Drug'
drug_df.columns = [f'Feature_{i+1}' for i in range(150)]

cell_line_df = pd.DataFrame.from_dict(cell_line_features, orient='index')
cell_line_df.index.name = 'CellLine'
cell_line_df.columns = [f'Feature_{i+1}' for i in range(200)]

# Save full feature matrices
drug_csv_path = "/complex_experiment/non_repeated/49000_drug_features.csv"
cell_line_csv_path = "/complex_experiment/non_repeated/24500_cell_line_features.csv"

drug_df.to_csv(drug_csv_path)
cell_line_df.to_csv(cell_line_csv_path)

# load features back

drug_df = pd.read_csv(drug_csv_path, index_col="Drug")
cell_line_df = pd.read_csv(cell_line_csv_path, index_col="CellLine")

drug_features = drug_df.to_dict(orient="index")
cell_line_features = cell_line_df.to_dict(orient="index")

drug_names = list(drug_df.index)
cell_line_names = list(cell_line_df.index)

# select informative features randomly

np.random.seed(13)

# random feature indices for drugs
selected_indices_drug = np.random.choice(150, 50, replace=False)

# random feature indices for cell lines
selected_indices_cell = np.random.choice(200, 50, replace=False)

# Save selected indices for later use
np.savetxt("/complex_experiment/non_repeated/selected_indices_drug_unique.txt", selected_indices_drug, fmt="%d")
np.savetxt("/complex_experiment/non_repeated/selected_indices_cell_unique.txt", selected_indices_cell, fmt="%d")

# Also save for repeated splits (lpo lco lodo ldo)
np.savetxt("/complex_experiment/selected_indices_drug.txt", selected_indices_drug, fmt="%d")
np.savetxt("/complex_experiment/selected_indices_cell.txt", selected_indices_cell, fmt="%d")

# create masked feature matrices (to use to generate scores)


# Keep only selected features for drugs (others set to zero)
drug_selected_df = drug_df.copy()
drug_selected_df.loc[:, :] = 0
for idx in selected_indices_drug:
    drug_selected_df.iloc[:, idx] = drug_df.iloc[:, idx]

# Keep only selected features for cell lines
cell_selected_df = cell_line_df.copy()
cell_selected_df.loc[:, :] = 0
for idx in selected_indices_cell:
    cell_selected_df.iloc[:, idx] = cell_line_df.iloc[:, idx]

# Save masked matrices
drug_selected_df.to_csv("/complex_experiment/non_repeated/49000_drug_features_selected.csv")
cell_selected_df.to_csv("/complex_experiment/non_repeated/24500_cell_line_features_selected.csv")

# Create smaller subsets from the main dataset (used in all standard experiments with split strategies lpo lco lodo ldo)
# 

base_path = "/complex_experiment/non_repeated/"
target_path = "/complex_experiment/"

# Cell line subsets
cell_full = pd.read_csv(base_path + "24500_cell_line_features.csv", index_col="CellLine")
cell_selected = pd.read_csv(base_path + "24500_cell_line_features_selected.csv", index_col="CellLine")

cell_full_50 = cell_full.iloc[:50]
cell_selected_50 = cell_selected.iloc[:50]

cell_full_50.to_csv(target_path + "50_cell_line_features.csv")
cell_selected_50.to_csv(target_path + "50_cell_line_features_selected.csv")

# Drug subsets
drug_full = pd.read_csv(base_path + "49000_drug_features.csv", index_col="Drug")
drug_selected = pd.read_csv(base_path + "49000_drug_features_selected.csv", index_col="Drug")

drug_full_50 = drug_full.iloc[:50]
drug_selected_50 = drug_selected.iloc[:50]

drug_full_50.to_csv(target_path + "50_drug_features.csv")
drug_selected_50.to_csv(target_path + "50_drug_features_selected.csv")
