import pandas as pd

# File paths
pc3_path = "/cta/users/ebcandir/MARSY/data/PC3_drugs_gene_expression.csv"
mcf7_path = "/cta/users/ebcandir/MARSY/data/MCF7_drugs_gene_expression.csv"
cell_lines_path = "/cta/users/ebcandir/MARSY/data/75_cell_lines_gene_expression.csv"
data_csv_path = "/cta/users/ebcandir/MARSY/data/data.csv"

# Load the data
pc3_data = pd.read_csv(pc3_path,  header=0)  # Drug names are headers
if pc3_data.columns[0] == pc3_data.columns.name:  # Check if the first column is incorrectly treated as an index
    pc3_data.reset_index(inplace=True)
mcf7_data = pd.read_csv(mcf7_path,  header=0)  # Drug names are headers
if mcf7_data.columns[0] == mcf7_data.columns.name:  # Check if the first column is incorrectly treated as an index
    mcf7_data.reset_index(inplace=True)
cell_lines_data = pd.read_csv(cell_lines_path, header=0)  # Read headers from the first row
if cell_lines_data.columns[0] == cell_lines_data.columns.name:  # Check if the first column is incorrectly treated as an index
    cell_lines_data.reset_index(inplace=True)
data = pd.read_csv(data_csv_path)  # Read only the first 2 rows

print(cell_lines_data)

# Initialize the feature dataset
feature_dataset = []

# Iterate through each row in the input data
for _, row in data.iterrows():
    print(row)
    drug1_pc3_features = pc3_data["ilomastat"].values
    drug1_mcf7_features = mcf7_data[row['Drug1_MCF7']].values
    drug2_pc3_features = pc3_data[row['Drug2_PC3']].values
    drug2_mcf7_features = mcf7_data[row['Drug2_MCF7']].values
    cell_line_features = cell_lines_data[row['Cell_line']].values

    # Concatenate all features for the row
    combined_features = (
        list(drug1_pc3_features) +
        list(drug1_mcf7_features) +
        list(drug2_pc3_features) +
        list(drug2_mcf7_features) +
        list(cell_line_features)
    )

    # Add to the feature dataset
    feature_dataset.append(combined_features)

# Create a DataFrame for the features
feature_columns = (
    [f"Drug1_PC3_{col}" for col in pc3_data.index] +
    [f"Drug1_MCF7_{col}" for col in mcf7_data.index] +
    [f"Drug2_PC3_{col}" for col in pc3_data.index] +
    [f"Drug2_MCF7_{col}" for col in mcf7_data.index] +
    [f"Cell_Line_{col}" for col in cell_lines_data.index]
)

features_df = pd.DataFrame(feature_dataset, columns=feature_columns)

# Save the features to a CSV file
output_path = "/cta/users/ebcandir/MARSY/data/feature_dataset.csv"
features_df.to_csv(output_path, index=False)

print(f"Feature dataset saved to {output_path}")
