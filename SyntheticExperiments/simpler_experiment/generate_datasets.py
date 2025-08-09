import numpy as np
import pandas as pd

#Feature generation

#Generate names for drugs and cell lines
drug_names = [f"Drug{i}" for i in range(1, 49001)]
cell_line_names = [f"CellLine{i}" for i in range(1, 24501)]

np.random.seed(42)

#Define mean and covariance for Gaussian-distributed feature vectors
mean_vector_drug = np.zeros(100)
cov_matrix_drug = np.eye(100)

mean_vector_cell = np.zeros(100)
cov_matrix_cell = np.eye(100)

drug_features = {
    drug: np.random.multivariate_normal(mean_vector_drug, cov_matrix_drug)
    for drug in drug_names
}

cell_line_features = {
    cell_line: np.random.multivariate_normal(mean_vector_cell, cov_matrix_cell)
    for cell_line in cell_line_names
}

#Convert to df and label columns
drug_df = pd.DataFrame.from_dict(drug_features, orient='index')
drug_df.index.name = 'Drug'
drug_df.columns = [f'Feature_{i+1}' for i in range(100)]

cell_line_df = pd.DataFrame.from_dict(cell_line_features, orient='index')
cell_line_df.index.name = 'CellLine'
cell_line_df.columns = [f'Feature_{i+1}' for i in range(100)]

#Save full feature matrices
drug_csv_path = "/simpler_experiment/non_repeated/49000_drug_features.csv"
cell_line_csv_path = "/simpler_experiment/non_repeated/24500_cell_line_features.csv"

drug_df.to_csv(drug_csv_path)
cell_line_df.to_csv(cell_line_csv_path)

print("saved 4900 24500") #load features back

drug_df = pd.read_csv(drug_csv_path, index_col="Drug")
cell_line_df = pd.read_csv(cell_line_csv_path, index_col="CellLine")

drug_features = drug_df.to_dict(orient="index")
cell_line_features = cell_line_df.to_dict(orient="index")

drug_names = list(drug_df.index)
cell_line_names = list(cell_line_df.index)

print("selected random weights") #select random features and assing weights to that features

np.random.seed(13)

#random selection for drugs
selected_indices_drug = np.random.choice(100, 20, replace=False)
weights_drug = np.random.uniform(0.5, 10.0, 20)

#random selection for cell lines
selected_indices_cell = np.random.choice(100, 20, replace=False)
weights_cell = np.random.uniform(0.5, 10.0, 20)

#Save weights
weights_drug_df = pd.DataFrame({
    'Feature': [f'Feature_{i+1}' for i in selected_indices_drug],
    'Weight': weights_drug
})
weights_drug_csv_path = "/simpler_experiment/weights_drug.csv"
weights_drug_df.to_csv(weights_drug_csv_path, index=False)

weights_cell_df = pd.DataFrame({
    'Feature': [f'Feature_{i+1}' for i in selected_indices_cell],
    'Weight': weights_cell
})
weights_cell_csv_path = "/simpler_experiment/weights_cell.csv"
weights_cell_df.to_csv(weights_cell_csv_path, index=False)


print("saved weights") #Shuffle drugs and pair them (unique pairs)
np.random.seed(42)
np.random.shuffle(drug_names)
drug_pairs = [(drug_names[i], drug_names[i + 1]) for i in range(0, len(drug_names), 2)]  #24,500 pairs

#Shuffle cell lines
np.random.seed(42)
np.random.shuffle(cell_line_names)

#Combine into triplets (1 pair + 1 unique cell line)
combinations = [(d1, d2, c) for (d1, d2), c in zip(drug_pairs, cell_line_names)]


print("combinded") #calculate scores

synergy_scores = []

for idx, (d1, d2, c) in enumerate(combinations):
    if idx % 5000 == 0:
        print(f"Processing {idx}/{len(combinations)}...")

    drug1_feat = np.array(list(drug_features[d1].values()))
    drug2_feat = np.array(list(drug_features[d2].values()))
    cell_feat = np.array(list(cell_line_features[c].values()))

    #Initialize zero arrays
    drug1_weighted = np.zeros(100)
    drug2_weighted = np.zeros(100)
    cell_weighted = np.zeros(100)

    #Apply weights only to selected features
    drug1_weighted[selected_indices_drug] = drug1_feat[selected_indices_drug] * weights_drug
    drug2_weighted[selected_indices_drug] = drug2_feat[selected_indices_drug] * weights_drug
    cell_weighted[selected_indices_cell] = cell_feat[selected_indices_cell] * weights_cell

    error = np.random.normal(0, 0.1)
    synergy_score = (np.sum(drug1_weighted + drug2_weighted + 2 * cell_weighted) / 2) + error

    synergy_scores.append((d1, d2, c, synergy_score))

#save

synergy_df = pd.DataFrame(synergy_scores, columns=['Drug_1', 'Drug_2', 'CellLine', 'SynergyScore'])

synergy_csv_path = "/simpler_experiment/non_repeated/synergy_scores_non_repeated.csv"
synergy_df.to_csv(synergy_csv_path, index=False)


print("saved scores") 

#Create smaller subsets from the main dataset (used in all standard experiments with split strategies lpo lco lodo ldo)
 

base_path = "/simpler_experiment/non_repeated/"
target_path = "/simpler_experiment/"

#Cell line subsets
cell_full = pd.read_csv(base_path + "24500_cell_line_features.csv", index_col="CellLine")

cell_line_df = cell_full.iloc[:50]

cell_line_df.to_csv(target_path + "50_cell_line_features.csv")

#Drug subsets
drug_full = pd.read_csv(base_path + "49000_drug_features.csv", index_col="Drug")

drug_df = drug_full.iloc[:50]

drug_df.to_csv(target_path + "50_drug_features.csv")

#read features

drug_df = pd.read_csv(target_path + "50_drug_features.csv", index_col="Drug")
cell_line_df = pd.read_csv(target_path + "50_cell_line_features.csv", index_col="CellLine")

drug_features = drug_df.to_dict(orient="index")
cell_line_features = cell_line_df.to_dict(orient="index")

drug_names = list(drug_df.index)
cell_line_names = list(cell_line_df.index)

#read existing weights (we randomly generated while creating regular repeated dataset) 
weights_drug_df = pd.read_csv(weights_drug_csv_path)
weights_cell_df = pd.read_csv(weights_cell_csv_path)

#extract feature indices (e.g., Feature_5 is index 4)
selected_indices_drug = [int(f.split('_')[1]) - 1 for f in weights_drug_df['Feature']]
weights_drug = weights_drug_df['Weight'].values

selected_indices_cell = [int(f.split('_')[1]) - 1 for f in weights_cell_df['Feature']]
weights_cell = weights_cell_df['Weight'].values
print("extract features")

#create triplets

drug_pairs = [(d1, d2) for i, d1 in enumerate(drug_names) for d2 in drug_names[i+1:]]
combinations = [(d1, d2, c) for d1, d2 in drug_pairs for c in cell_line_names]

print("triplets 61k") #calculate their scores

synergy_scores = []
drug1_weighted_features = []
drug2_weighted_features = []
cell_weighted_features = []

for d1, d2, c in combinations:
    drug1_feat = np.array(list(drug_features[d1].values()))
    drug2_feat = np.array(list(drug_features[d2].values()))
    cell_feat = np.array(list(cell_line_features[c].values()))

    #initialize zero arrays
    drug1_weighted = np.zeros(100)
    drug2_weighted = np.zeros(100)
    cell_weighted = np.zeros(100)

    #apply weights only to selected true features
    drug1_weighted[selected_indices_drug] = drug1_feat[selected_indices_drug] * weights_drug
    drug2_weighted[selected_indices_drug] = drug2_feat[selected_indices_drug] * weights_drug
    cell_weighted[selected_indices_cell] = cell_feat[selected_indices_cell] * weights_cell

    error = np.random.normal(0, 0.1)
    synergy_score = (np.sum(drug1_weighted + drug2_weighted + 2 * cell_weighted) / 2) + error

    synergy_scores.append((d1, d2, c, synergy_score))
    drug1_weighted_features.append(drug1_weighted[selected_indices_drug])
    drug2_weighted_features.append(drug2_weighted[selected_indices_drug])
    cell_weighted_features.append(cell_weighted[selected_indices_cell])

#save

synergy_df = pd.DataFrame(synergy_scores, columns=['Drug_1', 'Drug_2', 'CellLine', 'SynergyScore'])

#save final dataframe
synergy_csv_path = "/simpler_experiment/synergy_scores.csv"
synergy_df.to_csv(synergy_csv_path, index=False)

print("done")