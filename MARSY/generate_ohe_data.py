import pandas as pd

# Load the CSV file
df = pd.read_csv('/cta/users/ebcandir/MARSY/data/data.csv')

# Generate OHE for Drug1
ohe_drug1 = pd.get_dummies(df['Drug1_PC3'], prefix='Drug1').astype(int)
print(ohe_drug1.shape)
# Generate OHE for Drug2
ohe_drug2 = pd.get_dummies(df['Drug2_PC3'], prefix='Drug2').astype(int)
print(ohe_drug2.shape)
# Generate OHE for Cell Line
ohe_cellline = pd.get_dummies(df['Cell_line'], prefix='CellLine').astype(int)
print(ohe_cellline.shape)
# Concatenate the OHE columns with the original dataframe
df_ohe = pd.concat([df, ohe_drug1, ohe_drug2, ohe_cellline], axis=1)
df_ohe.drop(columns=['Drug1_PC3', 'Drug1_MCF7', 'Drug2_PC3', 'Drug2_MCF7', 'Cell_line', 'Ri1',	'Ri2',	'Synergy_Zip',	'Synergy_Smean',	'Tissue'], inplace=True)

print(df_ohe.head())
# Save the result to a CSV file
df_ohe.to_csv('/cta/users/ebcandir/MARSY/data/output_ohe.csv', index=False)

print("One-hot encoded data saved to output_ohe.csv")
