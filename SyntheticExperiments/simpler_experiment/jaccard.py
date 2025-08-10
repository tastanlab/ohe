import pandas as pd
import numpy as np
import glob
import os
import argparse

def read_weights_csv(path):
    df = pd.read_csv(path)
    return set(df["Feature"].str.extract(r"Feature_(\d+)", expand=False).astype(int).squeeze())

def get_top_k_features(df, prefix, k=20):
    sub_df = df[df["Feature"].str.startswith(prefix)].copy()
    sub_df["Feature_Num"] = sub_df["Feature"].str.extract(r"Feature_(\d+)", expand=False).astype(int)
    sub_df["Abs_Attr"] = sub_df["Standardized_Coefficient_Abs"].abs()
    top_k_df = sub_df.sort_values("Abs_Attr", ascending=False).head(k)
    return set(top_k_df["Feature_Num"])

def compute_overlap_stats(selected_set, top_k_set):
    intersection = selected_set & top_k_set
    union = selected_set | top_k_set
    jaccard = len(intersection) / len(union) if union else 0
    return len(intersection), jaccard

def write_stats(f, label, overlap_count, jaccard):
    f.write(f"Top 20 overlapping features in {label}: {overlap_count}\n")
    f.write(f"Jaccard Index: {jaccard:.3f}\n\n")

# === Main ===
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="lpo")
args = parser.parse_args()


# === Load selected features ===
drug_selected = read_weights_csv("/simpler_experiment/weights_drug.csv")
cell_selected = read_weights_csv("/simpler_experiment/weights_cell.csv")

output_path = f"/simpler_experiment/{args.split}/jaccard_{args.split}.txt"

with open(output_path, "w") as f:
        
    for i in range(1,4):
        input_dir = f"/simpler_experiment/{args.split}/l1_1e-0{i}/"
        coef_path = os.path.join(input_dir, f"coefficients_abs{i}.csv")

        files = sorted(glob.glob(os.path.join(input_dir, f"coefficients_{args.split}_*.csv")))
        print(f"Found {len(files)} files")

        if len(files) == 0:
            raise FileNotFoundError(f"No files found in {input_dir}")

        df_all = None

        for idx, file in enumerate(files):
            df = pd.read_csv(file)
            df = df[['Feature', 'Standardized_Coefficient']]
            df.rename(columns={'Standardized_Coefficient': f'fold_{idx+1}'}, inplace=True)

            if df_all is None:
                df_all = df
            else:
                df_all = df_all.merge(df, on='Feature', how='outer')

        df_all = df_all.fillna(0)

        # Absolute average
        fold_cols = [col for col in df_all.columns if col.startswith('fold_')]
        df_all['Standardized_Coefficient_Abs'] = df_all[fold_cols].abs().mean(axis=1)

        # Signed average 
        df_all['Standardized_Coefficient_Signed'] = df_all[fold_cols].mean(axis=1)

        df_all = df_all.sort_values(by='Standardized_Coefficient_Abs', ascending=False)

        # save
        df_all.to_csv(coef_path, index=False)

        f.write(f"==== L1 = 1e-0{i} ====\n")
        
        # Drug1
        top_drug1 = get_top_k_features(df_all, "Drug1_")
        count1, jaccard1 = compute_overlap_stats(drug_selected, top_drug1)
        write_stats(f, "Drug1", count1, jaccard1)

        # Drug2
        top_drug2 = get_top_k_features(df_all, "Drug2_")
        count2, jaccard2 = compute_overlap_stats(drug_selected, top_drug2)
        write_stats(f, "Drug2", count2, jaccard2)

        # Cell line
        top_cell = get_top_k_features(df_all, "CellLine_")
        count_c, jaccard_c = compute_overlap_stats(cell_selected, top_cell)
        write_stats(f, "Cell", count_c, jaccard_c)

