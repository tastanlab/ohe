import pandas as pd
import argparse

def read_txt_as_set(path):
    with open(path) as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def get_top_k_features(df, prefix, k=50):
    sub_df = df[df["Feature"].str.startswith(prefix)].copy()
    sub_df["Feature_Index"] = sub_df["Feature"].str.extract(r"_f(\d+)")[0].astype(int)
    sub_df["Abs_Attr"] = sub_df["Avg_Attribution_Abs"].abs()
    top_k_df = sub_df.sort_values("Abs_Attr", ascending=False).head(k)
    return top_k_df, set(top_k_df["Feature_Index"])

def compute_overlap_stats(selected_set, top_k_set):
    intersection = selected_set & top_k_set
    union = selected_set | top_k_set
    jaccard = len(intersection) / len(union) if union else 0
    return len(intersection), jaccard

def write_stats(f, label, overlap_count, jaccard):
    f.write(f"Top 50 overlapping features in {label}: {overlap_count}\n")
    f.write(f"Jaccard Index: {jaccard:.3f}\n\n")

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="lpo")
args = parser.parse_args()

drug_selected = read_txt_as_set("/complex_experiment/selected_indices_drug.txt")
cell_selected = read_txt_as_set("/complex_experiment/selected_indices_cell.txt")

output_path = f"/complex_experiment/{args.split}/feature_jaccard_{args.split}.txt"

with open(output_path, "w") as f:
    for i in range(1,4):
        f.write(f"==== L1 = 1e-0{i} ====\n")
        
        df = pd.read_csv(f"/complex_experiment/{args.split}/l1_1e-0{i}/ig_avg_abs{i}.csv")

        # Drug1
        top_drug1, indices_drug1 = get_top_k_features(df, "Drug1_")
        count1, jaccard1 = compute_overlap_stats(drug_selected, indices_drug1)
        write_stats(f, "Drug1", count1, jaccard1)

        # Drug2
        top_drug2, indices_drug2 = get_top_k_features(df, "Drug2_")
        count2, jaccard2 = compute_overlap_stats(drug_selected, indices_drug2)
        write_stats(f, "Drug2", count2, jaccard2)

        # Cell
        top_cell, indices_cell = get_top_k_features(df, "Cell_")
        count_c, jaccard_c = compute_overlap_stats(cell_selected, indices_cell)
        write_stats(f, "Cell", count_c, jaccard_c)
