import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

l1_value = ["1", "2", "3"]
sp = ["lpo", "lco", "lodo", "ldo"]

output_dir = "/complex_experiment"
os.makedirs(output_dir, exist_ok=True)

for i in l1_value:
    for split in sp:
        print(i, split)

        # inputs
        ig_csv_path = f"/complex_experiment/{split}/l1_1e-0{i}/ig_avg_abs.csv"
        drug_true_idx_path = "/complex_experiment/selected_indices_drug.txt"
        cell_true_idx_path = "/complex_experiment/selected_indices_cell.txt"

        # read data
        df = pd.read_csv(ig_csv_path)
        df["Abs_Attr"] = df["Avg_Attribution_Abs"].abs()

        drug_true_indices = np.loadtxt(drug_true_idx_path, dtype=int).tolist()
        cell_true_indices = np.loadtxt(cell_true_idx_path, dtype=int).tolist()

        # get indices and create true_labels 
        true_labels = []
        for feature in df["Feature"]:
            if feature.startswith("Drug1_") or feature.startswith("Drug2_"):
                idx = int(feature.split("_f")[-1])
                true_labels.append(1 if idx in drug_true_indices else 0)
            elif feature.startswith("Cell_"):
                idx = int(feature.split("_f")[-1])
                true_labels.append(1 if idx in cell_true_indices else 0)
            else:
                true_labels.append(0)

        true_labels = np.array(true_labels)
        ig_scores = df["Abs_Attr"].values

        # calculate aupr
        precision, recall, thresholds = precision_recall_curve(true_labels, ig_scores)
        aupr = auc(recall, precision)
        print(f"AUPR Score: {aupr:.4f}")
