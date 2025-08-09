import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import argparse
import os
import shap
import random
import gzip
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearSynergyModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearSynergyModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


 
def train_model(model, train_loader, val_loader, save_path, epochs=1000, lr=1e-3, patience=100, l1_lambda=1e-3):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    results = {"train_loss": [], "val_loss": []}

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)

            l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item()

        train_loss_avg = total_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        results["train_loss"].append(train_loss_avg)
        results["val_loss"].append(val_loss_avg)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss_avg:.4f}")

        # === Early stopping kontrolü ===
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            # En iyi modeli kaydet
            torch.save(model.state_dict(), save_path.replace('.json', '_best.pt'))
            print(f"--> New best val_loss: {best_val_loss:.4f}, model saved.")
        else:
            patience_counter += 1
            print(f"--> EarlyStopping patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"--> Early stopping triggered at epoch {epoch + 1}")
                break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path.replace('.json', '.pt'))  
    with open(save_path, "w") as f:
        json.dump(results, f)


def get_feature_arrays(indices, synergy_df, drug_feat_dict, cell_feat_dict):
    drug1_features = []
    drug2_features = []
    cell_features = []
    labels = []
    for idx in indices:
        row = synergy_df.iloc[idx]
        drug1_features.append(drug_feat_dict[row['Drug_1']])
        drug2_features.append(drug_feat_dict[row['Drug_2']])
        cell_features.append(cell_feat_dict[row['CellLine']])
        labels.append(row['SynergyScore'])

        # Swapped order
        drug1_features.append(drug_feat_dict[row['Drug_2']])
        drug2_features.append(drug_feat_dict[row['Drug_1']])
        cell_features.append(cell_feat_dict[row['CellLine']])
        labels.append(row['SynergyScore'])
    return np.array(drug1_features), np.array(drug2_features), np.array(cell_features), np.array(labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/simpler_experiment/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--split", type=str, default="lpo")
    args = parser.parse_args()

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    synergy_file = "/simpler_experiment/synergy_scores.csv"
    drug_feat_file = "/simpler_experiment/50_drug_features.csv"
    cell_feat_file = "/simpler_experiment/50_cell_line_features.csv"

    synergy_df = pd.read_csv(synergy_file)
    drug_features_df = pd.read_csv(drug_feat_file)
    cell_features_df = pd.read_csv(cell_feat_file)
    drug_feature_dict = drug_features_df.set_index('Drug').T.to_dict('list')
    cell_feature_dict = cell_features_df.set_index('CellLine').T.to_dict('list')

    l1_values = [1e-2, 1e-1]

    for l1_lambda in l1_values:
        print(f"\n=== Running with L1 lambda = {l1_lambda} ===")
        l1_dir = os.path.join(args.save_dir, args.split, f"l1_{l1_lambda:.0e}")
        os.makedirs(l1_dir, exist_ok=True)

        all_mse = []



        for fold_num in range(1, 11):
            print(f"\n=== Processing Fold {fold_num} ===")
            train_idx_file = f"/simpler_experiment/{args.split}/splits/train_set_{args.split}_{fold_num}.txt"
            val_idx_file = f"/simpler_experiment/{args.split}/splits/val_set_{args.split}_{fold_num}.txt"
            test_idx_file = f"/simpler_experiment/{args.split}/splits/test_set_{args.split}_{fold_num}.txt"

            train_indices = np.loadtxt(train_idx_file, dtype=int)
            val_indices = np.loadtxt(val_idx_file, dtype=int)
            test_indices = np.loadtxt(test_idx_file, dtype=int)

            # === Load feature arrays ===
            train_drug1_vec, train_drug2_vec, train_cell_vec, y_train = get_feature_arrays(train_indices, synergy_df, drug_feature_dict, cell_feature_dict)
            val_drug1_vec, val_drug2_vec, val_cell_vec, y_val = get_feature_arrays(val_indices, synergy_df, drug_feature_dict, cell_feature_dict)
            test_drug1_vec, test_drug2_vec, test_cell_vec, y_test = get_feature_arrays(test_indices, synergy_df, drug_feature_dict, cell_feature_dict)

               
            # === Combine arrays ===
            X_train = np.concatenate([train_drug1_vec, train_drug2_vec, train_cell_vec], axis=1)
            X_val = np.concatenate([val_drug1_vec, val_drug2_vec, val_cell_vec], axis=1)
            X_test = np.concatenate([test_drug1_vec, test_drug2_vec, test_cell_vec], axis=1)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            input_dim = X_train.shape[1]
            model = LinearSynergyModel(input_dim=input_dim)
            fold_name = f"{args.split}_{fold_num}"

            save_path = os.path.join(l1_dir, f"results_{args.split}_{fold_num}.json")
            train_model(model, train_loader, val_loader, save_path, epochs=args.epochs, patience=100,  l1_lambda=l1_lambda)


            model.load_state_dict(torch.load(save_path.replace('.json', '_best.pt')))
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device)
                    output = model(X)
                    preds.extend(output.cpu().numpy())
                    truths.extend(y.numpy())
            mse_value = np.mean((np.array(preds) - np.array(truths)) ** 2)


            print(f"Test MSE for {fold_name}: {mse_value:.4f}")

            all_mse.append((fold_name, mse_value))

            # === Tahminleri Kaydet ===
            preds_array = np.array(preds)
            truths_array = np.array(truths)

            preds_df = pd.DataFrame({
                "True_Label": truths_array,
                "Predicted_Label": preds_array
            })

            preds_save_path = os.path.join(l1_dir, f"predictions_{fold_name}.csv")
            preds_df.to_csv(preds_save_path, index=False)
            print(f"Predictions saved to {preds_save_path}")


            drug_feature_names = drug_features_df.columns[1:].tolist()
            cell_feature_names = cell_features_df.columns[1:].tolist()

            random.seed(42)
            indices = random.sample(range(X_test.shape[0]), min(100, X_test.shape[0]))

            # Standardized Coefficients
            X_test_np = X_test.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            X_std = np.std(X_test_np, axis=0)
            y_std = np.std(y_test_np)

            weights = model.linear.weight.detach().cpu().numpy().flatten()

            standardized_coeffs = weights * (X_std / y_std)

            # Feature names; Drug1 + Drug2 + CellLine
            drug1_feat_names = [f"Drug1_{feat}" for feat in drug_feature_names]
            drug2_feat_names = [f"Drug2_{feat}" for feat in drug_feature_names]
            cell_feat_names = [f"CellLine_{feat}" for feat in cell_feature_names]
            all_feat_names = drug1_feat_names + drug2_feat_names + cell_feat_names

            coeff_df = pd.DataFrame({
                "Feature": all_feat_names,
                "Weight": weights,
                "Standardized_Coefficient": standardized_coeffs
            })

            # Absolute value
            coeff_df = coeff_df.sort_values(by="Standardized_Coefficient", key=np.abs, ascending=False)

            # save
            coeff_save_path = os.path.join(l1_dir, f"coefficients_{fold_name}.csv")
            coeff_df.to_csv(coeff_save_path, index=False)
            print(f"Standardized coefficients saved to {coeff_save_path}")

            nonzero = np.sum(np.abs(weights) > 1e-6)
            print(f"L1 {l1_lambda:.0e} | Fold {fold_name}: {nonzero} non-zero weights out of {len(weights)}")



        mse_file = os.path.join(l1_dir, "test_mse_results.txt")
        with open(mse_file, "w") as f:
            for fold_name, mse in all_mse:
                f.write(f"{fold_name}: MSE = {mse:.4f}\n")

            avg_mse = sum(m for _, m in all_mse) / len(all_mse)
            f.write(f"Average MSE: {avg_mse:.4f}\n")
                
                
        print(f"\nTest MSE results saved to {mse_file}")

