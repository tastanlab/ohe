import os
import random
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# simple MLP model for synergy prediction
class MLPSynergyModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 128], dropout_rate=0.2):
        super(MLPSynergyModel, self).__init__()
        layers = []

        # input
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)



def train_model(model, train_loader, val_loader, save_path, epochs=1000, lr=1e-4, patience=100, l1_lambda=1e-3):
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

            # add L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation 
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

        # check for best
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), save_path.replace('.json', '_best.pt'))
            print(f"--> New best val_loss: {best_val_loss:.4f}, model saved.")
        else:
            patience_counter += 1
            print(f"--> EarlyStopping patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"--> Early stopping at epoch {epoch + 1}")
                break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path.replace('.json', '.pt'))
    with open(save_path, "w") as f:
        json.dump(results, f)


# Compute Integrated Gradients (IG)
def compute_ig_on_output(model, input_tensor, baseline=None, feature_names=None, save_path=None, n_steps=100):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    baseline = torch.zeros_like(input_tensor) if baseline is None else baseline.to(device)

    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=None,  # single scalar output IG on total prediction
        return_convergence_delta=True,
        n_steps=n_steps
    )

    avg_attr = attr.mean(dim=0).cpu().numpy()  # average across samples

    if feature_names and save_path:
        df = pd.DataFrame({"Feature": feature_names, "Avg_Attribution": avg_attr})
        df = df.sort_values(by="Avg_Attribution", key=np.abs, ascending=False)
        df.to_csv(save_path, index=False)
        print(f"IG attribution scores saved to {save_path}")
    return avg_attr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/complex_experiment/")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--split", type=str, default="lpo")
    parser.add_argument("--l1", type=float, default=1e-3)
    args = parser.parse_args()

    l1_str = f"l1_{args.l1:.0e}" # l1_1e-03

    # reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load synergy scores and features
    synergy_file = os.path.join(args.save_dir, "synergy_scores.csv")
    drug_feat_file = os.path.join(args.save_dir, "50_drug_features.csv")
    cell_feat_file = os.path.join(args.save_dir, "50_cell_line_features.csv")

    synergy_df = pd.read_csv(synergy_file)
    drug_features_df = pd.read_csv(drug_feat_file)
    cell_features_df = pd.read_csv(cell_feat_file)
    drug_feature_dict = drug_features_df.set_index('Drug').T.to_dict('list')
    cell_feature_dict = cell_features_df.set_index('CellLine').T.to_dict('list')

    l1_lambda = args.l1
    mse_list, pearson_list, spearman_list = [], [], []
    ig_all = []

    metrics_path = os.path.join(args.save_dir, args.split, f"{l1_str}_metrics.txt")
    ig_avg_path = os.path.join(args.save_dir, args.split, f"{l1_str}_ig_abs_avg.csv")

    with open(metrics_path, "w") as f_out:
        for fold in range(1, 11):
            fold_num = str(fold)
            fold_path = os.path.join(args.save_dir, args.split, l1_str)
            os.makedirs(fold_path, exist_ok=True)

            # load indices for this fold
            train_idx_file = f"/complex_experiment/splits_{args.split}/train_set_{args.split}_{fold_num}.txt"
            val_idx_file = f"/complex_experiment/splits_{args.split}/val_set_{args.split}_{fold_num}.txt"
            test_idx_file = f"/complex_experiment/splits_{args.split}/test_set_{args.split}_{fold_num}.txt"

            train_indices = np.loadtxt(train_idx_file, dtype=int)
            val_indices = np.loadtxt(val_idx_file, dtype=int)
            test_indices = np.loadtxt(test_idx_file, dtype=int)

            def get_feature_arrays(indices):
                drug1, drug2, cell, y = [], [], [], []
                for idx in indices:
                    row = synergy_df.iloc[idx]
                    drug1.append(drug_feature_dict[row['Drug_1']])
                    drug2.append(drug_feature_dict[row['Drug_2']])
                    cell.append(cell_feature_dict[row['CellLine']])
                    y.append(row['SynergyScore'])
                return np.array(drug1), np.array(drug2), np.array(cell), np.array(y)

            # build train, validation, and test tensors
            X_d1, X_d2, X_c, y = get_feature_arrays(train_indices)

            X_orig = np.concatenate([X_d1, X_d2, X_c], axis=1)
            X_reversed = np.concatenate([X_d2, X_d1, X_c], axis=1)

            X_train = np.concatenate([X_orig, X_reversed], axis=0)
            y_train = np.concatenate([y, y], axis=0)


            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)

            def make_tensor(indices):
                d1, d2, c, y = get_feature_arrays(indices)
                X = np.concatenate([d1, d2, c], axis=1)
                return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

            X_val, y_val = make_tensor(val_indices)
            X_test, y_test = make_tensor(test_indices)

            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

            # train the mode
            model = MLPSynergyModel(input_dim=X_train.shape[1])
            model_path = os.path.join(fold_path, f"mlp_model_{fold_num}.pt")
            if not os.path.exists(model_path):
                print("Training model...")
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)
                train_model(model, train_loader, val_loader,
                            save_path=os.path.join(fold_path, f"mlp_model_{fold_num}.json"),
                            epochs=args.epochs, l1_lambda=l1_lambda)

            # load best model and evaluate
            best_model_path = os.path.join(fold_path, f"mlp_model_{fold_num}_best.pt")
            model.load_state_dict(torch.load(best_model_path))
            model.to(device)
            model.eval()

            predictions, targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds = model(X_batch)
                    predictions.append(preds.cpu().numpy())
                    targets.append(y_batch.cpu().numpy())

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)

            # metrics
            mse = mean_squared_error(targets, predictions)
            pearson_corr, _ = pearsonr(targets, predictions)
            spearman_corr, _ = spearmanr(targets, predictions)

            f_out.write(f"Fold {fold} | MSE: {mse:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}\n")
            mse_list.append(mse)
            pearson_list.append(pearson_corr)
            spearman_list.append(spearman_corr)

            # run IG on 10% of test data 
            all_feat_names = [f"Drug1_f{i}" for i in range(X_d1.shape[1])] + \
                             [f"Drug2_f{i}" for i in range(X_d2.shape[1])] + \
                             [f"Cell_f{i}" for i in range(X_c.shape[1])]

            rng = np.random.default_rng(seed=42 + fold)
            test_size = int(len(X_test) * 0.1)
            selected = rng.choice(len(X_test), test_size, replace=False)
            input_tensor = X_test[selected]
            ig_output_path = os.path.join(fold_path, f"ig_output_fold{fold}.csv")
            ig_scores = compute_ig_on_output(
                model,
                input_tensor=input_tensor,
                feature_names=all_feat_names,
                save_path=ig_output_path,
                n_steps=100
            )
            ig_all.append(ig_scores)

        # average metrics
        f_out.write("\n=== Averages Across Folds ===\n")
        f_out.write(f"Avg MSE: {np.mean(mse_list):.4f}\n")
        f_out.write(f"Avg Pearson: {np.mean(pearson_list):.4f}\n")
        f_out.write(f"Avg Spearman: {np.mean(spearman_list):.4f}\n")

    # compute absolute and signed averages of IG scores across folds
    ig_all = np.array(ig_all)  # shape: (num_folds, num_features)

    # create df
    df_ig_avg = pd.DataFrame({
        "Feature": all_feat_names
    })

    # signed average
    df_ig_avg["Avg_Attribution_Signed"] = ig_all.mean(axis=0)
    # abs average
    df_ig_avg["Avg_Attribution_Abs"] = np.abs(ig_all).mean(axis=0)
    # Sort by absolute average importance
    df_ig_avg = df_ig_avg.sort_values(by="Avg_Attribution_Abs", ascending=False)
    df_ig_avg.to_csv(ig_avg_path, index=False)

