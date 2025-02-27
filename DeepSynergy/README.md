# DeepSynergy:  predicting anti-cancer drug synergy with Deep Learning

**DeepSynergy** is the one of the earliest deep neural network based approach used for predict **Loewe synergy score** for drug pair - cell line combinations.

- **Drug Features**: Chemical fingerprints.  
- **Cell Line Features**: Cell line gene expression profiles.

---

## DeepSynergy Architecture

The architecture of the DeepSynergy model is shown below:

![DeepSynergy Architecture](../figures/deepsynergy.png)  

*Figure 1: Architecture of DeepSynergy, inspired by Preuer et al. (2017)*

---

## Dataset and Splitting Information

DeepSynergy was trained on the **O'Neil** dataset using **Leave-Pair-Out (LPO)** splits and 5-fold cross-validation. The dataset, splits, and hyperparameters were provided by the authors.

---

### O'Neil Dataset

#### Combination Dataset
- **`labels.csv`**:  Available under the `DeepSynergy/` folder. Contains drug pair and cell line combination information with synergy scores.

#### Drug & Cell Line Features
- **Drug Features**:
  - Derived from SMILES, including:
    - **ECFP_6 fingerprints**: 1309 features.
    - **Physico-chemical properties**: 802 features.
    - **Binary toxicophore features**: 2276 features.


- **Cell Line Features**:
  - **Gene expression profiles**: 3984 genetic features.

- Preprocessed data file: `data_test_fold0_tanh.p.gz` (includes all drug and cell line features for DeepSynergy experiments).

#### One-Hot Encoded Features
- `one_hot_encoded/ohe_data_test_fold0.p`: Contains one-hot encoded representations for both drugs and cell lines.

> Note: You can generate `ohe_data_test_fold0.p` using the provided `generate_ohe_data.py` script under the `one_hot_encoded/`folder.


---
### Training DeepSynergy with Drug & Cell Line Features

```bash
python deepsynergy.py
```

### Training DeepSynergy with One-Hot Encoded Features

```bash
python one_hot_encoded/deepsynergy_ohe.py
```

## Reproducing Experiments

For further details on the DeepSynergy framework or dataset preparation, visit the official DeepSynergy [GitHub repository](https://github.com/KristinaPreuer/DeepSynergy/tree/master)

> **Important Note:** Ensure you provide the correct paths to the input files and directories.
