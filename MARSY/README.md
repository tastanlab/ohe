# MARSY: A Multitask Deep Learning Framework for Drug Combination Synergy Prediction

**MARSY** is a multitask deep learning-based model designed to predict **ZIP synergy scores** for drug combinations and single drug responses for individual drugs.  

- **Drug Features**: Gene expression signatures measured after treating **MCF7** and **PC3** cell lines with drugs.  
- **Cell Line Features**: Cell line gene expression profiles.

---

## MARSY Architecture

The architecture of the MARSY model is shown below:

![MARSY Architecture](../figures/marsy.png)  

*Figure 1: Architecture of MARSY, inspired by El Khili et al. (2023)*

---

## Dataset and Splitting Information

MARSY was trained on the **DrugComb** dataset using **Leave-Pair-Out (LPO)** splits and 5-fold cross-validation. The dataset, splits, and hyperparameters were provided by the authors.

Split files are located in:  
> - `data/lpo_folds`

---

### DrugComb Dataset

#### Combination Dataset
- **`data.csv`**: Available under the `data/` folder. Contains drug pair and cell line combination information with synergy scores.

#### Drug & Cell Line Features
- **Drug Features**:
  - `PC3_drugs_gene_expression.csv`: Gene expression profiles for drugs treated on the PC3 cell line (vector length: 978).
  - `MCF7_drugs_gene_expression.csv`: Gene expression profiles for drugs treated on the MCF7 cell line (vector length: 978).
  - To prepare drug features, concatenate `Drug1_PC3`, `Drug1_MCF7`, `Drug2_PC3`, and `Drug2_MCF7` for each drug pair.

- **Cell Line Features**:
  - `75_cell_lines_gene_expression.csv`: Gene expression profiles for 75 cell lines, each with 4639 genes.

- **Final Feature Vector**:
  After concatenating all features, the final vector length for each instance is:  
  `978 (Drug1_PC3) + 978 (Drug1_MCF7) + 978 (Drug2_PC3) + 978 (Drug2_MCF7) + 4639 (Cell Line) = 8551`.

> **Note:** You can download [`feature_dataset.csv`](https://huggingface.co/datasets/ebcandir/MARSY/resolve/main/feature_dataset.csv) to use  as input file for training.
  
#### One-Hot Encoded Features
- **`ohe_dataset.csv`**: Contains one-hot encoded representations for both drugs and cell lines.

> **Note:** You can generate `ohe_dataset.csv` using the provided `generate_ohe_data.py` script or download it directly [here](https://huggingface.co/datasets/ebcandir/MARSY/resolve/main/ohe_dataset.csv) to use  as input file for training.

#### Shuffled Features
In this setting, both drug and cell line feature vectors are randomly permuted so that each drug or cell line is assigned the feature vector of another component.  
> **Note:** You can download [`shuffled_feature_dataset.csv`](https://huggingface.co/datasets/ebcandir/MARSY/resolve/main/shuffled_feature_dataset.csv) to use  as input file for training.

#### MoLFormer Embeddings
Drugs are represented using pretrained **MoLFormer embeddings** obtained from their SMILES strings, while cell lines are represented using one-hot encoding.  
> **Note:** You can download [`molformer_dataset.csv`](https://huggingface.co/datasets/ebcandir/MARSY/resolve/main/molformer_dataset.csv) to use  as input file for training.

---

### Training MARSY 

```bash
python MARSY.py
```

- Set the data variable to the path of the feature file you want to use.
Example:
```bash
data = "ohe_dataset.csv"
```

Assign `triple_length` and `pair_length` based on the selected feature type:

| Feature Type                                   | triple_length | pair_length |
|-----------------------------------------------|---------------|-------------|
| Drug & Cell Line Features / Shuffled Features | 8551          | 3912        |
| One-Hot Encoded (OHE) Features                 | 1415          | 1340        |
| MoLFormer Features                             | 1611          | 1536        |


## Reproducing Experiments

For further details on the MARSY framework or dataset preparation, visit the official MARSY [GitHub repository](https://github.com/Emad-COMBINE-lab/MARSY/tree/main/data)

> **Important Note:** Ensure you provide the correct paths to the input files and directories.
