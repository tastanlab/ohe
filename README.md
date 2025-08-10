## **One-Hot News: Drug Synergy Models Shortcut Molecular Features**
## Overview
This repository contains the code and data supporting our study, "One-Hot News: Drug Synergy Models Shortcut Molecular Features". Our work reveals that drug synergy prediction models, instead of leveraging meaningful chemical or biological features, often learn shortcuts based on co-variation patterns in the dataset. By replacing rich molecular representations with simple one-hot encoded identifiers, we demonstrate that models can achieve comparable or even slightly improved performance—highlighting fundamental generalization issues in current deep learning approaches for drug synergy prediction.

---

### Splitting Methods
- **Leave-Triple-Out (LTO):** Random split.
- **Leave-Pair-Out (LPO):** Drug pairs do not appear in training, but individual drugs may.
- **Leave-CellLine-Out (LCO):** Cell lines do not appear in training.
- **Leave-One-Drug-Out (LODO):** One drug in the pair is never seen in training.
- **Leave-Drug-Out (LDO):** Neither drug in the pair appears in training.
![Split Methods](figures/split.jpg)
*Figure 1: Illustration of data split strategies, inspired by Preuer et al. (2017).*

---

## Results

---

### Comparison of Synergy Prediction Models 

To compare their ability to capture information from drug and cell line features that affect their synergy scores, we evaluated the following models:

- **DeepSynergy**
- **DeepDDS**
- **MatchMaker**
- **MARSY**
- **JointSyn**

## Approach
1. Reproduced each model with its **original hyperparameters**, **datasets**, and **split strategies**.
2. Replaced original drug/cell features with **one-hot encodings (OHE)** to test learning from entity identity.
3. Evaluated **four average-based baselines**:
   - **Overall Average:** predict the global mean for all test samples.
   - **Drug-Pair Average:** if the exact drug pair appears in training, use its mean; otherwise use the overall mean.
   - **Cell-Line Average:** if the test cell line appears in training, use its mean; otherwise use the overall mean.
   - **Cell-Line & ≥1-Drug Average:** if the test cell line and at least one of its drugs appear together in training, use that mean; otherwise use the overall mean.  
   *In the result table, only the best-performing variant among these is reported as **Best Average**.*
4. Included a **Shuffled Features** control: randomly permuted drug and cell-line feature vectors across samples, preserving marginal feature distributions but breaking entity-specific associations.
5. Considered **MoLFormer (pretrained drug embeddings)**: each drug encoded from its SMILES using the pretrained *MoLFormer-XL-both-10pct* model. Cell lines remain **one-hot**.
6. All settings were trained and evaluated under the same split protocols as the original works, with results reported using the original metrics.


![Model Performance Comparison](figures/OHE.pdf)
*Figure 2: Performance of models with original features, one-hot encoding (OHE), shuffled features, MoLFormer embeddings, and the Best Average baseline across datasets, split methods, and metrics.*

---

### Setting Up the Environment
To ensure reproducibility, install dependencies using Conda:
```
conda env create -f environment.yml
conda activate drug_synergy
```

---
We performed all experiments on systems equipped with Tesla V100-PCIE-32GB and Tesla V100S-PCIE-32GB GPUs.

---


## Training Instructions

Detailed instructions for training and reproducing the experiments are available in each model's respective folder. Follow the steps provided in the corresponding directories to set up and execute the models.
