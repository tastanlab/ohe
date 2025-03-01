# JointSyn: Dual-view jointly learning improves personalized drug synergy prediction

**JointSyn** is a recent multi-modal deep learning model that integrates various drug and cell line representations to predict drug synergy. The model employs a dual-view architecture, where drug combinations and cell lines are embedded separately before being passed to a prediction network.

- **Drug Features**: Molecular graphs, where atoms are nodes and bonds are edges.  
- **Cell Line Features**: Cell line gene expression profiles.
- **Drug Representations:**
  - **Molecular Graphs:** Nodes represent atoms, edges represent bonds, and each atom has a 78-dimensional atomic feature vector.
  - **Morgan Fingerprints:** Extended-connectivity fingerprints (ECFP6) with 1,309 features.
- **Cell Line Representation:** Cell line gene expression profiles.

---

## JointSyn Architecture

The architecture of the JointSyn model is shown below:

![JointSyn Architecture](../figures/jointsyn.png)  

*Figure 1: Architecture of JointSyn, inspired by Li et al. (2024)*

---

## Dataset and Splitting Information

JointSyn was trained on a **subset of the O’Neil dataset**, as detailed in its original publication. The dataset consists of:

- **12,033 drug combinations**
- **38 drugs**
- **34 cell lines**

For training, **Leave-Triple-Out (LTO) splitting** was applied, following the methodology provided by the authors. Each fold was repeated across **10 different replicates**, ensuring robustness in the evaluation.

> ⚠ **Note**: Due to compatibility issues with the required libraries and the CUDA version on our cluster, we were unable to train JointSyn using the original drug and cell line features. Instead, we report the results provided in the original JointSyn paper.

---

In one-hot encoded feature experiments, the graph-based components of the original architecture were removed. One-hot encoded features were directly provided as input, as shown in **Figure 2**:

![JointSyn Architecture 2](../figures/jointsyn_ohe.png)  

*Figure 2: Architecture of JointSyn for training with one hot encoded features*

---

### O'Neil Dataset

#### Combination Dataset
- **`data_to_split.csv`**:  Available under the `rawData/` folder. Contains drug pair and cell line combination information with synergy Loewe scores.

#### One-Hot Encoded Features
- **Drug Features**:
- `ohe_drug.csv`: Available under the `rawData/` folder. One-hot encoded representation for drugs.
- **Cell Line Features**:
- `ohe_cell.csv`: Available under the `rawData/` folder. One-hot encoded representation for cell lines.

---

### Training JointSyn with One-Hot Encoded Features

```bash
python main_ohe.py
```


## Reproducing Experiments

For further details on the JointSyn framework, original dataset or dataset preparation, visit the official JointSyn [GitHub repository](https://github.com/LiHongCSBLab/JointSyn)

> **Important Note:** Ensure you provide the correct paths to the input files and directories.
