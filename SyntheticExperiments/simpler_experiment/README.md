## Linear Experiments


### Dataset Generation
With `generate_dataset.py`, you can create the following datasets:

- **Non-repeated dataset**  
  - Generate feature vectors for **49,000 drugs** and **24,500 cell lines**.  
  - Create **24,500** unique (Drug1, Drug2, CellLine) triplets where no drug or cell line is repeated.

- **Repeated dataset**  
  - Fully cross a panel of **50 drugs** and **50 cell lines**.  
  - This yields **61,250** triplets covering all possible combinations.

- **Generate score**  
  - For score generation, randomly select **20 features** for each entity. Informative features get random weights from U[0.5, 10]; non-informative features get weight 0. 
  - Synergy score 
    $\ ((w_d \cdot x_{d1}) + (w_d \cdot x_{d2}) + 2 \times (w_c \cdot x_c) / 2 )+ \varepsilon$
    
    
    where;
    - $\ (x_{d1}, x_{d2}\)$ : feature vectors of drug 1 and drug 2. 
    - $\ (x_c\)$ : feature vector of the cell line.  
    - $\ (w_{d1}, w_{d2}, w_c\)$ : weight vectors corresponding to each entity.  
    - $\ \(\varepsilon\)$ : noise term. 



### Preprocessed Files
You can download all preprocessed files from:  
- **All experiments:**  
  https://huggingface.co/datasets/ebcandir/synthetic_experiments/tree/main/simple_experiment
- **Non-repeated dataset:**  
  https://huggingface.co/datasets/ebcandir/synthetic_experiments/tree/main/simple_experiment/non_repeated

### Training / Evaluation (main.py)

**For repeated dataset:**
- `synergy_file`: `synergy_scores.csv`  
- `drug_feat_file`: `50_drug_features.csv`  
- `cell_feat_file`: `50_cell_line_features.csv`  
- All split files (LPO, LCO, LODO, LDO) are available at the above link.

**For non-repeated dataset:**
- `synergy_file`: `ssynergy_scores_non_repeated.csv`  
- `drug_feat_file`: `49000_drug_features.csv`  
- `cell_feat_file`: `24500_cell_line_features.csv`  
- The `splits_non_repeated/` folder contains the generated splits for this dataset.

> **Note:** Ensure that file and folder names match the argument values you pass to `main.py`.
