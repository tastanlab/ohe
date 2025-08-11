## Non-Linear Experiments


### Dataset Generation
With `generate_dataset.py`, you can create the following datasets:

- **Non-repeated dataset**  
  - Generate feature vectors for **49,000 drugs** and **24,500 cell lines**.  
  - Create **24,500** unique (Drug1, Drug2, CellLine) triplets where no drug or cell line is repeated.

- **Repeated dataset**  
  - Fully cross a panel of **50 drugs** and **50 cell lines**.  
  - This yields **61,250** triplets covering all possible combinations.

- **Feature selection/masking**  
  - For score generation, randomly select **20 informative features** for each entity, mask the others to 0, and save these masked feature vectors.

### Non-linear Score Generation
Use `generate_score.py` to produce **non-linear** scores for either the **repeated** or **non-repeated** datasets you generated.

### Preprocessed Files
You can download all preprocessed files from:  
- **All experiments:**  
  https://huggingface.co/datasets/ebcandir/synthetic_experiments/tree/main/complex_experiment  
- **Non-repeated dataset:**  
  https://huggingface.co/datasets/ebcandir/synthetic_experiments/tree/main/complex_experiment/non_repeated

### Training / Evaluation (main.py)

**For repeated dataset:**
- `synergy_file`: `synergy_scores.csv`  
- `drug_feat_file`: `50_drug_features.csv`  
- `cell_feat_file`: `50_cell_line_features.csv`  
- All split files (LPO, LCO, LODO, LDO) are available at the above link.

**For non-repeated dataset:**
- `synergy_file`: `synergy_scores_unique.csv`  
- `drug_feat_file`: `49000_drug_features.csv`  
- `cell_feat_file`: `24500_cell_line_features.csv`  
- The `non_repeated_splits/` folder contains the generated splits for this dataset.

> **Note:** Ensure that file and folder names match the argument values you pass to `main.py`.
