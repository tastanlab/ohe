# Synthetic Data Experiments

## Purpose
The goal of these experiments is to evaluate whether machine learning models truly rely on informative features or instead exploit repetition patterns in the data.  
We simulate drug–drug–cell line synergy prediction tasks under controlled conditions where the ground truth is known.  
By comparing performance and feature recovery in datasets **with** and **without** entity repetition, we can measure how much generalization depends on genuine feature learning versus memorization.

## Dataset Types
1. **Non-repeated:** 24,500 unique (Drug₁, Drug₂, CellLine) triples — each entity appears only once.  
2. **Repeated:** Full cross of 50 drugs × 50 cell lines (61,250 triples) — entities repeat in different combinations.

## Experimental Setups

### Linear Model (Simpler Experiment)
- **Feature generation:**  
  - Each drug → 100-D vector from N(0, I)  
  - Each cell line → 100-D vector from N(0, I)   
- **Score generation:**  
  - For each entity, **20 features** randomly chosen as informative.  
  - Informative features get random weights from U[0.5, 10]; non-informative features get weight **0**. 
  - Synergy score = linear combination of entity features + Gaussian noise (σ = 0.1).  
  - True equation:
    $\ ((w_d \cdot x_{d1}) + (w_d \cdot x_{d2}) + 2 \times (w_c \cdot x_c) / 2 )+ \varepsilon$
    
    
    where;
    - $\ (x_{d1}, x_{d2}\)$ : feature vectors of drug 1 and drug 2. 
    - $\ (x_c\)$ : feature vector of the cell line.  
    - $\ (w_{d1}, w_{d2}, w_c\)$ : weight vectors corresponding to each entity.  
    - $\ \(\varepsilon\)$ : noise term. 

- **Goal:** Check if a predictive linear model can recover the true informative features under repeated vs. non-repeated conditions.

### Non-linear Model (Complex Experiment)
- **Feature generation:**  
  - Drug vectors: 150-D, Cell line vectors: 200-D (from N(0, I))  
- **Score generation:**  
  - 50 informative coordinates per entity; rest set to 0  
  - Pass masked entity vectors through a fixed 3-layer MLP (256 → 128 → 1, ReLU) with random weights  
  - Scale output by 200 to get synergy score  
- **Goal:** Measure how a predictive MLP recovers informative features in a more complex, nonlinear mapping.

## Evaluation
- **Performance metrics:** Mean Squared Error (MSE), Pearson correlation (PCC), Spearman correlation (SCC).  
- **Feature recovery:**  
  - Linear model → absolute standardized coefficients  
  - Non-linear model → Integrated Gradients (zero baseline, 100 steps)  
  - Compare top-k features to ground-truth informative set (Jaccard index).

## Results

**Feature recovery success** of the model for the linear and the nonlinear synthetic synergy model setups.  
Jaccard overlap between model-identified features and ground-truth informative features on the *repeated* and *non-repeated* datasets.  
Columns report Jaccard index for Drug 1, Drug 2, and Cell Line across L1 penalties (λ).  
For the linear block, top-20 features are taken from standardized absolute coefficients (averaged over 10 random splits).  
For the nonlinear block, top-50 features are taken from Integrated Gradients (IG).  
**NA** indicates “not applicable”: in the non-repeated dataset, no entity recurs, therefore LPO/LCO/LODO/LDO are undefined.

| Dataset        | Split Method | Drug1 (λ=0.1) | Drug2 (λ=0.1) | Cell Line (λ=0.1) | Drug1 (λ=0.01) | Drug2 (λ=0.01) | Cell Line (λ=0.01) | Drug1 (λ=0.001) | Drug2 (λ=0.001) | Cell Line (λ=0.001) |
|----------------|--------------|--------------|--------------|-------------------|----------------|----------------|--------------------|-----------------|-----------------|---------------------|
| **Linear Model** ||||||||||||
| **Repeated**   | LPO   | 0.91 | 0.91 | 0.48 | 1.00 | 1.00 | 0.48 | 0.43 | 0.43 | 0.43 |
|                | LCO   | 0.74 | 0.74 | 0.54 | 0.74 | 0.68 | 0.54 | 0.74 | 0.68 | 0.54 |
|                | LODO  | 0.54 | 0.54 | 0.48 | 0.54 | 0.54 | 0.48 | 0.54 | 0.54 | 0.48 |
|                | LDO   | 0.38 | 0.43 | 0.48 | 0.38 | 0.38 | 0.48 | 0.33 | 0.33 | 0.48 |
| **Non-repeating** | NA | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
|||||||||||||
| **Nonlinear Model** ||||||||||||
| **Repeated**   | LPO   | 0.21 | 0.24 | 0.19 | 0.30 | 0.30 | 0.43 | 0.32 | 0.33 | 0.41 |
|                | LCO   | 0.22 | 0.19 | 0.21 | 0.30 | 0.32 | 0.33 | 0.27 | 0.25 | 0.24 |
|                | LODO  | 0.27 | 0.21 | 0.16 | 0.28 | 0.28 | 0.27 | 0.32 | 0.22 | 0.22 |
|                | LDO   | 0.25 | 0.22 | 0.18 | 0.21 | 0.21 | 0.35 | 0.25 | 0.16 | 0.22 |
| **Non-repeating** | NA | 0.79 | 0.82 | 0.82 | 0.70 | 0.79 | 0.70 | 0.49 | 0.47 | 0.49 |

