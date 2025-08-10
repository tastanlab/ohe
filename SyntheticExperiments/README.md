# Synthetic Data Experiments

## Purpose
The goal of these experiments is to evaluate whether machine learning models truly rely on informative features or instead exploit repetition patterns in the data.  
We simulate drug–drug–cell line synergy prediction tasks under controlled conditions where the ground truth is known.  
By comparing performance and feature recovery in datasets **with** and **without** entity repetition, we can measure how much generalization depends on genuine feature learning versus memorization.

## Dataset Types
1. **Non-repeated:** 24,500 unique (Drug₁, Drug₂, CellLine) triples — each entity appears only once.  
2. **Repeated:** Full cross of 50 drugs × 50 cell lines (61,250 triples) — entities repeat in different combinations.

## Experimental Setups

### Linear Model
- **Feature generation:**  
  - Each drug → 100-D vector from N(0, I)  
  - Each cell line → 100-D vector from N(0, I)   
- **Score generation:**  
  - For each entity, **20 features** randomly chosen as informative.  
  - Informative features get random weights from U[0.5, 10]; non-informative features get weight **0**. 
  - Synergy score = linear combination of entity features + Gaussian noise (σ = 0.1).  
  - True equation:  
    \[
    y = w_{d1}^\top x_{d1} + w_{d2}^\top x_{d2} + w_c^\top z_c + \varepsilon
    \]
- **Goal:** Check if a predictive linear model can recover the true informative features under repeated vs. non-repeated conditions.

### Nonlinear Model
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
  - Nonlinear model → Integrated Gradients (zero baseline, 100 steps)  
  - Compare top-k features to ground-truth informative set (Jaccard index).

