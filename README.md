# AI for Polymer Representation

This repository contains a comprehensive analysis of polymer representations and clustering using machine learning techniques. The analysis is implemented in the Jupyter notebook `Polymer_Representation.ipynb`.

## Overview

The project implements multiple polymer representation methods and evaluates their performance in unsupervised clustering and supervised classification tasks:

### Representations
- **Morgan Fingerprints (ECFP)**: Circular fingerprints capturing local atom environments
- **MACCS Keys**: 166 predefined structural keys for substructure patterns
- **RDKit Descriptors**: Physicochemical descriptors combined with MACCS motifs
- **Transformer Embeddings**: Pretrained polyBERT embeddings for SMILES sequences

### Clustering Methods
- **K-means**: Applied to transformer embeddings with elbow and silhouette analysis
- **Butina Clustering**: Tanimoto-distance based clustering for Morgan FP and MACCS keys
- **Hierarchical Clustering**: Ward linkage on Tanimoto distances

### Validation
- Clustering quality metrics (ARI, NMI, Silhouette) against ground truth polymer classes
- Supervised baselines using Random Forest, Logistic Regression, and KNN with 5-fold CV
- Dimensionality reduction visualization with UMAP

## Dataset

The analysis uses the PI1070 polymer dataset, which is cleaned and canonicalized using RDKit. The dataset includes polymer SMILES strings and associated properties.

## Results

Key outputs include:
- Cluster assignments for all methods
- Validation metrics comparing clustering quality
- Supervised classification performance across representations
- UMAP visualizations of chemical space
- Representative polymer structures per cluster
- MACCS key frequency analysis

## Files

- `Polymer_Representation (2).ipynb`: Main analysis notebook
- `PI1070_cleaned_with_clusters.csv`: Processed dataset with cluster labels
- `supervised_cv_results.csv`: Supervised classification results
- `plots/`: Directory containing all generated plots and visualizations
- `*.npy`: Numpy arrays of computed representations and UMAP embeddings

## Dependencies

- rdkit
- umap-learn
- scikit-learn
- pandas
- numpy
- matplotlib
- plotly
- transformers
- torch

## Usage

1. Install dependencies: `pip install -r requirements.txt` (if available) or run the installation cell in the notebook
2. Execute the notebook cells in order
3. Results will be saved to the `plots/` directory and as CSV/numpy files

