#!/usr/bin/env python3
"""
Script to update Polymer_Representation.ipynb with new requirements:
1. Save input representations to disk with mapping table
2. Replace Butina with Murtagh hierarchical clustering (K=2-25)
3. Implement ARI/NMI evaluation for K=2-25
4. Replace 5-fold CV with 5x 70:30 stratified splits
5. Evaluate on both train and test sets
6. Create UMAP plots for K values (5,10,15,25)
7. Generate cluster vs polymer_class contingency heatmaps
"""

import json
import sys

def create_updated_notebook():
    """Create the updated notebook with all required changes."""

    # Read the original notebook
    with open('Polymer_Representation.ipynb', 'r') as f:
        nb = json.load(f)

    # We'll keep the existing cells up to and including the representation computation
    # Then add new cells for the updated analysis

    # Find the index where we should start making changes
    # We'll keep everything up to Section 3 (clustering) and replace from there

    new_cells = []

    # Keep cells 0-15 (up to end of representation computation)
    # This includes: imports, data loading, cleaning, and representation computation
    for i in range(min(16, len(nb['cells']))):
        new_cells.append(nb['cells'][i])

    # Add new section for saving representations
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Save Input Representations to Disk\n",
            "\n",
            "Save all computed representations in a reusable format with a mapping table for alignment."
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pickle\n",
            "from pathlib import Path\n",
            "\n",
            "# Create directory for representations\n",
            "repr_dir = Path('representations')\n",
            "repr_dir.mkdir(exist_ok=True)\n",
            "\n",
            "# Save each representation\n",
            "print(\"Saving representations to disk...\")\n",
            "\n",
            "# 1. Morgan fingerprints\n",
            "np.savez_compressed(repr_dir / 'morgan_fp.npz', \n",
            "                   data=morgan_fp,\n",
            "                   shape=morgan_fp.shape)\n",
            "print(f\"✓ Morgan fingerprints saved: {morgan_fp.shape}\")\n",
            "\n",
            "# 2. MACCS keys\n",
            "np.savez_compressed(repr_dir / 'maccs_keys.npz',\n",
            "                   data=maccs_keys,\n",
            "                   shape=maccs_keys.shape)\n",
            "print(f\"✓ MACCS keys saved: {maccs_keys.shape}\")\n",
            "\n",
            "# 3. RDKit descriptors + MACCS\n",
            "np.savez_compressed(repr_dir / 'rdkit_maccs.npz',\n",
            "                   data=rdkit_maccs_scaled,\n",
            "                   shape=rdkit_maccs_scaled.shape)\n",
            "print(f\"✓ RDKit+MACCS saved: {rdkit_maccs_scaled.shape}\")\n",
            "\n",
            "# 4. Transformer embeddings\n",
            "np.savez_compressed(repr_dir / 'transformer_emb.npz',\n",
            "                   data=transformer_scaled,\n",
            "                   shape=transformer_scaled.shape)\n",
            "print(f\"✓ Transformer embeddings saved: {transformer_scaled.shape}\")\n",
            "\n",
            "# 5. Tanimoto distance matrix for Morgan fingerprints (for clustering)\n",
            "from rdkit import DataStructs\n",
            "from rdkit.Chem import AllChem\n",
            "\n",
            "def compute_tanimoto_distance_matrix(fps):\n",
            "    \"\"\"Compute pairwise Tanimoto distance matrix.\"\"\"\n",
            "    n = len(fps)\n",
            "    dist_matrix = np.zeros((n, n))\n",
            "    \n",
            "    for i in range(n):\n",
            "        # Compute similarities for row i\n",
            "        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)\n",
            "        # Convert to distances\n",
            "        dist_matrix[i, :] = 1.0 - np.array(sims)\n",
            "    \n",
            "    return dist_matrix\n",
            "\n",
            "print(\"\\nComputing Tanimoto distance matrix for Morgan fingerprints...\")\n",
            "morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in df['mol']]\n",
            "tanimoto_dist = compute_tanimoto_distance_matrix(morgan_fps)\n",
            "np.savez_compressed(repr_dir / 'tanimoto_distance_morgan.npz',\n",
            "                   data=tanimoto_dist,\n",
            "                   shape=tanimoto_dist.shape)\n",
            "print(f\"✓ Tanimoto distance matrix saved: {tanimoto_dist.shape}\")\n",
            "\n",
            "# 6. Save mapping table\n",
            "mapping_df = df[['smiles_canonical', 'polymer_class']].copy()\n",
            "mapping_df['polymer_id'] = range(len(mapping_df))\n",
            "mapping_df['row_index'] = range(len(mapping_df))\n",
            "mapping_df = mapping_df[['polymer_id', 'smiles_canonical', 'polymer_class', 'row_index']]\n",
            "mapping_df.to_csv(repr_dir / 'mapping_table.csv', index=False)\n",
            "print(f\"\\n✓ Mapping table saved: {len(mapping_df)} polymers\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"All representations saved successfully!\")\n",
            "print(\"=\"*60)"
        ]
    })

    # Add new section for Murtagh hierarchical clustering
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Murtagh Hierarchical Clustering (K = 2 to 25)\n",
            "\n",
            "Implement hierarchical clustering using Murtagh-style linkage on Tanimoto distances.\n",
            "We'll test K values from 2 to 25 and evaluate cluster quality."
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scipy.cluster.hierarchy import linkage, fcluster, dendrogram\n",
            "from scipy.spatial.distance import squareform\n",
            "from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Ensure plots directory exists\n",
            "Path('plots').mkdir(exist_ok=True)\n",
            "\n",
            "def murtagh_hierarchical_clustering(distance_matrix, k_values, method='average'):\n",
            "    \"\"\"\n",
            "    Perform Murtagh hierarchical clustering for multiple K values.\n",
            "    \n",
            "    Parameters:\n",
            "    -----------\n",
            "    distance_matrix : ndarray\n",
            "        Square distance matrix\n",
            "    k_values : list\n",
            "        List of K values to test\n",
            "    method : str\n",
            "        Linkage method ('average', 'complete', 'ward', 'single')\n",
            "        \n",
            "    Returns:\n",
            "    --------\n",
            "    dict with linkage matrix and cluster assignments for each K\n",
            "    \"\"\"\n",
            "    # Convert distance matrix to condensed form for scipy\n",
            "    condensed_dist = squareform(distance_matrix, checks=False)\n",
            "    \n",
            "    # Perform hierarchical clustering\n",
            "    print(f\"Performing {method} linkage hierarchical clustering...\")\n",
            "    linkage_matrix = linkage(condensed_dist, method=method)\n",
            "    \n",
            "    # Get cluster assignments for each K\n",
            "    results = {\n",
            "        'linkage_matrix': linkage_matrix,\n",
            "        'clusters': {},\n",
            "        'metrics': []\n",
            "    }\n",
            "    \n",
            "    for k in k_values:\n",
            "        clusters = fcluster(linkage_matrix, k, criterion='maxclust')\n",
            "        results['clusters'][k] = clusters\n",
            "    \n",
            "    return results\n",
            "\n",
            "# Define K values to test (2 to 25)\n",
            "k_values = list(range(2, 26))\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"MURTAGH HIERARCHICAL CLUSTERING ON TANIMOTO DISTANCE\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Perform clustering\n",
            "murtagh_results = murtagh_hierarchical_clustering(\n",
            "    tanimoto_dist, \n",
            "    k_values,\n",
            "    method='average'  # Murtagh-style average linkage\n",
            ")\n",
            "\n",
            "print(f\"✓ Clustering completed for K = {k_values[0]} to {k_values[-1]}\")\n",
            "print(f\"✓ Total cluster assignments computed: {len(k_values)}\")"
        ]
    })

    # Add evaluation of clustering
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Evaluate Clustering Quality (ARI, NMI, Silhouette)\n",
            "\n",
            "For each K value, compute:\n",
            "- **ARI (Adjusted Rand Index)**: Measures agreement with true polymer classes\n",
            "- **NMI (Normalized Mutual Information)**: Measures shared information with true classes\n",
            "- **Silhouette Score**: Internal cluster quality metric"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate clustering for each K\n",
            "print(\"\\nEvaluating clustering quality for each K...\\n\")\n",
            "\n",
            "for k in k_values:\n",
            "    clusters = murtagh_results['clusters'][k]\n",
            "    \n",
            "    # Compute metrics\n",
            "    sil_score = silhouette_score(tanimoto_dist, clusters, metric='precomputed')\n",
            "    \n",
            "    # If polymer_class is available, compute ARI and NMI\n",
            "    if 'polymer_class' in df.columns and df['polymer_class'].notna().any():\n",
            "        true_labels = df['polymer_class'].values\n",
            "        ari = adjusted_rand_score(true_labels, clusters)\n",
            "        nmi = normalized_mutual_info_score(true_labels, clusters)\n",
            "    else:\n",
            "        ari = None\n",
            "        nmi = None\n",
            "    \n",
            "    # Store results\n",
            "    murtagh_results['metrics'].append({\n",
            "        'K': k,\n",
            "        'n_clusters': len(np.unique(clusters)),\n",
            "        'silhouette': sil_score,\n",
            "        'ARI': ari,\n",
            "        'NMI': nmi\n",
            "    })\n",
            "\n",
            "# Convert to DataFrame for easy viewing\n",
            "metrics_df = pd.DataFrame(murtagh_results['metrics'])\n",
            "\n",
            "print(\"Clustering Quality Metrics:\")\n",
            "print(\"=\"*70)\n",
            "print(metrics_df.to_string(index=False))\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Save metrics to CSV\n",
            "metrics_df.to_csv('plots/clustering_metrics_k2_to_k25.csv', index=False)\n",
            "print(\"\\n✓ Metrics saved to: plots/clustering_metrics_k2_to_k25.csv\")"
        ]
    })

    # Add visualization of clustering metrics
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot ARI and NMI vs K\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "\n",
            "# Plot 1: ARI vs K\n",
            "axes[0].plot(metrics_df['K'], metrics_df['ARI'], 'o-', linewidth=2, markersize=8, color='#2E86AB')\n",
            "best_k_ari = metrics_df.loc[metrics_df['ARI'].idxmax(), 'K']\n",
            "axes[0].axvline(best_k_ari, color='red', linestyle='--', alpha=0.7, label=f'Best K = {int(best_k_ari)}')\n",
            "axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')\n",
            "axes[0].set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')\n",
            "axes[0].set_title('ARI vs Number of Clusters', fontsize=14, fontweight='bold')\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "axes[0].legend()\n",
            "axes[0].set_xticks(range(2, 26, 2))\n",
            "\n",
            "# Plot 2: NMI vs K\n",
            "axes[1].plot(metrics_df['K'], metrics_df['NMI'], 'o-', linewidth=2, markersize=8, color='#A23B72')\n",
            "best_k_nmi = metrics_df.loc[metrics_df['NMI'].idxmax(), 'K']\n",
            "axes[1].axvline(best_k_nmi, color='red', linestyle='--', alpha=0.7, label=f'Best K = {int(best_k_nmi)}')\n",
            "axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')\n",
            "axes[1].set_ylabel('Normalized Mutual Information (NMI)', fontsize=12, fontweight='bold')\n",
            "axes[1].set_title('NMI vs Number of Clusters', fontsize=14, fontweight='bold')\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "axes[1].legend()\n",
            "axes[1].set_xticks(range(2, 26, 2))\n",
            "\n",
            "# Plot 3: Silhouette Score vs K\n",
            "axes[2].plot(metrics_df['K'], metrics_df['silhouette'], 'o-', linewidth=2, markersize=8, color='#F18F01')\n",
            "best_k_sil = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'K']\n",
            "axes[2].axvline(best_k_sil, color='red', linestyle='--', alpha=0.7, label=f'Best K = {int(best_k_sil)}')\n",
            "axes[2].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')\n",
            "axes[2].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')\n",
            "axes[2].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')\n",
            "axes[2].grid(True, alpha=0.3)\n",
            "axes[2].legend()\n",
            "axes[2].set_xticks(range(2, 26, 2))\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('plots/01_clustering_metrics_vs_k.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n✓ Plot saved: plots/01_clustering_metrics_vs_k.png\")\n",
            "print(f\"\\nBest K values:\")\n",
            "print(f\"  - ARI: K = {int(best_k_ari)} (ARI = {metrics_df.loc[metrics_df['ARI'].idxmax(), 'ARI']:.4f})\")\n",
            "print(f\"  - NMI: K = {int(best_k_nmi)} (NMI = {metrics_df.loc[metrics_df['NMI'].idxmax(), 'NMI']:.4f})\")\n",
            "print(f\"  - Silhouette: K = {int(best_k_sil)} (Score = {metrics_df.loc[metrics_df['silhouette'].idxmax(), 'silhouette']:.4f})\")"
        ]
    })

    # Add dendrogram visualization
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot dendrogram\n",
            "plt.figure(figsize=(20, 8))\n",
            "dendrogram(\n",
            "    murtagh_results['linkage_matrix'],\n",
            "    no_labels=True,\n",
            "    color_threshold=0,\n",
            "    above_threshold_color='#2E86AB'\n",
            ")\n",
            "plt.title('Hierarchical Clustering Dendrogram (Murtagh Average Linkage)', \n",
            "          fontsize=16, fontweight='bold', pad=20)\n",
            "plt.xlabel('Polymer Index', fontsize=14, fontweight='bold')\n",
            "plt.ylabel('Tanimoto Distance', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('plots/02_dendrogram_murtagh.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(\"✓ Dendrogram saved: plots/02_dendrogram_murtagh.png\")"
        ]
    })

    # Add section for supervised learning with 70:30 split
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Supervised Learning: 5 Repeats of 70:30 Train-Test Split\n",
            "\n",
            "For each representation (MACCS, Transformer, Morgan, RDKit+MACCS), we'll:\n",
            "1. Perform 5 independent 70:30 stratified train-test splits\n",
            "2. Train Logistic Regression classifier\n",
            "3. Evaluate on **both training and test sets**\n",
            "4. Report mean ± std for Accuracy and F1-score"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "def supervised_evaluation_70_30(X, y, representation_name, n_repeats=5):\n",
            "    \"\"\"\n",
            "    Perform supervised evaluation with 5 repeats of 70:30 train-test split.\n",
            "    \n",
            "    Parameters:\n",
            "    -----------\n",
            "    X : ndarray\n",
            "        Feature matrix\n",
            "    y : array\n",
            "        Target labels\n",
            "    representation_name : str\n",
            "        Name of the representation\n",
            "    n_repeats : int\n",
            "        Number of train-test splits to perform\n",
            "        \n",
            "    Returns:\n",
            "    --------\n",
            "    dict with results for each split\n",
            "    \"\"\"\n",
            "    results = []\n",
            "    \n",
            "    print(f\"\\n{'='*70}\")\n",
            "    print(f\"Representation: {representation_name}\")\n",
            "    print(f\"{'='*70}\")\n",
            "    \n",
            "    for i in range(n_repeats):\n",
            "        # Stratified 70:30 split\n",
            "        X_train, X_test, y_train, y_test = train_test_split(\n",
            "            X, y, \n",
            "            test_size=0.30, \n",
            "            stratify=y, \n",
            "            random_state=42 + i  # Different seed for each repeat\n",
            "        )\n",
            "        \n",
            "        # Train Logistic Regression\n",
            "        clf = LogisticRegression(max_iter=1000, random_state=42)\n",
            "        clf.fit(X_train, y_train)\n",
            "        \n",
            "        # Predict on both train and test\n",
            "        y_train_pred = clf.predict(X_train)\n",
            "        y_test_pred = clf.predict(X_test)\n",
            "        \n",
            "        # Compute metrics\n",
            "        train_acc = accuracy_score(y_train, y_train_pred)\n",
            "        test_acc = accuracy_score(y_test, y_test_pred)\n",
            "        train_f1 = f1_score(y_train, y_train_pred, average='macro')\n",
            "        test_f1 = f1_score(y_test, y_test_pred, average='macro')\n",
            "        \n",
            "        results.append({\n",
            "            'split': i + 1,\n",
            "            'train_size': len(X_train),\n",
            "            'test_size': len(X_test),\n",
            "            'train_accuracy': train_acc,\n",
            "            'test_accuracy': test_acc,\n",
            "            'train_f1_macro': train_f1,\n",
            "            'test_f1_macro': test_f1\n",
            "        })\n",
            "        \n",
            "        print(f\"\\nSplit {i+1}:\")\n",
            "        print(f\"  Train: {len(X_train)} samples | Test: {len(X_test)} samples\")\n",
            "        print(f\"  Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}\")\n",
            "        print(f\"  Train F1 (macro): {train_f1:.4f} | Test F1 (macro): {test_f1:.4f}\")\n",
            "    \n",
            "    # Compute summary statistics\n",
            "    results_df = pd.DataFrame(results)\n",
            "    summary = {\n",
            "        'representation': representation_name,\n",
            "        'train_acc_mean': results_df['train_accuracy'].mean(),\n",
            "        'train_acc_std': results_df['train_accuracy'].std(),\n",
            "        'test_acc_mean': results_df['test_accuracy'].mean(),\n",
            "        'test_acc_std': results_df['test_accuracy'].std(),\n",
            "        'train_f1_mean': results_df['train_f1_macro'].mean(),\n",
            "        'train_f1_std': results_df['train_f1_macro'].std(),\n",
            "        'test_f1_mean': results_df['test_f1_macro'].mean(),\n",
            "        'test_f1_std': results_df['test_f1_macro'].std()\n",
            "    }\n",
            "    \n",
            "    print(f\"\\n{'='*70}\")\n",
            "    print(f\"Summary for {representation_name}:\")\n",
            "    print(f\"  Train Accuracy: {summary['train_acc_mean']:.4f} ± {summary['train_acc_std']:.4f}\")\n",
            "    print(f\"  Test Accuracy:  {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}\")\n",
            "    print(f\"  Train F1 (macro): {summary['train_f1_mean']:.4f} ± {summary['train_f1_std']:.4f}\")\n",
            "    print(f\"  Test F1 (macro):  {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}\")\n",
            "    print(f\"{'='*70}\")\n",
            "    \n",
            "    return {'details': results_df, 'summary': summary}\n",
            "\n",
            "# Prepare representations for supervised learning\n",
            "representations = {\n",
            "    'Morgan FP': morgan_fp,\n",
            "    'MACCS Keys': maccs_keys,\n",
            "    'RDKit+MACCS': rdkit_maccs_scaled,\n",
            "    'Transformer': transformer_scaled\n",
            "}\n",
            "\n",
            "# Get true labels\n",
            "y_true = df['polymer_class'].values\n",
            "\n",
            "# Store all results\n",
            "all_supervised_results = {}\n",
            "summary_list = []\n",
            "\n",
            "print(\"\\n\" + \"#\"*70)\n",
            "print(\"SUPERVISED LEARNING: 5 REPEATS OF 70:30 STRATIFIED SPLIT\")\n",
            "print(\"#\"*70)\n",
            "\n",
            "for repr_name, X in representations.items():\n",
            "    results = supervised_evaluation_70_30(X, y_true, repr_name, n_repeats=5)\n",
            "    all_supervised_results[repr_name] = results\n",
            "    summary_list.append(results['summary'])\n",
            "\n",
            "# Create summary DataFrame\n",
            "summary_df = pd.DataFrame(summary_list)\n",
            "summary_df.to_csv('plots/supervised_results_70_30_split.csv', index=False)\n",
            "print(\"\\n✓ Results saved: plots/supervised_results_70_30_split.csv\")"
        ]
    })

    # Add visualization of supervised results
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot supervised learning results\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "representations_order = ['Morgan FP', 'MACCS Keys', 'RDKit+MACCS', 'Transformer']\n",
            "summary_df_sorted = summary_df.set_index('representation').loc[representations_order].reset_index()\n",
            "\n",
            "x_pos = np.arange(len(representations_order))\n",
            "width = 0.35\n",
            "\n",
            "# Plot 1: Accuracy comparison\n",
            "train_acc = summary_df_sorted['train_acc_mean'].values\n",
            "train_acc_std = summary_df_sorted['train_acc_std'].values\n",
            "test_acc = summary_df_sorted['test_acc_mean'].values\n",
            "test_acc_std = summary_df_sorted['test_acc_std'].values\n",
            "\n",
            "axes[0].bar(x_pos - width/2, train_acc, width, yerr=train_acc_std,\n",
            "           label='Train', capsize=5, color='#2E86AB', alpha=0.8)\n",
            "axes[0].bar(x_pos + width/2, test_acc, width, yerr=test_acc_std,\n",
            "           label='Test', capsize=5, color='#F18F01', alpha=0.8)\n",
            "\n",
            "axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')\n",
            "axes[0].set_title('Classification Accuracy by Representation', fontsize=14, fontweight='bold')\n",
            "axes[0].set_xticks(x_pos)\n",
            "axes[0].set_xticklabels(representations_order, rotation=15, ha='right')\n",
            "axes[0].legend()\n",
            "axes[0].grid(True, alpha=0.3, axis='y')\n",
            "axes[0].set_ylim([0, 1.0])\n",
            "\n",
            "# Plot 2: F1-score comparison\n",
            "train_f1 = summary_df_sorted['train_f1_mean'].values\n",
            "train_f1_std = summary_df_sorted['train_f1_std'].values\n",
            "test_f1 = summary_df_sorted['test_f1_mean'].values\n",
            "test_f1_std = summary_df_sorted['test_f1_std'].values\n",
            "\n",
            "axes[1].bar(x_pos - width/2, train_f1, width, yerr=train_f1_std,\n",
            "           label='Train', capsize=5, color='#2E86AB', alpha=0.8)\n",
            "axes[1].bar(x_pos + width/2, test_f1, width, yerr=test_f1_std,\n",
            "           label='Test', capsize=5, color='#F18F01', alpha=0.8)\n",
            "\n",
            "axes[1].set_ylabel('F1-Score (Macro)', fontsize=12, fontweight='bold')\n",
            "axes[1].set_title('F1-Score by Representation', fontsize=14, fontweight='bold')\n",
            "axes[1].set_xticks(x_pos)\n",
            "axes[1].set_xticklabels(representations_order, rotation=15, ha='right')\n",
            "axes[1].legend()\n",
            "axes[1].grid(True, alpha=0.3, axis='y')\n",
            "axes[1].set_ylim([0, 1.0])\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('plots/03_supervised_learning_results.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(\"✓ Plot saved: plots/03_supervised_learning_results.png\")"
        ]
    })

    # Add UMAP visualization for selected K values
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. UMAP Visualization for Selected K Values\n",
            "\n",
            "Create UMAP visualizations colored by cluster assignments for K = 5, 10, 15, 25"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import umap\n",
            "\n",
            "# Compute UMAP embeddings for all representations\n",
            "print(\"Computing UMAP embeddings...\\n\")\n",
            "\n",
            "umap_embeddings = {}\n",
            "\n",
            "# Morgan FP (Jaccard distance)\n",
            "print(\"  - Morgan FP...\")\n",
            "umap_embeddings['Morgan FP'] = umap.UMAP(\n",
            "    n_neighbors=25, min_dist=0.2, metric='jaccard', random_state=42\n",
            ").fit_transform(morgan_fp)\n",
            "\n",
            "# MACCS Keys (Jaccard distance)\n",
            "print(\"  - MACCS Keys...\")\n",
            "umap_embeddings['MACCS Keys'] = umap.UMAP(\n",
            "    n_neighbors=25, min_dist=0.2, metric='jaccard', random_state=42\n",
            ").fit_transform(maccs_keys)\n",
            "\n",
            "# RDKit+MACCS (Euclidean distance)\n",
            "print(\"  - RDKit+MACCS...\")\n",
            "umap_embeddings['RDKit+MACCS'] = umap.UMAP(\n",
            "    n_neighbors=25, min_dist=0.2, metric='euclidean', random_state=42\n",
            ").fit_transform(rdkit_maccs_scaled)\n",
            "\n",
            "# Transformer (Euclidean distance)\n",
            "print(\"  - Transformer...\")\n",
            "umap_embeddings['Transformer'] = umap.UMAP(\n",
            "    n_neighbors=25, min_dist=0.2, metric='euclidean', random_state=42\n",
            ").fit_transform(transformer_scaled)\n",
            "\n",
            "print(\"\\n✓ UMAP embeddings computed for all representations\")"
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot UMAP for selected K values (5, 10, 15, 25)\n",
            "selected_k_values = [5, 10, 15, 25]\n",
            "\n",
            "for k in selected_k_values:\n",
            "    cluster_labels = murtagh_results['clusters'][k]\n",
            "    \n",
            "    fig, axes = plt.subplots(2, 2, figsize=(16, 14))\n",
            "    axes = axes.ravel()\n",
            "    \n",
            "    for idx, (repr_name, umap_coords) in enumerate(umap_embeddings.items()):\n",
            "        scatter = axes[idx].scatter(\n",
            "            umap_coords[:, 0], \n",
            "            umap_coords[:, 1],\n",
            "            c=cluster_labels,\n",
            "            cmap='tab20',\n",
            "            s=20,\n",
            "            alpha=0.7\n",
            "        )\n",
            "        \n",
            "        axes[idx].set_title(f'{repr_name} (K={k})', fontsize=14, fontweight='bold')\n",
            "        axes[idx].set_xlabel('UMAP 1', fontsize=11)\n",
            "        axes[idx].set_ylabel('UMAP 2', fontsize=11)\n",
            "        \n",
            "        # Add colorbar\n",
            "        cbar = plt.colorbar(scatter, ax=axes[idx])\n",
            "        cbar.set_label('Cluster', fontsize=10)\n",
            "    \n",
            "    plt.suptitle(f'UMAP Visualization Colored by Clusters (K={k})', \n",
            "                 fontsize=16, fontweight='bold', y=1.00)\n",
            "    plt.tight_layout()\n",
            "    plt.savefig(f'plots/04_umap_clusters_k{k}.png', dpi=300, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    \n",
            "    print(f\"✓ Saved: plots/04_umap_clusters_k{k}.png\")"
        ]
    })

    # Add contingency heatmap
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Cluster vs Polymer Class Contingency Heatmaps\n",
            "\n",
            "Create heatmaps showing the relationship between cluster assignments and true polymer classes for selected K values."
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import confusion_matrix\n",
            "\n",
            "# Create contingency heatmaps for K = 5, 10, 15, 25\n",
            "selected_k_for_heatmap = [5, 10, 15, 25]\n",
            "\n",
            "for k in selected_k_for_heatmap:\n",
            "    cluster_labels = murtagh_results['clusters'][k]\n",
            "    true_labels = df['polymer_class'].values\n",
            "    \n",
            "    # Create contingency table\n",
            "    contingency = pd.crosstab(\n",
            "        cluster_labels, \n",
            "        true_labels, \n",
            "        rownames=['Cluster'], \n",
            "        colnames=['Polymer Class']\n",
            "    )\n",
            "    \n",
            "    # Plot heatmap\n",
            "    plt.figure(figsize=(12, 8))\n",
            "    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', \n",
            "                cbar_kws={'label': 'Count'})\n",
            "    plt.title(f'Cluster vs Polymer Class Contingency (K={k})\\n' + \n",
            "              f'ARI = {metrics_df[metrics_df[\"K\"] == k][\"ARI\"].values[0]:.4f}, ' +\n",
            "              f'NMI = {metrics_df[metrics_df[\"K\"] == k][\"NMI\"].values[0]:.4f}',\n",
            "              fontsize=14, fontweight='bold', pad=15)\n",
            "    plt.xlabel('True Polymer Class', fontsize=12, fontweight='bold')\n",
            "    plt.ylabel('Cluster Assignment', fontsize=12, fontweight='bold')\n",
            "    plt.tight_layout()\n",
            "    plt.savefig(f'plots/05_contingency_heatmap_k{k}.png', dpi=300, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    \n",
            "    print(f\"✓ Saved: plots/05_contingency_heatmap_k{k}.png\")\n",
            "    \n",
            "    # Print purity analysis\n",
            "    print(f\"\\nCluster Purity Analysis for K={k}:\")\n",
            "    print(\"=\"*60)\n",
            "    for cluster_id in sorted(contingency.index):\n",
            "        cluster_counts = contingency.loc[cluster_id]\n",
            "        majority_class = cluster_counts.idxmax()\n",
            "        majority_count = cluster_counts.max()\n",
            "        total_count = cluster_counts.sum()\n",
            "        purity = majority_count / total_count\n",
            "        print(f\"Cluster {cluster_id}: {total_count:3d} polymers, \"\n",
            "              f\"Majority = {majority_class} ({majority_count}/{total_count} = {purity:.2%})\")\n",
            "    print(\"=\"*60 + \"\\n\")"
        ]
    })

    # Add summary section
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Summary and Key Findings\n",
            "\n",
            "Generate a comprehensive summary of all analyses performed."
        ]
    })

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"\\n\" + \"#\"*80)\n",
            "print(\"#\" + \" \"*78 + \"#\")\n",
            "print(\"#\" + \" \"*20 + \"POLYMER REPRESENTATION ANALYSIS SUMMARY\" + \" \"*19 + \"#\")\n",
            "print(\"#\" + \" \"*78 + \"#\")\n",
            "print(\"#\"*80 + \"\\n\")\n",
            "\n",
            "print(\"1. DATASET INFORMATION\")\n",
            "print(\"=\"*80)\n",
            "print(f\"   Total polymers analyzed: {len(df)}\")\n",
            "print(f\"   Polymer classes: {df['polymer_class'].nunique()}\")\n",
            "print(f\"   Class distribution:\")\n",
            "for cls, count in df['polymer_class'].value_counts().items():\n",
            "    print(f\"     - {cls}: {count} ({count/len(df)*100:.1f}%)\")\n",
            "print()\n",
            "\n",
            "print(\"2. REPRESENTATIONS COMPUTED\")\n",
            "print(\"=\"*80)\n",
            "print(f\"   ✓ Morgan Fingerprints: {morgan_fp.shape}\")\n",
            "print(f\"   ✓ MACCS Keys: {maccs_keys.shape}\")\n",
            "print(f\"   ✓ RDKit Descriptors + MACCS: {rdkit_maccs_scaled.shape}\")\n",
            "print(f\"   ✓ Transformer Embeddings (polyBERT): {transformer_scaled.shape}\")\n",
            "print(f\"   ✓ Tanimoto Distance Matrix: {tanimoto_dist.shape}\")\n",
            "print()\n",
            "\n",
            "print(\"3. MURTAGH HIERARCHICAL CLUSTERING RESULTS\")\n",
            "print(\"=\"*80)\n",
            "print(f\"   K values tested: {k_values[0]} to {k_values[-1]}\")\n",
            "print(f\"   Best K by ARI: {int(best_k_ari)} (ARI = {metrics_df.loc[metrics_df['ARI'].idxmax(), 'ARI']:.4f})\")\n",
            "print(f\"   Best K by NMI: {int(best_k_nmi)} (NMI = {metrics_df.loc[metrics_df['NMI'].idxmax(), 'NMI']:.4f})\")\n",
            "print(f\"   Best K by Silhouette: {int(best_k_sil)} (Score = {metrics_df.loc[metrics_df['silhouette'].idxmax(), 'silhouette']:.4f})\")\n",
            "print()\n",
            "\n",
            "print(\"4. SUPERVISED LEARNING RESULTS (5 x 70:30 Splits)\")\n",
            "print(\"=\"*80)\n",
            "for repr_name in representations_order:\n",
            "    summary = summary_df[summary_df['representation'] == repr_name].iloc[0]\n",
            "    print(f\"\\n   {repr_name}:\")\n",
            "    print(f\"     Train Accuracy: {summary['train_acc_mean']:.4f} ± {summary['train_acc_std']:.4f}\")\n",
            "    print(f\"     Test Accuracy:  {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}\")\n",
            "    print(f\"     Train F1:       {summary['train_f1_mean']:.4f} ± {summary['train_f1_std']:.4f}\")\n",
            "    print(f\"     Test F1:        {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}\")\n",
            "print()\n",
            "\n",
            "# Find best representation\n",
            "best_repr = summary_df.loc[summary_df['test_acc_mean'].idxmax(), 'representation']\n",
            "best_test_acc = summary_df['test_acc_mean'].max()\n",
            "print(f\"   Best representation: {best_repr} (Test Accuracy = {best_test_acc:.4f})\")\n",
            "print()\n",
            "\n",
            "print(\"5. OUTPUTS GENERATED\")\n",
            "print(\"=\"*80)\n",
            "print(\"   Representations saved in: ./representations/\")\n",
            "print(\"   Plots saved in: ./plots/\")\n",
            "print(\"     - 01_clustering_metrics_vs_k.png\")\n",
            "print(\"     - 02_dendrogram_murtagh.png\")\n",
            "print(\"     - 03_supervised_learning_results.png\")\n",
            "print(\"     - 04_umap_clusters_k5.png, k10.png, k15.png, k25.png\")\n",
            "print(\"     - 05_contingency_heatmap_k5.png, k10.png, k15.png, k25.png\")\n",
            "print(\"   CSV files:\")\n",
            "print(\"     - clustering_metrics_k2_to_k25.csv\")\n",
            "print(\"     - supervised_results_70_30_split.csv\")\n",
            "print()\n",
            "\n",
            "print(\"#\"*80)\n",
            "print(\"#\" + \" \"*26 + \"ANALYSIS COMPLETE!\" + \" \"*35 + \"#\")\n",
            "print(\"#\"*80 + \"\\n\")"
        ]
    })

    # Save the updated notebook
    nb['cells'] = new_cells

    with open('Polymer_Representation_Updated.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)

    print(\"Updated notebook saved as: Polymer_Representation_Updated.ipynb\")
    return 0

if __name__ == \"__main__\":
    sys.exit(create_updated_notebook())
