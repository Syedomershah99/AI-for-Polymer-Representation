#!/usr/bin/env python3
"""
Polymer Representation Analysis - Executable Script
Generates all plots and saves to plots/ folder
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up paths
WORK_DIR = "/Users/Omer/Desktop/Research"
PLOT_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
os.chdir(WORK_DIR)

print("=" * 60)
print("POLYMER REPRESENTATION ANALYSIS")
print("=" * 60)

# Import libraries
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Draw
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    accuracy_score, f1_score, pairwise_distances
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

import umap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("All imports successful!")

# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================
print("\n[1/10] Loading and cleaning dataset...")

DATA_PATH = os.path.join(WORK_DIR, "PI1070.csv")
df_raw = pd.read_csv(DATA_PATH)
print(f"  Loaded: {df_raw.shape}")

def canonicalize_smiles(smiles):
    try:
        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

df = df_raw.copy()
df['smiles_canonical'] = df['smiles'].apply(canonicalize_smiles)
n_invalid = df['smiles_canonical'].isna().sum()
df = df[df['smiles_canonical'].notna()].reset_index(drop=True)
print(f"  Removed {n_invalid} invalid SMILES")

n_before = len(df)
df = df.drop_duplicates(subset='smiles_canonical', keep='first').reset_index(drop=True)
print(f"  Removed {n_before - len(df)} duplicates")
print(f"  Final dataset: {len(df)} polymers")

df['mol'] = df['smiles_canonical'].apply(Chem.MolFromSmiles)

# ============================================================
# 2. COMPUTE REPRESENTATIONS
# ============================================================
print("\n[2/10] Computing representations...")

# Morgan fingerprints
print("  Computing Morgan fingerprints...")
fps_morgan_rdkit = []
fps_morgan_array = []
for mol in tqdm(df['mol'], desc="  Morgan FP"):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fps_morgan_rdkit.append(fp)
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps_morgan_array.append(arr)
X_morgan = np.vstack(fps_morgan_array)
print(f"  Morgan FP shape: {X_morgan.shape}")

# MACCS keys
print("  Computing MACCS keys...")
fps_maccs_rdkit = []
fps_maccs_array = []
for mol in tqdm(df['mol'], desc="  MACCS"):
    fp = MACCSkeys.GenMACCSKeys(mol)
    fps_maccs_rdkit.append(fp)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps_maccs_array.append(arr)
X_maccs = np.vstack(fps_maccs_array)
print(f"  MACCS keys shape: {X_maccs.shape}")

# RDKit descriptors
print("  Computing RDKit descriptors...")
descriptor_names = [d[0] for d in Descriptors._descList]
descriptor_fns = [d[1] for d in Descriptors._descList]
results = []
for mol in tqdm(df['mol'], desc="  RDKit Desc"):
    vals = []
    for fn in descriptor_fns:
        try:
            v = fn(mol)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = np.nan
        except:
            v = np.nan
        vals.append(v)
    results.append(vals)
X_rdkit_desc = np.array(results, dtype=np.float32)
valid_mask = ~np.all(np.isnan(X_rdkit_desc), axis=0)
X_rdkit_desc = X_rdkit_desc[:, valid_mask]
col_median = np.nanmedian(X_rdkit_desc, axis=0)
nan_idx = np.where(np.isnan(X_rdkit_desc))
X_rdkit_desc[nan_idx] = np.take(col_median, nan_idx[1])
print(f"  RDKit descriptors shape: {X_rdkit_desc.shape}")

# Combined RDKit + MACCS
X_desc_maccs = np.hstack([X_rdkit_desc, X_maccs.astype(np.float32)])
print(f"  RDKit + MACCS shape: {X_desc_maccs.shape}")

# Transformer embeddings
print("  Computing Transformer embeddings (polyBERT)...")
try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("kuelumbus/polyBERT")
    model = AutoModel.from_pretrained("kuelumbus/polyBERT").to(device)
    model.eval()

    embeddings = []
    batch_size = 32
    smiles_list = df['smiles_canonical'].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="  Transformer"):
            batch = smiles_list[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            output = model(**tokens)
            emb = output.last_hidden_state[:, 0, :]
            embeddings.append(emb.cpu().numpy())

    X_transformer = np.vstack(embeddings)
    print(f"  Transformer embeddings shape: {X_transformer.shape}")
except Exception as e:
    print(f"  Warning: Transformer embeddings failed ({e})")
    print("  Using PCA of Morgan FP as fallback")
    pca = PCA(n_components=100, random_state=42)
    X_transformer = pca.fit_transform(X_morgan.astype(np.float32))

# Standardize
scaler_desc_maccs = StandardScaler()
X_desc_maccs_scaled = scaler_desc_maccs.fit_transform(X_desc_maccs)

scaler_transformer = StandardScaler()
X_transformer_scaled = scaler_transformer.fit_transform(X_transformer)

print("\n  === Representation Summary ===")
print(f"  Morgan FP:        {X_morgan.shape}")
print(f"  MACCS Keys:       {X_maccs.shape}")
print(f"  RDKit + MACCS:    {X_desc_maccs_scaled.shape}")
print(f"  Transformer:      {X_transformer_scaled.shape}")

# ============================================================
# 3. K-MEANS CLUSTERING
# ============================================================
print("\n[3/10] K-means clustering analysis...")

k_range = range(2, 16)
inertias = []
silhouettes = []

for k in tqdm(k_range, desc="  K-means"):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_transformer_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_transformer_scaled, labels))

best_k = list(k_range)[np.argmax(silhouettes)]
print(f"  Best k by silhouette: {best_k} (score: {max(silhouettes):.4f})")

# Plot elbow and silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Curve for K-Means', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), silhouettes, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=best_k, color='g', linestyle='--', label=f'Best k={best_k}')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_kmeans_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: 01_kmeans_analysis.png")

# Final K-means
if 'polymer_class' in df.columns:
    k_final = df['polymer_class'].nunique()
else:
    k_final = best_k

kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans_final.fit_predict(X_transformer_scaled)
print(f"  K-means with k={k_final} applied")

# ============================================================
# 4. TANIMOTO CLUSTERING
# ============================================================
print("\n[4/10] Tanimoto-distance clustering...")

def butina_cluster(fps, cutoff=0.3):
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=cutoff, isDistData=True)
    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels, clusters

print("  Butina clustering on Morgan FP...")
df['butina_morgan'], clusters_morgan = butina_cluster(fps_morgan_rdkit, cutoff=0.3)
print(f"  Morgan clusters: {len(clusters_morgan)}")

print("  Butina clustering on MACCS...")
df['butina_maccs'], clusters_maccs = butina_cluster(fps_maccs_rdkit, cutoff=0.3)
print(f"  MACCS clusters: {len(clusters_maccs)}")

# Hierarchical clustering
print("  Computing Tanimoto distance matrix...")
n = len(fps_morgan_rdkit)
dist_matrix = np.zeros((n, n), dtype=np.float32)
for i in tqdm(range(n), desc="  Distance matrix"):
    sims = DataStructs.BulkTanimotoSimilarity(fps_morgan_rdkit[i], fps_morgan_rdkit)
    dist_matrix[i, :] = [1.0 - s for s in sims]

condensed = squareform(dist_matrix)
linkage_morgan = linkage(condensed, method='average')

if 'polymer_class' in df.columns:
    n_coarse = df['polymer_class'].nunique()
else:
    n_coarse = 10

df['hier_morgan'] = fcluster(linkage_morgan, n_coarse, criterion='maxclust') - 1
print(f"  Hierarchical clusters: {n_coarse}")

# Dendrogram
plt.figure(figsize=(15, 8))
dendrogram(linkage_morgan, truncate_mode='lastp', p=30, leaf_rotation=90,
           leaf_font_size=10, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Morgan FP, Tanimoto Distance)', fontsize=14)
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_dendrogram_morgan.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: 02_dendrogram_morgan.png")

# ============================================================
# 5. CLUSTERING VALIDATION
# ============================================================
print("\n[5/10] Clustering validation...")

if 'polymer_class' in df.columns:
    true_labels = df['polymer_class'].values

    validation_results = []

    # K-means
    ari = adjusted_rand_score(true_labels, df['kmeans_cluster'].values)
    nmi = normalized_mutual_info_score(true_labels, df['kmeans_cluster'].values)
    sil = silhouette_score(X_transformer_scaled, df['kmeans_cluster'].values)
    validation_results.append({'Method': 'K-means (Transformer)', 'ARI': ari, 'NMI': nmi, 'Silhouette': sil})

    # Butina Morgan
    ari = adjusted_rand_score(true_labels, df['butina_morgan'].values)
    nmi = normalized_mutual_info_score(true_labels, df['butina_morgan'].values)
    sil = silhouette_score(X_morgan, df['butina_morgan'].values) if df['butina_morgan'].nunique() > 1 else 0
    validation_results.append({'Method': 'Butina (Morgan)', 'ARI': ari, 'NMI': nmi, 'Silhouette': sil})

    # Butina MACCS
    ari = adjusted_rand_score(true_labels, df['butina_maccs'].values)
    nmi = normalized_mutual_info_score(true_labels, df['butina_maccs'].values)
    sil = silhouette_score(X_maccs, df['butina_maccs'].values) if df['butina_maccs'].nunique() > 1 else 0
    validation_results.append({'Method': 'Butina (MACCS)', 'ARI': ari, 'NMI': nmi, 'Silhouette': sil})

    # Hierarchical
    ari = adjusted_rand_score(true_labels, df['hier_morgan'].values)
    nmi = normalized_mutual_info_score(true_labels, df['hier_morgan'].values)
    sil = silhouette_score(X_morgan, df['hier_morgan'].values)
    validation_results.append({'Method': 'Hierarchical (Morgan)', 'ARI': ari, 'NMI': nmi, 'Silhouette': sil})

    validation_df = pd.DataFrame(validation_results)
    print(validation_df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(validation_df))
    width = 0.25

    ax.bar(x - width, validation_df['ARI'], width, label='ARI', color='steelblue')
    ax.bar(x, validation_df['NMI'], width, label='NMI', color='coral')
    ax.bar(x + width, validation_df['Silhouette'], width, label='Silhouette', color='seagreen')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Clustering Validation: ARI / NMI / Silhouette', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(validation_df['Method'], rotation=20, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_clustering_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 03_clustering_validation.png")

# ============================================================
# 6. SUPERVISED BASELINES
# ============================================================
print("\n[6/10] Supervised baselines (5-fold CV)...")

if 'polymer_class' in df.columns:
    y = df['polymer_class'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []

    representations = {
        'Morgan FP': X_morgan,
        'MACCS Keys': X_maccs,
        'RDKit + MACCS': X_desc_maccs_scaled,
        'Transformer': X_transformer_scaled
    }

    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LogisticReg': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    for rep_name, X in representations.items():
        print(f"  Evaluating {rep_name}...")
        for clf_name, clf in classifiers.items():
            y_pred = cross_val_predict(clf, X, y_encoded, cv=skf)
            acc = accuracy_score(y_encoded, y_pred)
            f1_m = f1_score(y_encoded, y_pred, average='macro')
            f1_w = f1_score(y_encoded, y_pred, average='weighted')
            all_results.append({
                'Representation': rep_name, 'Classifier': clf_name,
                'Accuracy': acc, 'F1 (macro)': f1_m, 'F1 (weighted)': f1_w
            })

    results_df = pd.DataFrame(all_results)
    print("\n  === Supervised Results ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(WORK_DIR, 'supervised_cv_results.csv'), index=False)

    # Plot
    pivot_acc = results_df.pivot(index='Representation', columns='Classifier', values='Accuracy')

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_acc.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Supervised Classification: Accuracy by Representation (5-Fold CV)', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(title='Classifier')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_supervised_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 04_supervised_accuracy.png")

# ============================================================
# 7. UMAP EMBEDDINGS
# ============================================================
print("\n[7/10] Computing UMAP embeddings...")

def compute_umap(X, metric='euclidean'):
    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.2, metric=metric, random_state=42)
    return reducer.fit_transform(X)

print("  UMAP for Morgan FP...")
Z_morgan = compute_umap(X_morgan.astype(np.float32), metric='jaccard')
print("  UMAP for MACCS...")
Z_maccs = compute_umap(X_maccs.astype(np.float32), metric='jaccard')
print("  UMAP for RDKit+MACCS...")
Z_desc_maccs = compute_umap(X_desc_maccs_scaled, metric='euclidean')
print("  UMAP for Transformer...")
Z_transformer = compute_umap(X_transformer_scaled, metric='euclidean')

# Save UMAP embeddings
np.save(os.path.join(WORK_DIR, 'Z_morgan_umap.npy'), Z_morgan)
np.save(os.path.join(WORK_DIR, 'Z_transformer_umap.npy'), Z_transformer)

# ============================================================
# 8. UMAP VISUALIZATIONS
# ============================================================
print("\n[8/10] Creating UMAP visualizations...")

embeddings = {
    'Morgan FP (ECFP)': Z_morgan,
    'MACCS Keys': Z_maccs,
    'RDKit + MACCS': Z_desc_maccs,
    'Transformer (polyBERT)': Z_transformer
}

# By polymer class
if 'polymer_class' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (name, Z) in zip(axes, embeddings.items()):
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=df['polymer_class'].values, s=15, alpha=0.7, cmap='tab10')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')

    plt.colorbar(scatter, ax=axes, shrink=0.6, label='polymer_class')
    fig.suptitle('Chemical Space Colored by Polymer Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '05_umap_by_class.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 05_umap_by_class.png")

# By K-means cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, (name, Z) in zip(axes, embeddings.items()):
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=df['kmeans_cluster'].values, s=15, alpha=0.7, cmap='tab10')
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

plt.colorbar(scatter, ax=axes, shrink=0.6, label='K-means Cluster')
fig.suptitle('Chemical Space Colored by K-means Cluster', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_umap_by_kmeans.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: 06_umap_by_kmeans.png")

# By properties
props = ['density', 'bulk_modulus', 'thermal_conductivity', 'static_dielectric_const']
props_available = [p for p in props if p in df.columns]

for prop in props_available:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    v = df[prop].values
    vmin, vmax = np.nanmin(v), np.nanmax(v)

    for ax, (name, Z) in zip(axes, embeddings.items()):
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=v, s=15, alpha=0.7, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')

    plt.colorbar(scatter, ax=axes, shrink=0.6, label=prop)
    fig.suptitle(f'Chemical Space Colored by {prop}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'07_umap_by_{prop}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07_umap_by_{prop}.png")

# ============================================================
# 9. REPRESENTATIVE STRUCTURES
# ============================================================
print("\n[9/10] Finding representative polymers...")

def find_representatives(X, labels, method='centroid'):
    reps = {}
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        X_cluster = X[mask]

        if len(indices) == 1:
            reps[cluster_id] = indices[0]
            continue

        if method == 'centroid':
            centroid = X_cluster.mean(axis=0)
            distances = np.linalg.norm(X_cluster - centroid, axis=1)
            reps[cluster_id] = indices[np.argmin(distances)]
        else:
            dist_matrix = pairwise_distances(X_cluster)
            reps[cluster_id] = indices[np.argmin(dist_matrix.sum(axis=1))]

    return reps

reps_kmeans = find_representatives(X_transformer_scaled, df['kmeans_cluster'].values)
reps_hier = find_representatives(X_morgan.astype(np.float32), df['hier_morgan'].values, method='medoid')

# Visualize K-means representatives
mols = []
legends = []
for cluster_id in sorted(reps_kmeans.keys()):
    idx = reps_kmeans[cluster_id]
    mols.append(df.loc[idx, 'mol'])
    n_members = (df['kmeans_cluster'] == cluster_id).sum()
    legends.append(f"Cluster {cluster_id}\n(n={n_members})")

img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300), legends=legends)
img.save(os.path.join(PLOT_DIR, '08_kmeans_representatives.png'))
print(f"  Saved: 08_kmeans_representatives.png")

# Visualize hierarchical representatives
mols = []
legends = []
for cluster_id in sorted(reps_hier.keys()):
    idx = reps_hier[cluster_id]
    mols.append(df.loc[idx, 'mol'])
    n_members = (df['hier_morgan'] == cluster_id).sum()
    legends.append(f"Cluster {cluster_id}\n(n={n_members})")

img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300), legends=legends)
img.save(os.path.join(PLOT_DIR, '09_hierarchical_representatives.png'))
print(f"  Saved: 09_hierarchical_representatives.png")

# ============================================================
# 10. MACCS ANALYSIS
# ============================================================
print("\n[10/10] MACCS key analysis...")

maccs_freq = X_maccs.mean(axis=0)

plt.figure(figsize=(14, 6))
plt.bar(range(len(maccs_freq)), maccs_freq, color='steelblue', alpha=0.7)
plt.xlabel('MACCS Key Index', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('MACCS Key Frequency Distribution Across Polymers', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_maccs_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: 10_maccs_distribution.png")

# MACCS heatmap by class
if 'polymer_class' in df.columns:
    class_profiles = {}
    for cls in sorted(df['polymer_class'].unique()):
        mask = df['polymer_class'] == cls
        class_profiles[cls] = X_maccs[mask].mean(axis=0)

    profile_matrix = np.array([class_profiles[cls] for cls in sorted(class_profiles.keys())])
    key_variance = profile_matrix.var(axis=0)
    top_keys = np.argsort(key_variance)[::-1][:30]

    plt.figure(figsize=(16, 8))
    plt.imshow(profile_matrix[:, top_keys], aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Key Frequency')
    plt.xlabel('MACCS Key Index')
    plt.ylabel('Polymer Class')
    plt.title('MACCS Key Profiles by Polymer Class (Top 30 Variable Keys)', fontsize=14)
    plt.yticks(range(len(class_profiles)), sorted(class_profiles.keys()))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '11_maccs_class_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 11_maccs_class_heatmap.png")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

df_save = df.drop(columns=['mol'])
df_save.to_csv(os.path.join(WORK_DIR, 'PI1070_cleaned_with_clusters.csv'), index=False)
np.save(os.path.join(WORK_DIR, 'X_morgan.npy'), X_morgan)
np.save(os.path.join(WORK_DIR, 'X_maccs.npy'), X_maccs)
np.save(os.path.join(WORK_DIR, 'X_transformer.npy'), X_transformer)

print(f"\nAll plots saved to: {PLOT_DIR}")
print(f"Data saved to: {WORK_DIR}")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY REPORT - Jan 6, 2026")
print("=" * 60)
print(f"\n1. DATASET")
print(f"   Original: {len(df_raw)} | Cleaned: {len(df)}")
if 'polymer_class' in df.columns:
    print(f"   Classes: {df['polymer_class'].nunique()}")

print(f"\n2. REPRESENTATIONS")
print(f"   Morgan FP: {X_morgan.shape}")
print(f"   MACCS Keys: {X_maccs.shape}")
print(f"   Transformer: {X_transformer.shape}")

print(f"\n3. CLUSTERING")
print(f"   K-means: {df['kmeans_cluster'].nunique()} clusters")
print(f"   Butina (Morgan): {df['butina_morgan'].nunique()} clusters")
print(f"   Hierarchical: {df['hier_morgan'].nunique()} clusters")

if 'polymer_class' in df.columns:
    print(f"\n4. VALIDATION")
    print(validation_df.to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
