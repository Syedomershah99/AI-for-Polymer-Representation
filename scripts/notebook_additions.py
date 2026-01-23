"""
Additional analyses to add to the polymer notebook:
1. 5-fold CV with StratifiedKFold (not repeated 70:30)
2. Clustering for ALL representations (Morgan, MACCS, RDKit+MACCS, Transformer)
3. K-means clustering for Transformer embeddings
4. Train models using cluster IDs to predict polymer class
5. Generate and save all plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm
import joblib

print("="*80)
print("ADDITIONAL ANALYSES")
print("="*80)

# Assuming X_morgan, X_maccs, X_rdkit_maccs, X_transformer, df are already defined

# ============================================================================
# 1. STRATIFIED K-FOLD CROSS-VALIDATION (5-fold)
# ============================================================================

print("\n" + "="*80)
print("1. STRATIFIED 5-FOLD CROSS-VALIDATION")
print("="*80)

representations = {
    'Morgan FP': X_morgan,
    'MACCS Keys': X_maccs,
    'RDKit+MACCS': X_desc_maccs,
    'Transformer': X_transformer
}

# Remove classes with <5 samples (needed for 5-fold CV)
y = df['polymer_class'].values
class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts >= 5].index
mask = np.isin(y, valid_classes)

print(f"\nFiltering: {mask.sum()} samples (removed {(~mask).sum()} from rare classes)")

y_filtered = y[mask]
cv_results = []

for repr_name, X in representations.items():
    print(f"\n{'='*60}")
    print(f"Representation: {repr_name}")
    print(f"{'='*60}")

    X_filtered = X[mask]

    # 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_filtered, y_filtered), 1):
        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]

        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')

        fold_results.append({
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1
        })

        print(f"Fold {fold}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
              f"Test F1={test_f1:.4f}")

    # Aggregate results
    cv_results.append({
        'representation': repr_name,
        'train_acc_mean': np.mean([r['train_acc'] for r in fold_results]),
        'train_acc_std': np.std([r['train_acc'] for r in fold_results]),
        'test_acc_mean': np.mean([r['test_acc'] for r in fold_results]),
        'test_acc_std': np.std([r['test_acc'] for r in fold_results]),
        'train_f1_mean': np.mean([r['train_f1'] for r in fold_results]),
        'train_f1_std': np.std([r['train_f1'] for r in fold_results]),
        'test_f1_mean': np.mean([r['test_f1'] for r in fold_results]),
        'test_f1_std': np.std([r['test_f1'] for r in fold_results])
    })

    print(f"\nMean Test Acc: {cv_results[-1]['test_acc_mean']:.4f} ± {cv_results[-1]['test_acc_std']:.4f}")
    print(f"Mean Test F1:  {cv_results[-1]['test_f1_mean']:.4f} ± {cv_results[-1]['test_f1_std']:.4f}")

# Save results
cv_df = pd.DataFrame(cv_results)
cv_df.to_csv('plots/09_stratified_5fold_cv_results.csv', index=False)
print("\n✓ Saved: plots/09_stratified_5fold_cv_results.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(cv_results))
width = 0.35

# Accuracy
axes[0].bar(x - width/2, cv_df['train_acc_mean'], width,
            yerr=cv_df['train_acc_std'], label='Train', color='#2E86AB', alpha=0.8, capsize=5)
axes[0].bar(x + width/2, cv_df['test_acc_mean'], width,
            yerr=cv_df['test_acc_std'], label='Test', color='#A23B72', alpha=0.8, capsize=5)
axes[0].set_xlabel('Representation', fontweight='bold')
axes[0].set_ylabel('Accuracy', fontweight='bold')
axes[0].set_title('5-Fold Stratified CV - Accuracy', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cv_df['representation'], rotation=15, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# F1
axes[1].bar(x - width/2, cv_df['train_f1_mean'], width,
            yerr=cv_df['train_f1_std'], label='Train', color='#2E86AB', alpha=0.8, capsize=5)
axes[1].bar(x + width/2, cv_df['test_f1_mean'], width,
            yerr=cv_df['test_f1_std'], label='Test', color='#A23B72', alpha=0.8, capsize=5)
axes[1].set_xlabel('Representation', fontweight='bold')
axes[1].set_ylabel('F1-Score (Macro)', fontweight='bold')
axes[1].set_title('5-Fold Stratified CV - F1 Score', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(cv_df['representation'], rotation=15, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/09_stratified_5fold_cv.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/09_stratified_5fold_cv.png")
plt.show()


# ============================================================================
# 2. CLUSTERING FOR ALL REPRESENTATIONS
# ============================================================================

print("\n" + "="*80)
print("2. HIERARCHICAL CLUSTERING FOR ALL REPRESENTATIONS")
print("="*80)

all_clustering_results = {}

for repr_name, X in representations.items():
    print(f"\n{'='*60}")
    print(f"Clustering: {repr_name}")
    print(f"{'='*60}")

    # Compute distance matrix (Euclidean for continuous, Tanimoto for binary)
    if repr_name in ['Morgan FP', 'MACCS Keys']:
        # Binary - use Tanimoto/Jaccard
        print("Computing Tanimoto distance...")
        n = len(X)
        distances = []
        for i in tqdm(range(n-1), desc='Distance matrix'):
            for j in range(i+1, n):
                fp1, fp2 = X[i], X[j]
                intersection = np.sum(fp1 & fp2)
                union = np.sum(fp1 | fp2)
                sim = intersection / union if union > 0 else 0
                distances.append(1 - sim)
        dist_matrix = np.array(distances)
    else:
        # Continuous - use Euclidean
        print("Computing Euclidean distance...")
        dist_matrix = pdist(X, metric='euclidean')

    # Hierarchical clustering
    print("Performing hierarchical clustering...")
    linkage_matrix = linkage(dist_matrix, method='average')

    # Extract clusters for K=2 to 25
    cluster_assignments = {}
    metrics_list = []

    dist_matrix_square = squareform(dist_matrix)

    for k in tqdm(range(2, 26), desc='Extracting clusters'):
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        cluster_assignments[k] = clusters

        # Compute metrics
        ari = adjusted_rand_score(df['polymer_class'], clusters)
        nmi = normalized_mutual_info_score(df['polymer_class'], clusters)
        sil = silhouette_score(dist_matrix_square, clusters, metric='precomputed')

        metrics_list.append({
            'K': k,
            'representation': repr_name,
            'ARI': ari,
            'NMI': nmi,
            'Silhouette': sil
        })

    all_clustering_results[repr_name] = {
        'linkage': linkage_matrix,
        'clusters': cluster_assignments,
        'metrics': metrics_list
    }

    print(f"✓ Completed clustering for {repr_name}")

# Save all metrics
all_metrics_df = pd.DataFrame([m for r in all_clustering_results.values() for m in r['metrics']])
all_metrics_df.to_csv('plots/10_all_representations_clustering_metrics.csv', index=False)
print("\n✓ Saved: plots/10_all_representations_clustering_metrics.csv")

# Plot clustering metrics for all representations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for repr_name in representations.keys():
    repr_data = all_metrics_df[all_metrics_df['representation'] == repr_name]

    axes[0].plot(repr_data['K'], repr_data['ARI'], marker='o', label=repr_name, linewidth=2)
    axes[1].plot(repr_data['K'], repr_data['NMI'], marker='o', label=repr_name, linewidth=2)
    axes[2].plot(repr_data['K'], repr_data['Silhouette'], marker='o', label=repr_name, linewidth=2)

axes[0].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[0].set_ylabel('ARI', fontweight='bold')
axes[0].set_title('ARI vs K (All Representations)', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[1].set_ylabel('NMI', fontweight='bold')
axes[1].set_title('NMI vs K (All Representations)', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[2].set_ylabel('Silhouette', fontweight='bold')
axes[2].set_title('Silhouette vs K (All Representations)', fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/10_all_representations_clustering.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/10_all_representations_clustering.png")
plt.show()


# ============================================================================
# 3. K-MEANS CLUSTERING ON TRANSFORMER EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("3. K-MEANS CLUSTERING ON TRANSFORMER EMBEDDINGS")
print("="*80)

kmeans_results = []

for k in tqdm(range(2, 26), desc='K-means clustering'):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_transformer)

    # Compute metrics
    ari = adjusted_rand_score(df['polymer_class'], clusters)
    nmi = normalized_mutual_info_score(df['polymer_class'], clusters)
    sil = silhouette_score(X_transformer, clusters)
    inertia = kmeans.inertia_

    kmeans_results.append({
        'K': k,
        'ARI': ari,
        'NMI': nmi,
        'Silhouette': sil,
        'Inertia': inertia,
        'clusters': clusters
    })

    print(f"K={k}: ARI={ari:.4f}, NMI={nmi:.4f}, Silhouette={sil:.4f}")

# Save results
kmeans_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'clusters'} for r in kmeans_results])
kmeans_df.to_csv('plots/11_kmeans_transformer_metrics.csv', index=False)
print("\n✓ Saved: plots/11_kmeans_transformer_metrics.csv")

# Plot K-means metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(kmeans_df['K'], kmeans_df['ARI'], marker='o', color='#2E86AB', linewidth=2)
axes[0, 0].set_xlabel('K', fontweight='bold')
axes[0, 0].set_ylabel('ARI', fontweight='bold')
axes[0, 0].set_title('K-means: ARI vs K', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(kmeans_df['K'], kmeans_df['NMI'], marker='o', color='#A23B72', linewidth=2)
axes[0, 1].set_xlabel('K', fontweight='bold')
axes[0, 1].set_ylabel('NMI', fontweight='bold')
axes[0, 1].set_title('K-means: NMI vs K', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(kmeans_df['K'], kmeans_df['Silhouette'], marker='o', color='#F18F01', linewidth=2)
axes[1, 0].set_xlabel('K', fontweight='bold')
axes[1, 0].set_ylabel('Silhouette', fontweight='bold')
axes[1, 0].set_title('K-means: Silhouette vs K', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(kmeans_df['K'], kmeans_df['Inertia'], marker='o', color='#C73E1D', linewidth=2)
axes[1, 1].set_xlabel('K', fontweight='bold')
axes[1, 1].set_ylabel('Inertia (Within-cluster SSE)', fontweight='bold')
axes[1, 1].set_title('K-means: Elbow Plot', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/11_kmeans_transformer.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/11_kmeans_transformer.png")
plt.show()


# ============================================================================
# 4. TRAIN MODELS USING CLUSTER IDS TO PREDICT POLYMER CLASS
# ============================================================================

print("\n" + "="*80)
print("4. SUPERVISED LEARNING USING CLUSTER IDS AS FEATURES")
print("="*80)

# Create feature matrices with cluster assignments
cluster_feature_results = []

for k in [5, 10, 15, 20, 25]:
    print(f"\n{'='*60}")
    print(f"Training with K={k} cluster features")
    print(f"{'='*60}")

    # Collect cluster assignments from all representations
    cluster_features = []
    feature_names = []

    # Hierarchical clustering for each representation
    for repr_name in representations.keys():
        if repr_name in all_clustering_results:
            clusters = all_clustering_results[repr_name]['clusters'][k]
            cluster_features.append(clusters.reshape(-1, 1))
            feature_names.append(f'{repr_name}_cluster')

    # K-means clusters from transformer
    kmeans_clusters = [r['clusters'] for r in kmeans_results if r['K'] == k][0]
    cluster_features.append(kmeans_clusters.reshape(-1, 1))
    feature_names.append('Transformer_KMeans_cluster')

    # Combine all cluster features
    X_clusters = np.hstack(cluster_features)

    print(f"Feature matrix shape: {X_clusters.shape}")
    print(f"Features: {feature_names}")

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_clusters[mask], y_filtered), 1):
        X_train = X_clusters[mask][train_idx]
        X_test = X_clusters[mask][test_idx]
        y_train = y_filtered[train_idx]
        y_test = y_filtered[test_idx]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')

        fold_results.append({
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1
        })

    cluster_feature_results.append({
        'K': k,
        'train_acc_mean': np.mean([r['train_acc'] for r in fold_results]),
        'train_acc_std': np.std([r['train_acc'] for r in fold_results]),
        'test_acc_mean': np.mean([r['test_acc'] for r in fold_results]),
        'test_acc_std': np.std([r['test_acc'] for r in fold_results]),
        'train_f1_mean': np.mean([r['train_f1'] for r in fold_results]),
        'train_f1_std': np.std([r['train_f1'] for r in fold_results]),
        'test_f1_mean': np.mean([r['test_f1'] for r in fold_results]),
        'test_f1_std': np.std([r['test_f1'] for r in fold_results])
    })

    print(f"Test Acc: {cluster_feature_results[-1]['test_acc_mean']:.4f} ± {cluster_feature_results[-1]['test_acc_std']:.4f}")
    print(f"Test F1:  {cluster_feature_results[-1]['test_f1_mean']:.4f} ± {cluster_feature_results[-1]['test_f1_std']:.4f}")

# Save results
cluster_pred_df = pd.DataFrame(cluster_feature_results)
cluster_pred_df.to_csv('plots/12_cluster_based_prediction.csv', index=False)
print("\n✓ Saved: plots/12_cluster_based_prediction.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(cluster_pred_df['K'], cluster_pred_df['train_acc_mean'],
             marker='o', label='Train', linewidth=2, color='#2E86AB')
axes[0].fill_between(cluster_pred_df['K'],
                      cluster_pred_df['train_acc_mean'] - cluster_pred_df['train_acc_std'],
                      cluster_pred_df['train_acc_mean'] + cluster_pred_df['train_acc_std'],
                      alpha=0.2, color='#2E86AB')
axes[0].plot(cluster_pred_df['K'], cluster_pred_df['test_acc_mean'],
             marker='s', label='Test', linewidth=2, color='#A23B72')
axes[0].fill_between(cluster_pred_df['K'],
                      cluster_pred_df['test_acc_mean'] - cluster_pred_df['test_acc_std'],
                      cluster_pred_df['test_acc_mean'] + cluster_pred_df['test_acc_std'],
                      alpha=0.2, color='#A23B72')
axes[0].set_xlabel('K (Number of Clusters)', fontweight='bold')
axes[0].set_ylabel('Accuracy', fontweight='bold')
axes[0].set_title('Cluster-Based Classification: Accuracy', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(cluster_pred_df['K'], cluster_pred_df['train_f1_mean'],
             marker='o', label='Train', linewidth=2, color='#2E86AB')
axes[1].fill_between(cluster_pred_df['K'],
                      cluster_pred_df['train_f1_mean'] - cluster_pred_df['train_f1_std'],
                      cluster_pred_df['train_f1_mean'] + cluster_pred_df['train_f1_std'],
                      alpha=0.2, color='#2E86AB')
axes[1].plot(cluster_pred_df['K'], cluster_pred_df['test_f1_mean'],
             marker='s', label='Test', linewidth=2, color='#A23B72')
axes[1].fill_between(cluster_pred_df['K'],
                      cluster_pred_df['test_f1_mean'] - cluster_pred_df['test_f1_std'],
                      cluster_pred_df['test_f1_mean'] + cluster_pred_df['test_f1_std'],
                      alpha=0.2, color='#A23B72')
axes[1].set_xlabel('K (Number of Clusters)', fontweight='bold')
axes[1].set_ylabel('F1-Score (Macro)', fontweight='bold')
axes[1].set_title('Cluster-Based Classification: F1-Score', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/12_cluster_based_prediction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/12_cluster_based_prediction.png")
plt.show()

print("\n" + "="*80)
print("ALL ADDITIONAL ANALYSES COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - plots/09_stratified_5fold_cv_results.csv")
print("  - plots/09_stratified_5fold_cv.png")
print("  - plots/10_all_representations_clustering_metrics.csv")
print("  - plots/10_all_representations_clustering.png")
print("  - plots/11_kmeans_transformer_metrics.csv")
print("  - plots/11_kmeans_transformer.png")
print("  - plots/12_cluster_based_prediction.csv")
print("  - plots/12_cluster_based_prediction.png")
