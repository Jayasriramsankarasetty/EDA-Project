"""
Clustering Model Module for India Census 2011 EDA & Clustering Project

This module contains functions for performing unsupervised learning 
(clustering) analysis using KMeans and other clustering algorithms.

Author: Data Science Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def determine_optimal_clusters(X, max_k=10, methods=['elbow', 'silhouette']):
    """
    Determine optimal number of clusters using multiple methods.
    
    Parameters:
    -----------
    X : array-like
        Standardized feature matrix
    max_k : int
        Maximum number of clusters to test
    methods : list
        List of methods to use ('elbow', 'silhouette')
        
    Returns:
    --------
    dict
        Results from different methods
    """
    print(f"🔍 Determining optimal number of clusters (k=2 to {max_k})")
    print("="*60)
    
    k_range = range(2, max_k + 1)
    results = {
        'k_values': list(k_range),
        'inertias': [],
        'silhouette_scores': []
    }
    
    # Test different k values
    for k in k_range:
        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        results['inertias'].append(inertia)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X, cluster_labels)
        results['silhouette_scores'].append(sil_score)
        
        print(f"k={k}: Inertia={inertia:.2f}, Silhouette Score={sil_score:.3f}")
    
    # Determine optimal k using different methods
    optimal_k_results = {}
    
    if 'silhouette' in methods:
        optimal_k_silhouette = k_range[np.argmax(results['silhouette_scores'])]
        max_silhouette = max(results['silhouette_scores'])
        optimal_k_results['silhouette'] = {
            'k': optimal_k_silhouette,
            'score': max_silhouette
        }
    
    if 'elbow' in methods:
        # Simple elbow detection (find point where rate of decrease slows)
        inertias = results['inertias']
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        elbow_k = k_range[np.argmax(np.diff(diffs))] + 2
        optimal_k_results['elbow'] = {
            'k': elbow_k,
            'inertia': inertias[elbow_k - k_range[0]]
        }
    
    results['optimal_k'] = optimal_k_results
    
    print(f"\\n🎯 OPTIMAL K RECOMMENDATIONS:")
    for method, result in optimal_k_results.items():
        if method == 'silhouette':
            print(f"Silhouette Method: k = {result['k']} (score: {result['score']:.3f})")
        elif method == 'elbow':
            print(f"Elbow Method: k ≈ {result['k']}")
    
    return results

def plot_cluster_optimization(results, title="Cluster Optimization Analysis"):
    """
    Plot elbow method and silhouette analysis results.
    
    Parameters:
    -----------
    results : dict
        Results from determine_optimal_clusters function
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    k_values = results['k_values']
    inertias = results['inertias']
    silhouette_scores = results['silhouette_scores']
    
    # Elbow Method Plot
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.set_title('Elbow Method for Optimal k', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Highlight elbow point if available
    if 'elbow' in results.get('optimal_k', {}):
        elbow_k = results['optimal_k']['elbow']['k']
        if elbow_k in k_values:
            elbow_idx = k_values.index(elbow_k)
            ax1.scatter(elbow_k, inertias[elbow_idx], color='red', s=100, zorder=5)
            ax1.annotate(f'Elbow\\nk={elbow_k}', (elbow_k, inertias[elbow_idx]),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Add annotations for inertias
    for i, (k, inertia) in enumerate(zip(k_values, inertias)):
        ax1.annotate(f'{inertia:.1f}', (k, inertia), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Silhouette Score Plot
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Highlight best silhouette score
    if 'silhouette' in results.get('optimal_k', {}):
        best_k = results['optimal_k']['silhouette']['k']
        best_score = results['optimal_k']['silhouette']['score']
        best_idx = k_values.index(best_k)
        ax2.scatter(best_k, best_score, color='green', s=100, zorder=5)
        ax2.annotate(f'Best\\nk={best_k}\\n{best_score:.3f}', (best_k, best_score),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add annotations for silhouette scores
    for i, (k, score) in enumerate(zip(k_values, silhouette_scores)):
        ax2.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def perform_kmeans_clustering(X, n_clusters, random_state=42):
    """
    Perform KMeans clustering.
    
    Parameters:
    -----------
    X : array-like
        Standardized feature matrix
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (KMeans model, cluster labels, silhouette score)
    """
    print(f"🎯 Performing KMeans clustering with k={n_clusters}")
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X, cluster_labels)
    
    print(f"✅ KMeans clustering completed")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(cluster_labels) * 100
        print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
    
    return kmeans, cluster_labels, sil_score

def perform_pca_analysis(X, n_components=2, random_state=42):
    """
    Perform Principal Component Analysis for visualization.
    
    Parameters:
    -----------
    X : array-like
        Standardized feature matrix
    n_components : int
        Number of principal components
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (PCA model, transformed data, explained variance ratio)
    """
    print(f"📊 Performing PCA with {n_components} components")
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    print(f"✅ PCA completed")
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)")
    
    return pca, X_pca, explained_variance

def plot_cluster_visualization(X_pca, cluster_labels, cluster_centers_pca=None, 
                             feature_names=None, pca_model=None, state_names=None,
                             title="Cluster Visualization"):
    """
    Visualize clusters in 2D PCA space.
    
    Parameters:
    -----------
    X_pca : array-like
        PCA-transformed data
    cluster_labels : array-like
        Cluster assignments
    cluster_centers_pca : array-like, optional
        PCA-transformed cluster centers
    feature_names : list, optional
        Original feature names
    pca_model : PCA, optional
        Fitted PCA model
    state_names : list, optional
        Names of data points for annotation
    title : str
        Plot title
    """
    plt.figure(figsize=(14, 10))
    
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    # Create scatter plot for each cluster
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add cluster centers if provided
    if cluster_centers_pca is not None:
        plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    # Add state labels if provided
    if state_names is not None and len(state_names) == len(X_pca):
        for i, name in enumerate(state_names):
            plt.annotate(name, (X_pca[i, 0], X_pca[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # Set labels with variance explained if PCA model is provided
    if pca_model is not None:
        xlabel = f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)'
        ylabel = f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)'
    else:
        xlabel = 'First Principal Component'
        ylabel = 'Second Principal Component'
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontweight='bold', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_cluster_characteristics(df, cluster_labels, features, state_names=None):
    """
    Analyze characteristics of each cluster.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset with features
    cluster_labels : array-like
        Cluster assignments
    features : list
        List of features used for clustering
    state_names : list, optional
        Names of data points
        
    Returns:
    --------
    pandas.DataFrame
        Cluster characteristics summary
    """
    print("📊 ANALYZING CLUSTER CHARACTERISTICS")
    print("="*50)
    
    # Create DataFrame with clusters
    cluster_df = df[features].copy()
    cluster_df['Cluster'] = cluster_labels
    
    if state_names is not None:
        cluster_df['State_Name'] = state_names
    
    # Calculate cluster means
    cluster_means = cluster_df.groupby('Cluster')[features].mean()
    
    print("\\n📈 CLUSTER MEAN VALUES:")
    print(cluster_means.round(2))
    
    # Create detailed summary
    cluster_summary = []
    
    for cluster_id in sorted(np.unique(cluster_labels)):
        cluster_subset = cluster_df[cluster_df['Cluster'] == cluster_id]
        
        summary = {
            'Cluster': cluster_id,
            'Size': len(cluster_subset),
            'Percentage': len(cluster_subset) / len(cluster_df) * 100
        }
        
        # Add mean values for each feature
        for feature in features:
            summary[f'Mean_{feature}'] = cluster_subset[feature].mean()
            summary[f'Std_{feature}'] = cluster_subset[feature].std()
        
        cluster_summary.append(summary)
    
    summary_df = pd.DataFrame(cluster_summary)
    
    print(f"\\n📋 DETAILED CLUSTER SUMMARY:")
    for _, row in summary_df.iterrows():
        cluster_id = int(row['Cluster'])
        size = int(row['Size'])
        percentage = row['Percentage']
        
        print(f"\\nCluster {cluster_id} ({size} samples, {percentage:.1f}%):")
        
        if state_names is not None and 'State_Name' in cluster_df.columns:
            cluster_states = cluster_df[cluster_df['Cluster'] == cluster_id]['State_Name'].tolist()
            print(f"  States: {', '.join(cluster_states)}")
        
        print("  Key characteristics:")
        for feature in features:
            mean_val = row[f'Mean_{feature}']
            std_val = row[f'Std_{feature}']
            print(f"    • {feature}: {mean_val:.2f} ± {std_val:.2f}")
    
    return summary_df, cluster_means

def plot_cluster_characteristics(cluster_means, title="Cluster Characteristics"):
    """
    Visualize cluster characteristics.
    
    Parameters:
    -----------
    cluster_means : pandas.DataFrame
        Mean values for each cluster
    title : str
        Plot title
    """
    n_features = len(cluster_means.columns)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(cluster_means.columns):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Bar plot for each cluster
        bars = ax.bar(range(len(cluster_means)), cluster_means[feature], 
                     alpha=0.8, color=plt.cm.Set1(np.linspace(0, 1, len(cluster_means))))
        
        # Customize plot
        ax.set_xlabel('Cluster')
        ax.set_ylabel(f'Mean {feature}')
        ax.set_title(f'Average {feature} by Cluster', fontweight='bold')
        ax.set_xticks(range(len(cluster_means)))
        ax.set_xticklabels([f'Cluster {i}' for i in cluster_means.index])
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, cluster_means[feature])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(cluster_means[feature]),
                   f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def interpret_clusters(cluster_means, features):
    """
    Provide interpretations for clusters based on their characteristics.
    
    Parameters:
    -----------
    cluster_means : pandas.DataFrame
        Mean values for each cluster
    features : list
        List of features used for clustering
        
    Returns:
    --------
    dict
        Cluster interpretations
    """
    print("🏷️ CLUSTER INTERPRETATION")
    print("="*40)
    
    interpretations = {}
    
    # Calculate overall means for comparison
    overall_means = cluster_means.mean()
    
    for cluster_id in cluster_means.index:
        cluster_values = cluster_means.loc[cluster_id]
        
        characteristics = []
        
        # Analyze each feature
        for feature in features:
            cluster_val = cluster_values[feature]
            overall_val = overall_means[feature]
            
            if cluster_val > overall_val * 1.1:  # 10% above average
                characteristics.append(f"High {feature}")
            elif cluster_val < overall_val * 0.9:  # 10% below average
                characteristics.append(f"Low {feature}")
        
        # Create interpretation
        if characteristics:
            interpretation = ", ".join(characteristics[:3])  # Limit to top 3 characteristics
        else:
            interpretation = "Average characteristics"
        
        interpretations[cluster_id] = interpretation
        
        print(f"\\nCluster {cluster_id}: {interpretation}")
        print(f"  Key features:")
        for feature in features:
            cluster_val = cluster_values[feature]
            overall_val = overall_means[feature]
            deviation = ((cluster_val - overall_val) / overall_val) * 100
            print(f"    • {feature}: {cluster_val:.2f} ({deviation:+.1f}% vs average)")
    
    return interpretations

def save_clustering_results(model, scaler, pca_model, cluster_data, output_dir):
    """
    Save clustering models and results.
    
    Parameters:
    -----------
    model : sklearn estimator
        Fitted clustering model
    scaler : StandardScaler
        Fitted scaler
    pca_model : PCA
        Fitted PCA model
    cluster_data : pandas.DataFrame
        Data with cluster assignments
    output_dir : str
        Directory to save results
    """
    import os
    
    print(f"💾 Saving clustering results to {output_dir}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Save models
        joblib.dump(model, os.path.join(output_dir, 'clustering_model.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(pca_model, os.path.join(output_dir, 'pca_model.pkl'))
        
        # Save cluster results
        cluster_data.to_csv(os.path.join(output_dir, 'cluster_results.csv'), index=False)
        
        print("✅ All clustering results saved successfully")
        
        # List saved files
        saved_files = ['clustering_model.pkl', 'scaler.pkl', 'pca_model.pkl', 'cluster_results.csv']
        for file in saved_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  ✓ {file} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing clustering functions...")
    
    # Create sample data
    from sklearn.datasets import make_blobs
    
    X_sample, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    feature_names = ['Feature1', 'Feature2']
    
    print("✅ Sample data created")
    
    # Test optimal cluster determination
    print("\\n1. Testing optimal cluster determination...")
    results = determine_optimal_clusters(X_sample, max_k=8)
    plot_cluster_optimization(results)
    
    # Test KMeans clustering
    print("\\n2. Testing KMeans clustering...")
    optimal_k = results['optimal_k']['silhouette']['k']
    model, labels, score = perform_kmeans_clustering(X_sample, optimal_k)
    
    # Test PCA
    print("\\n3. Testing PCA...")
    pca_model, X_pca, explained_var = perform_pca_analysis(X_sample)
    
    # Test visualization
    print("\\n4. Testing cluster visualization...")
    centers_pca = pca_model.transform(model.cluster_centers_)
    plot_cluster_visualization(X_pca, labels, centers_pca, feature_names, pca_model)
    
    print("\\n🎉 All clustering tests completed successfully!")