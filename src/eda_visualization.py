"""
EDA Visualization Module for India Census 2011 EDA & Clustering Project

This module contains functions for generating comprehensive exploratory 
data analysis visualizations.

Author: Data Science Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def set_plot_style(figsize=(12, 8), font_size=12):
    """
    Set global plotting style.
    
    Parameters:
    -----------
    figsize : tuple
        Default figure size
    font_size : int
        Default font size
    """
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_data_overview(df, title="Dataset Overview"):
    """
    Create an overview visualization of the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to visualize
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Data types distribution
    ax1 = axes[0, 0]
    dtype_counts = df.dtypes.value_counts()
    ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    ax1.set_title('Data Types Distribution', fontweight='bold')
    
    # 2. Missing values
    ax2 = axes[0, 1]
    missing_data = df.isnull().sum().sort_values(ascending=False)
    if missing_data.sum() > 0:
        missing_data = missing_data[missing_data > 0][:10]  # Top 10
        ax2.barh(range(len(missing_data)), missing_data.values)
        ax2.set_yticks(range(len(missing_data)))
        ax2.set_yticklabels(missing_data.index)
        ax2.set_xlabel('Missing Values Count')
        ax2.set_title('Missing Values by Column', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16, fontweight='bold')
        ax2.set_title('Missing Values Analysis', fontweight='bold')
    
    # 3. Numerical columns distribution
    ax3 = axes[1, 0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ax3.bar(range(len(numeric_cols)), [1]*len(numeric_cols))
    ax3.set_xticks(range(len(numeric_cols)))
    ax3.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Numerical Columns ({len(numeric_cols)})', fontweight='bold')
    
    # 4. Dataset shape info
    ax4 = axes[1, 1]
    info_text = f"""
    Rows: {df.shape[0]:,}
    Columns: {df.shape[1]:,}
    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
    Duplicates: {df.duplicated().sum():,}
    """
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=14, 
            verticalalignment='center')
    ax4.set_title('Dataset Statistics', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_distribution_analysis(df, features, title="Distribution Analysis"):
    """
    Plot distribution analysis for selected features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    features : list
        List of features to analyze
    title : str
        Plot title
    """
    # Filter existing features
    existing_features = [f for f in features if f in df.columns]
    
    if not existing_features:
        print("❌ No valid features found for distribution analysis")
        return
    
    n_features = len(existing_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(existing_features):
        ax = axes[i]
        data = df[feature].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Add KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
        except:
            pass
        
        # Add statistics lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
        
        # Statistics (using pandas methods instead of scipy)
        skewness = data.skew()
        kurtosis_val = data.kurtosis()
        
        ax.set_title(f'{feature}\\nSkew: {skewness:.2f}, Kurtosis: {kurtosis_val:.2f}', 
                    fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    print(f"\\n📊 DISTRIBUTION STATISTICS:")
    for feature in existing_features:
        data = df[feature].dropna()
        print(f"{feature}: Mean={data.mean():.2f}, Std={data.std():.2f}, "
              f"Skew={skew(data):.2f}, Kurt={kurtosis(data):.2f}")

def plot_boxplots_outliers(df, features, title="Outlier Analysis"):
    """
    Create box plots for outlier detection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    features : list
        List of features to analyze
    title : str
        Plot title
    """
    existing_features = [f for f in features if f in df.columns]
    
    if not existing_features:
        print("❌ No valid features found for outlier analysis")
        return
    
    n_features = len(existing_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    outlier_summary = {}
    
    for i, feature in enumerate(existing_features):
        ax = axes[i]
        data = df[feature].dropna()
        
        # Box plot
        bp = ax.boxplot(data, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Calculate outliers using IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_percentage = len(outliers) / len(data) * 100
        
        outlier_summary[feature] = {
            'count': len(outliers),
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        ax.set_title(f'{feature}\\nOutliers: {len(outliers)} ({outlier_percentage:.1f}%)', 
                    fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print outlier summary
    print(f"\\n📦 OUTLIER SUMMARY:")
    for feature, stats in outlier_summary.items():
        print(f"{feature}: {stats['count']} outliers ({stats['percentage']:.1f}%) "
              f"[Bounds: {stats['lower_bound']:.2f} - {stats['upper_bound']:.2f}]")
    
    return outlier_summary

def plot_correlation_heatmap(df, features=None, title="Correlation Analysis"):
    """
    Create correlation heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    features : list, optional
        List of features to include in correlation analysis
    title : str
        Plot title
    """
    if features is None:
        # Use all numerical columns
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in [np.number, 'float64', 'int64']]
    
    if len(numeric_features) < 2:
        print("❌ Need at least 2 numerical features for correlation analysis")
        return
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0, 
                mask=mask,
                square=True, 
                fmt='.2f',
                cbar_kws={"shrink": 0.8},
                linewidths=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Find and print highest correlations
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    high_corr = []
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            corr_val = upper_triangle.loc[row, col]
            if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                high_corr.append((row, col, corr_val))
    
    # Sort by absolute correlation value
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\\n🔗 HIGHEST CORRELATIONS (|r| > 0.5):")
    for i, (var1, var2, corr) in enumerate(high_corr[:10]):
        print(f"{i+1:2d}. {var1} <-> {var2}: {corr:.3f}")
    
    return correlation_matrix

def plot_bivariate_analysis(df, feature_pairs, title="Bivariate Analysis"):
    """
    Create scatter plots for bivariate analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    feature_pairs : list of tuples
        List of (x_feature, y_feature) pairs
    title : str
        Plot title
    """
    n_pairs = len(feature_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (x_feature, y_feature) in enumerate(feature_pairs):
        if x_feature not in df.columns or y_feature not in df.columns:
            print(f"⚠️ Warning: Features {x_feature} or {y_feature} not found")
            continue
        
        ax = axes[i]
        
        # Scatter plot
        scatter = ax.scatter(df[x_feature], df[y_feature], 
                           alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        try:
            z = np.polyfit(df[x_feature].dropna(), df[y_feature].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[x_feature], p(df[x_feature]), "r--", alpha=0.8, linewidth=2)
        except:
            pass
        
        # Calculate correlation
        correlation = df[x_feature].corr(df[y_feature])
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'{y_feature} vs {x_feature}\\nCorrelation: {correlation:.3f}', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_feature_comparison(df, features, groupby_col, title="Feature Comparison"):
    """
    Compare features across different groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    features : list
        List of features to compare
    groupby_col : str
        Column to group by
    title : str
        Plot title
    """
    if groupby_col not in df.columns:
        print(f"❌ Groupby column '{groupby_col}' not found")
        return
    
    existing_features = [f for f in features if f in df.columns]
    
    if not existing_features:
        print("❌ No valid features found for comparison")
        return
    
    n_features = len(existing_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(existing_features):
        ax = axes[i]
        
        # Group by and calculate means
        grouped_data = df.groupby(groupby_col)[feature].mean().sort_values(ascending=False)
        
        bars = ax.bar(range(len(grouped_data)), grouped_data.values, alpha=0.8)
        ax.set_xticks(range(len(grouped_data)))
        ax.set_xticklabels(grouped_data.index, rotation=45, ha='right')
        ax.set_ylabel(f'Average {feature}')
        ax.set_title(f'Average {feature} by {groupby_col}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, grouped_data.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(grouped_data.values),
                   f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def create_interactive_scatter(df, x_col, y_col, color_col=None, size_col=None, 
                             hover_col=None, title="Interactive Scatter Plot"):
    """
    Create interactive scatter plot using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    x_col : str
        X-axis feature
    y_col : str
        Y-axis feature
    color_col : str, optional
        Feature for color coding
    size_col : str, optional
        Feature for size coding
    hover_col : str, optional
        Feature to show on hover
    title : str
        Plot title
    """
    # Check if columns exist
    required_cols = [x_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return
    
    # Create scatter plot
    fig = px.scatter(df, x=x_col, y=y_col, 
                     color=color_col if color_col in df.columns else None,
                     size=size_col if size_col in df.columns else None,
                     hover_name=hover_col if hover_col in df.columns else None,
                     title=title,
                     width=800, height=600)
    
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    
    fig.show()
    
    return fig

def generate_eda_report(df, features=None, save_plots=False, output_dir=None):
    """
    Generate comprehensive EDA report.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    features : list, optional
        List of features to analyze
    save_plots : bool
        Whether to save plots
    output_dir : str
        Directory to save plots
    """
    print("📊 GENERATING COMPREHENSIVE EDA REPORT")
    print("="*60)
    
    # Dataset overview
    print("\\n1. Dataset Overview")
    plot_data_overview(df, "Dataset Overview")
    
    # Feature selection
    if features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = numeric_features[:8]  # Limit to first 8 for manageable output
    
    existing_features = [f for f in features if f in df.columns]
    
    if not existing_features:
        print("❌ No valid numerical features found")
        return
    
    # Distribution analysis
    print("\\n2. Distribution Analysis")
    plot_distribution_analysis(df, existing_features, "Feature Distributions")
    
    # Outlier analysis
    print("\\n3. Outlier Analysis")
    outlier_summary = plot_boxplots_outliers(df, existing_features, "Outlier Detection")
    
    # Correlation analysis
    print("\\n4. Correlation Analysis")
    corr_matrix = plot_correlation_heatmap(df, existing_features, "Feature Correlations")
    
    # Statistical summary
    print("\\n5. Statistical Summary")
    stats_df = df[existing_features].describe()
    print(stats_df.round(3))
    
    # Feature insights
    print("\\n6. Feature Insights")
    for feature in existing_features:
        data = df[feature].dropna()
        print(f"\\n{feature}:")
        print(f"  • Range: {data.min():.2f} - {data.max():.2f}")
        print(f"  • IQR: {data.quantile(0.75) - data.quantile(0.25):.2f}")
        print(f"  • Skewness: {skew(data):.3f} ({'Right-skewed' if skew(data) > 0.5 else 'Left-skewed' if skew(data) < -0.5 else 'Approximately symmetric'})")
        print(f"  • Missing values: {df[feature].isnull().sum()} ({df[feature].isnull().mean()*100:.1f}%)")
    
    print(f"\\n✅ EDA Report completed for {len(existing_features)} features")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing EDA visualization functions...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Feature1': np.random.normal(100, 15, 1000),
        'Feature2': np.random.exponential(2, 1000),
        'Feature3': np.random.uniform(0, 100, 1000),
        'Feature4': np.random.gamma(2, 2, 1000),
        'Category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some missing values
    sample_data.loc[sample_data.sample(50).index, 'Feature1'] = np.nan
    
    print("✅ Sample data created")
    
    # Test functions
    features_to_test = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    
    print("\\n1. Testing data overview...")
    plot_data_overview(sample_data)
    
    print("\\n2. Testing distribution analysis...")
    plot_distribution_analysis(sample_data, features_to_test)
    
    print("\\n3. Testing correlation heatmap...")
    plot_correlation_heatmap(sample_data, features_to_test)
    
    print("\\n🎉 All visualization tests completed!")