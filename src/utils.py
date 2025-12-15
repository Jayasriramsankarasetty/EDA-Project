"""
Utility Functions Module for India Census 2011 EDA & Clustering Project

This module contains utility functions for data handling, model operations,
and general helper functions used throughout the project.

Author: Data Science Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_project_directories(base_path):
    """
    Create project directory structure.
    
    Parameters:
    -----------
    base_path : str
        Base project path
        
    Returns:
    --------
    dict
        Dictionary of created directory paths
    """
    directories = {
        'data': os.path.join(base_path, 'data'),
        'notebooks': os.path.join(base_path, 'notebooks'),
        'src': os.path.join(base_path, 'src'),
        'outputs': os.path.join(base_path, 'outputs'),
        'dashboard': os.path.join(base_path, 'dashboard'),
        'docs': os.path.join(base_path, 'docs'),
        'plots': os.path.join(base_path, 'plots')
    }
    
    created_dirs = []
    
    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            created_dirs.append(name)
        except Exception as e:
            print(f"⚠️ Warning: Could not create {name} directory: {e}")
    
    print(f"✅ Created directories: {created_dirs}")
    return directories

def load_model(model_path):
    """
    Load a saved model using joblib.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def save_model(model, model_path):
    """
    Save a model using joblib.
    
    Parameters:
    -----------
    model : object
        Model to save
    model_path : str
        Path to save the model
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model saved to: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def save_dataframe(df, file_path, format='csv'):
    """
    Save DataFrame to file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    file_path : str
        Output file path
    format : str
        File format ('csv', 'excel', 'json', 'parquet')
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ DataFrame saved to: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving DataFrame: {e}")
        return False

def load_dataframe(file_path, format='csv'):
    """
    Load DataFrame from file.
    
    Parameters:
    -----------
    file_path : str
        Input file path
    format : str
        File format ('csv', 'excel', 'json', 'parquet')
        
    Returns:
    --------
    pandas.DataFrame or None
        Loaded DataFrame
    """
    try:
        if format.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif format.lower() == 'excel':
            df = pd.read_excel(file_path)
        elif format.lower() == 'json':
            df = pd.read_json(file_path)
        elif format.lower() == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ DataFrame loaded from: {file_path}")
        print(f"   Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading DataFrame: {e}")
        return None

def calculate_statistics(data, feature_col):
    """
    Calculate comprehensive statistics for a feature.
    
    Parameters:
    -----------
    data : pandas.DataFrame or pandas.Series
        Input data
    feature_col : str or None
        Feature column name (None if data is Series)
        
    Returns:
    --------
    dict
        Statistical summary
    """
    if isinstance(data, pd.DataFrame):
        if feature_col not in data.columns:
            print(f"❌ Feature '{feature_col}' not found in data")
            return {}
        series = data[feature_col]
    else:
        series = data
    
    # Remove missing values
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return {'error': 'No valid data points'}
    
    # Calculate statistics
    stats = {
        'count': len(clean_data),
        'mean': clean_data.mean(),
        'median': clean_data.median(),
        'mode': clean_data.mode().iloc[0] if len(clean_data.mode()) > 0 else None,
        'std': clean_data.std(),
        'var': clean_data.var(),
        'min': clean_data.min(),
        'max': clean_data.max(),
        'range': clean_data.max() - clean_data.min(),
        'q1': clean_data.quantile(0.25),
        'q3': clean_data.quantile(0.75),
        'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25),
        'skewness': clean_data.skew(),
        'kurtosis': clean_data.kurtosis(),
        'missing_count': len(series) - len(clean_data),
        'missing_percentage': (len(series) - len(clean_data)) / len(series) * 100
    }
    
    return stats

def format_number(number, decimals=2, use_thousands_separator=True):
    """
    Format numbers for display.
    
    Parameters:
    -----------
    number : float
        Number to format
    decimals : int
        Number of decimal places
    use_thousands_separator : bool
        Whether to use thousands separator
        
    Returns:
    --------
    str
        Formatted number string
    """
    if pd.isna(number):
        return 'N/A'
    
    if use_thousands_separator:
        return f"{number:,.{decimals}f}"
    else:
        return f"{number:.{decimals}f}"

def create_summary_report(data_dict, title="Analysis Summary Report"):
    """
    Create a comprehensive summary report.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing analysis results
    title : str
        Report title
        
    Returns:
    --------
    str
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"{title.upper()}")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    for section_name, section_data in data_dict.items():
        report_lines.append(f"{section_name.upper()}")
        report_lines.append("-" * 30)
        
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"{key}: {format_number(value)}")
                else:
                    report_lines.append(f"{key}: {value}")
        elif isinstance(section_data, (list, tuple)):
            for item in section_data:
                report_lines.append(f"  • {item}")
        else:
            report_lines.append(str(section_data))
        
        report_lines.append("")
    
    return "\\n".join(report_lines)

def save_text_report(content, file_path):
    """
    Save text content to file.
    
    Parameters:
    -----------
    content : str
        Text content to save
    file_path : str
        Output file path
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Report saved to: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving report: {e}")
        return False

def validate_data_quality(df, critical_columns=None):
    """
    Validate data quality and return quality metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    critical_columns : list, optional
        List of critical columns that must have data
        
    Returns:
    --------
    dict
        Data quality metrics
    """
    print("🔍 VALIDATING DATA QUALITY")
    print("=" * 30)
    
    quality_metrics = {}
    
    # Basic metrics
    quality_metrics['total_rows'] = len(df)
    quality_metrics['total_columns'] = len(df.columns)
    quality_metrics['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # Missing value analysis
    missing_data = df.isnull().sum()
    quality_metrics['total_missing_values'] = missing_data.sum()
    quality_metrics['missing_percentage'] = (missing_data.sum() / (len(df) * len(df.columns))) * 100
    quality_metrics['columns_with_missing'] = len(missing_data[missing_data > 0])
    
    # Duplicate analysis
    quality_metrics['duplicate_rows'] = df.duplicated().sum()
    quality_metrics['duplicate_percentage'] = (df.duplicated().sum() / len(df)) * 100
    
    # Data type analysis
    dtype_counts = df.dtypes.value_counts()
    quality_metrics['data_types'] = dtype_counts.to_dict()
    
    # Critical column validation
    if critical_columns:
        critical_issues = []
        for col in critical_columns:
            if col not in df.columns:
                critical_issues.append(f"Missing column: {col}")
            elif df[col].isnull().sum() > 0:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                critical_issues.append(f"Missing values in {col}: {missing_pct:.1f}%")
        
        quality_metrics['critical_issues'] = critical_issues
    
    # Completeness score (0-100)
    completeness = ((len(df) * len(df.columns) - missing_data.sum()) / (len(df) * len(df.columns))) * 100
    quality_metrics['completeness_score'] = completeness
    
    # Overall quality score (simple heuristic)
    quality_score = completeness
    
    if quality_metrics['duplicate_percentage'] > 5:
        quality_score -= 10
    
    if critical_columns and quality_metrics.get('critical_issues'):
        quality_score -= len(quality_metrics['critical_issues']) * 5
    
    quality_metrics['overall_quality_score'] = max(0, min(100, quality_score))
    
    # Quality assessment
    if quality_score >= 90:
        quality_assessment = "Excellent"
    elif quality_score >= 80:
        quality_assessment = "Good"
    elif quality_score >= 70:
        quality_assessment = "Acceptable"
    elif quality_score >= 60:
        quality_assessment = "Poor"
    else:
        quality_assessment = "Very Poor"
    
    quality_metrics['quality_assessment'] = quality_assessment
    
    # Print summary
    print(f"Overall Quality Score: {quality_score:.1f}/100 ({quality_assessment})")
    print(f"Completeness: {completeness:.1f}%")
    print(f"Missing Values: {quality_metrics['total_missing_values']:,} ({quality_metrics['missing_percentage']:.1f}%)")
    print(f"Duplicates: {quality_metrics['duplicate_rows']:,} ({quality_metrics['duplicate_percentage']:.1f}%)")
    
    if critical_columns and quality_metrics.get('critical_issues'):
        print(f"Critical Issues: {len(quality_metrics['critical_issues'])}")
        for issue in quality_metrics['critical_issues']:
            print(f"  ⚠️ {issue}")
    
    return quality_metrics

def plot_feature_comparison(data, features, group_col=None, title="Feature Comparison"):
    """
    Create a comparison plot for multiple features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset
    features : list
        List of features to compare
    group_col : str, optional
        Column to group by
    title : str
        Plot title
    """
    # Filter existing features
    existing_features = [f for f in features if f in data.columns]
    
    if not existing_features:
        print("❌ No valid features found for comparison")
        return
    
    if group_col and group_col not in data.columns:
        print(f"❌ Group column '{group_col}' not found")
        group_col = None
    
    # Create comparison plot
    if group_col:
        # Grouped comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate means by group
        grouped_data = data.groupby(group_col)[existing_features].mean()
        
        # Create stacked bar plot
        grouped_data.T.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(f"{title} by {group_col}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Features")
        ax.set_ylabel("Average Value")
        ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
    else:
        # Simple feature comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate means
        means = [data[f].mean() for f in existing_features]
        
        bars = ax.bar(existing_features, means, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(means),
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel("Average Value")
        plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def export_results_to_json(results_dict, file_path):
    """
    Export results dictionary to JSON file.
    
    Parameters:
    -----------
    results_dict : dict
        Results to export
    file_path : str
        Output JSON file path
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_types(results_dict)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results exported to JSON: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error exporting to JSON: {e}")
        return False

def get_file_info(file_path):
    """
    Get information about a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    dict
        File information
    """
    try:
        stat_info = os.stat(file_path)
        
        info = {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'size_bytes': stat_info.st_size,
            'size_mb': stat_info.st_size / 1024**2,
            'created': datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified': datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'exists': True
        }
        
        return info
    except Exception as e:
        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'error': str(e),
            'exists': False
        }

def clean_column_names(df):
    """
    Clean column names by removing special characters and standardizing format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns to clean
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    
    # Store mapping of old to new names
    name_mapping = {}
    
    for old_name in df.columns:
        # Remove special characters and replace with underscores
        new_name = old_name.replace(' ', '_').replace('-', '_').replace('/', '_')
        new_name = ''.join(c if c.isalnum() or c == '_' else '' for c in new_name)
        
        # Remove multiple consecutive underscores
        while '__' in new_name:
            new_name = new_name.replace('__', '_')
        
        # Remove leading/trailing underscores
        new_name = new_name.strip('_')
        
        # Ensure it starts with letter or underscore
        if new_name and new_name[0].isdigit():
            new_name = f"col_{new_name}"
        
        # Handle empty names
        if not new_name:
            new_name = f"col_{list(df.columns).index(old_name)}"
        
        name_mapping[old_name] = new_name
    
    # Rename columns
    df_clean = df_clean.rename(columns=name_mapping)
    
    # Check for duplicates and handle them
    new_columns = list(df_clean.columns)
    seen = set()
    for i, col in enumerate(new_columns):
        original_col = col
        counter = 1
        while col in seen:
            col = f"{original_col}_{counter}"
            counter += 1
        seen.add(col)
        new_columns[i] = col
    
    df_clean.columns = new_columns
    
    print(f"✅ Cleaned {len(name_mapping)} column names")
    if len(name_mapping) <= 10:  # Show mapping for small datasets
        for old, new in name_mapping.items():
            if old != new:
                print(f"  '{old}' -> '{new}'")
    
    return df_clean

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing utility functions...")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'Feature 1': [1, 2, 3, np.nan, 5],
        'Feature-2': [10, 20, 30, 40, 50],
        'Feature/3': [100, 200, 300, 400, 500],
        'Category': ['A', 'B', 'A', 'C', 'B']
    })
    
    print("✅ Sample data created")
    
    # Test column name cleaning
    print("\\n1. Testing column name cleaning...")
    cleaned_data = clean_column_names(sample_data)
    print(f"Original columns: {list(sample_data.columns)}")
    print(f"Cleaned columns: {list(cleaned_data.columns)}")
    
    # Test statistics calculation
    print("\\n2. Testing statistics calculation...")
    stats = calculate_statistics(cleaned_data, 'Feature_1')
    print(f"Statistics for Feature_1: {stats}")
    
    # Test data quality validation
    print("\\n3. Testing data quality validation...")
    quality_metrics = validate_data_quality(cleaned_data, critical_columns=['Feature_1', 'Category'])
    
    # Test summary report creation
    print("\\n4. Testing summary report creation...")
    report_data = {
        'Dataset Info': {
            'rows': len(cleaned_data),
            'columns': len(cleaned_data.columns)
        },
        'Quality Metrics': quality_metrics
    }
    
    report = create_summary_report(report_data, "Test Analysis Report")
    print("Summary report created successfully")
    
    print("\\n🎉 All utility tests completed successfully!")