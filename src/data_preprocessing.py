"""
Data Preprocessing Module for India Census 2011 EDA & Clustering Project

This module contains functions for loading, cleaning, and preprocessing 
the India Census 2011 dataset.

Author: Data Science Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """
    Load the India Census 2011 dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📏 Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def filter_data_by_level(df, level='STATE', tru='Total'):
    """
    Filter dataset by administrative level and area type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    level : str
        Administrative level ('STATE', 'DISTRICT', etc.)
    tru : str
        Area type ('Total', 'Rural', 'Urban')
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataset
    """
    if 'Level' not in df.columns or 'TRU' not in df.columns:
        print("⚠️ Warning: Required columns 'Level' or 'TRU' not found")
        return df
    
    filtered_df = df[(df['Level'] == level) & (df['TRU'] == tru)].copy()
    print(f"📊 Filtered {level}-level {tru} records: {len(filtered_df)}")
    return filtered_df

def clean_data(df):
    """
    Clean the dataset by removing irrelevant columns and handling missing values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()
    
    # Remove administrative hierarchy columns (keep Name for identification)
    cols_to_remove = ['State', 'District', 'Subdistt', 'Town/Village', 'Ward', 'EB', 'Level', 'TRU']
    existing_cols_to_remove = [col for col in cols_to_remove if col in df_clean.columns]
    
    if existing_cols_to_remove:
        df_clean = df_clean.drop(columns=existing_cols_to_remove, errors='ignore')
        print(f"🗑️ Removed columns: {existing_cols_to_remove}")
    
    # Rename 'Name' to more specific identifier
    if 'Name' in df_clean.columns:
        df_clean['State_Name'] = df_clean['Name']
        df_clean = df_clean.drop('Name', axis=1)
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    if duplicates_removed > 0:
        print(f"🔄 Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_values = df_clean.isnull().sum().sum()
    if missing_values > 0:
        print(f"⚠️ Found {missing_values} missing values")
        # For numerical columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    print(f"✅ Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean

def create_features(df):
    """
    Create meaningful features from raw census data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with engineered features
    """
    df_features = df.copy()
    
    print("🔧 Creating features...")
    
    # 1. Sex Ratio (Females per 1000 Males)
    if 'TOT_F' in df_features.columns and 'TOT_M' in df_features.columns:
        df_features['Sex_Ratio'] = (df_features['TOT_F'] / df_features['TOT_M']) * 1000
    
    # 2. Literacy Rate (%)
    if 'P_LIT' in df_features.columns and 'TOT_P' in df_features.columns:
        df_features['Literacy_Rate'] = (df_features['P_LIT'] / df_features['TOT_P']) * 100
    
    # 3. Male Literacy Rate (%)
    if 'M_LIT' in df_features.columns and 'TOT_M' in df_features.columns:
        df_features['Male_Literacy_Rate'] = (df_features['M_LIT'] / df_features['TOT_M']) * 100
    
    # 4. Female Literacy Rate (%)
    if 'F_LIT' in df_features.columns and 'TOT_F' in df_features.columns:
        df_features['Female_Literacy_Rate'] = (df_features['F_LIT'] / df_features['TOT_F']) * 100
    
    # 5. Child Population Ratio (0-6 years)
    if 'P_06' in df_features.columns and 'TOT_P' in df_features.columns:
        df_features['Child_Population_Ratio'] = (df_features['P_06'] / df_features['TOT_P']) * 100
    
    # 6. Work Participation Rate
    if 'TOT_WORK_P' in df_features.columns and 'TOT_P' in df_features.columns:
        df_features['Work_Participation_Rate'] = (df_features['TOT_WORK_P'] / df_features['TOT_P']) * 100
    
    # 7. Male Work Participation Rate
    if 'TOT_WORK_M' in df_features.columns and 'TOT_M' in df_features.columns:
        df_features['Male_Work_Rate'] = (df_features['TOT_WORK_M'] / df_features['TOT_M']) * 100
    
    # 8. Female Work Participation Rate
    if 'TOT_WORK_F' in df_features.columns and 'TOT_F' in df_features.columns:
        df_features['Female_Work_Rate'] = (df_features['TOT_WORK_F'] / df_features['TOT_F']) * 100
    
    # 9. SC Population Percentage
    if 'P_SC' in df_features.columns and 'TOT_P' in df_features.columns:
        df_features['SC_Population_Percent'] = (df_features['P_SC'] / df_features['TOT_P']) * 100
    
    # 10. ST Population Percentage
    if 'P_ST' in df_features.columns and 'TOT_P' in df_features.columns:
        df_features['ST_Population_Percent'] = (df_features['P_ST'] / df_features['TOT_P']) * 100
    
    # 11. Dependency Ratio (Non-working population dependency)
    if 'NON_WORK_P' in df_features.columns and 'TOT_WORK_P' in df_features.columns:
        # Avoid division by zero
        work_mask = df_features['TOT_WORK_P'] > 0
        df_features.loc[work_mask, 'Dependency_Ratio'] = (
            df_features.loc[work_mask, 'NON_WORK_P'] / df_features.loc[work_mask, 'TOT_WORK_P']
        )
    
    # 12. Gender Literacy Gap
    if 'Male_Literacy_Rate' in df_features.columns and 'Female_Literacy_Rate' in df_features.columns:
        df_features['Gender_Literacy_Gap'] = df_features['Male_Literacy_Rate'] - df_features['Female_Literacy_Rate']
    
    # 13. Gender Work Gap
    if 'Male_Work_Rate' in df_features.columns and 'Female_Work_Rate' in df_features.columns:
        df_features['Gender_Work_Gap'] = df_features['Male_Work_Rate'] - df_features['Female_Work_Rate']
    
    # Count newly created features
    new_features = ['Sex_Ratio', 'Literacy_Rate', 'Male_Literacy_Rate', 'Female_Literacy_Rate', 
                   'Child_Population_Ratio', 'Work_Participation_Rate', 'Male_Work_Rate', 
                   'Female_Work_Rate', 'SC_Population_Percent', 'ST_Population_Percent',
                   'Dependency_Ratio', 'Gender_Literacy_Gap', 'Gender_Work_Gap']
    
    existing_new_features = [col for col in new_features if col in df_features.columns]
    print(f"✅ Created {len(existing_new_features)} new features: {existing_new_features}")
    
    return df_features

def prepare_clustering_data(df, clustering_features=None):
    """
    Prepare data for clustering analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features
    clustering_features : list
        List of features to use for clustering
        
    Returns:
    --------
    pandas.DataFrame, list
        Cleaned clustering data and feature names
    """
    if clustering_features is None:
        clustering_features = [
            'Literacy_Rate', 'Sex_Ratio', 'Work_Participation_Rate', 
            'Child_Population_Ratio', 'SC_Population_Percent', 'ST_Population_Percent',
            'Gender_Literacy_Gap', 'TOT_P'
        ]
    
    # Filter existing features
    available_features = [col for col in clustering_features if col in df.columns]
    
    if not available_features:
        print("❌ No clustering features found in dataset")
        return None, []
    
    # Create clustering dataset
    cluster_cols = available_features + ['State_Name']
    cluster_data = df[cluster_cols].copy()
    
    # Remove rows with missing values
    initial_rows = len(cluster_data)
    cluster_data_clean = cluster_data.dropna()
    rows_removed = initial_rows - len(cluster_data_clean)
    
    if rows_removed > 0:
        print(f"⚠️ Removed {rows_removed} rows with missing values")
    
    print(f"✅ Clustering data prepared. Shape: {cluster_data_clean.shape}")
    print(f"Features: {available_features}")
    
    return cluster_data_clean, available_features

def standardize_features(data, feature_columns):
    """
    Standardize features for clustering.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    feature_columns : list
        List of feature columns to standardize
        
    Returns:
    --------
    numpy.ndarray, StandardScaler
        Standardized features and fitted scaler
    """
    scaler = StandardScaler()
    X = data[feature_columns]
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Features standardized. Shape: {X_scaled.shape}")
    print(f"Mean after scaling: {X_scaled.mean(axis=0).round(6)}")
    print(f"Std after scaling: {X_scaled.std(axis=0).round(6)}")
    
    return X_scaled, scaler

def save_processed_data(df, file_path):
    """
    Save processed data to CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to save
    file_path : str
        Output file path
        
    Returns:
    --------
    bool
        Success status
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ Data saved to: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

def get_data_summary(df):
    """
    Get comprehensive summary of the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': len(df.columns),
        'rows': len(df),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Add numerical summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numerical_features'] = len(numeric_cols)
        summary['numerical_summary'] = df[numeric_cols].describe().to_dict()
    
    return summary

# Example usage and testing
if __name__ == "__main__":
    # This will run when the script is executed directly
    print("🧪 Testing data preprocessing functions...")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'Level': ['STATE'] * 5,
        'TRU': ['Total'] * 5,
        'Name': ['State1', 'State2', 'State3', 'State4', 'State5'],
        'TOT_P': [1000000, 2000000, 1500000, 800000, 1200000],
        'TOT_M': [520000, 1040000, 780000, 416000, 624000],
        'TOT_F': [480000, 960000, 720000, 384000, 576000],
        'P_LIT': [750000, 1600000, 1200000, 560000, 960000],
        'TOT_WORK_P': [400000, 800000, 600000, 320000, 480000],
        'P_SC': [150000, 300000, 225000, 120000, 180000],
        'P_ST': [50000, 100000, 75000, 40000, 60000]
    })
    
    print("Sample data created successfully ✅")
    
    # Test filtering
    filtered_data = filter_data_by_level(sample_data)
    print(f"Filtered data shape: {filtered_data.shape}")
    
    # Test cleaning
    cleaned_data = clean_data(filtered_data)
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Test feature creation
    feature_data = create_features(cleaned_data)
    print(f"Feature data shape: {feature_data.shape}")
    
    # Test clustering preparation
    cluster_data, features = prepare_clustering_data(feature_data)
    print(f"Cluster data shape: {cluster_data.shape if cluster_data is not None else 'None'}")
    
    print("🎉 All tests completed successfully!")