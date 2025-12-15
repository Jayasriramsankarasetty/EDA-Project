"""
India Census 2011 - EDA & Clustering Dashboard
Clean, Simple, and Fully Working Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# Page Configuration
st.set_page_config(
    page_title="India Census 2011 - EDA Dashboard",
    page_icon="🇮🇳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A5F;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        color: #1E3A5F;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============== DATA LOADING ==============
@st.cache_data
def load_data():
    """Load and prepare the census dataset."""
    # Get the directory where app.py is located
    current_dir = Path(__file__).resolve().parent
    # Go up to project root and into data folder
    data_path = current_dir.parent / "data" / "2011-IndiaStateDistSbDist-0000.xlsx - Data.csv"
    
    # Fallback paths if the above doesn't work
    if not data_path.exists():
        data_path = Path("d:/EDA-Project/data/2011-IndiaStateDistSbDist-0000.xlsx - Data.csv")
    if not data_path.exists():
        data_path = Path("../data/2011-IndiaStateDistSbDist-0000.xlsx - Data.csv")
    
    df = pd.read_csv(str(data_path))
    return df

@st.cache_data
def prepare_state_data(df):
    """Prepare state-level aggregated data with engineered features."""
    # Filter state-level total data
    df_state = df[(df['Level'] == 'STATE') & (df['TRU'] == 'Total')].copy()
    
    # Create derived features
    if 'TOT_F' in df_state.columns and 'TOT_M' in df_state.columns:
        df_state['Sex_Ratio'] = (df_state['TOT_F'] / df_state['TOT_M']) * 1000
    
    if 'P_LIT' in df_state.columns and 'TOT_P' in df_state.columns:
        df_state['Literacy_Rate'] = (df_state['P_LIT'] / df_state['TOT_P']) * 100
    
    if 'M_LIT' in df_state.columns and 'TOT_M' in df_state.columns:
        df_state['Male_Literacy_Rate'] = (df_state['M_LIT'] / df_state['TOT_M']) * 100
    
    if 'F_LIT' in df_state.columns and 'TOT_F' in df_state.columns:
        df_state['Female_Literacy_Rate'] = (df_state['F_LIT'] / df_state['TOT_F']) * 100
    
    if 'P_06' in df_state.columns and 'TOT_P' in df_state.columns:
        df_state['Child_Population_Ratio'] = (df_state['P_06'] / df_state['TOT_P']) * 100
    
    if 'TOT_WORK_P' in df_state.columns and 'TOT_P' in df_state.columns:
        df_state['Work_Participation_Rate'] = (df_state['TOT_WORK_P'] / df_state['TOT_P']) * 100
    
    if 'P_SC' in df_state.columns and 'TOT_P' in df_state.columns:
        df_state['SC_Population_Percent'] = (df_state['P_SC'] / df_state['TOT_P']) * 100
    
    if 'P_ST' in df_state.columns and 'TOT_P' in df_state.columns:
        df_state['ST_Population_Percent'] = (df_state['P_ST'] / df_state['TOT_P']) * 100
    
    # Gender gaps
    if 'Male_Literacy_Rate' in df_state.columns and 'Female_Literacy_Rate' in df_state.columns:
        df_state['Gender_Literacy_Gap'] = df_state['Male_Literacy_Rate'] - df_state['Female_Literacy_Rate']
    
    return df_state

@st.cache_data
def prepare_district_data(df):
    """Prepare district-level data."""
    df_district = df[(df['Level'] == 'DISTRICT') & (df['TRU'] == 'Total')].copy()
    
    # Create derived features
    if 'TOT_F' in df_district.columns and 'TOT_M' in df_district.columns:
        df_district['Sex_Ratio'] = (df_district['TOT_F'] / df_district['TOT_M']) * 1000
    
    if 'P_LIT' in df_district.columns and 'TOT_P' in df_district.columns:
        df_district['Literacy_Rate'] = (df_district['P_LIT'] / df_district['TOT_P']) * 100
    
    if 'TOT_WORK_P' in df_district.columns and 'TOT_P' in df_district.columns:
        df_district['Work_Participation_Rate'] = (df_district['TOT_WORK_P'] / df_district['TOT_P']) * 100
    
    return df_district

# ============== MAIN APP ==============
def main():
    # Header
    st.markdown('<h1 class="main-title">🇮🇳 India Census 2011 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Exploratory Data Analysis & Demographic Insights</p>', unsafe_allow_html=True)
    
    # Load Data
    try:
        df = load_data()
        df_state = prepare_state_data(df)
        df_district = prepare_district_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["🏠 Overview", "📈 Distribution Analysis", "🔗 Correlation Analysis", 
         "🗺️ State Comparison", "📊 Deep EDA", "🤖 ML Clustering", "📋 Data Explorer"]
    )
    
    # ============== PAGE: OVERVIEW ==============
    if page == "🏠 Overview":
        st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pop = df[df['Level'] == 'India']['TOT_P'].values[0] if len(df[df['Level'] == 'India']) > 0 else df_state['TOT_P'].sum()
            st.metric("🇮🇳 Total Population", f"{total_pop/1e9:.2f} Billion")
        
        with col2:
            st.metric("🏛️ States/UTs", len(df_state))
        
        with col3:
            st.metric("🏘️ Districts", len(df_district))
        
        with col4:
            avg_literacy = df_state['Literacy_Rate'].mean() if 'Literacy_Rate' in df_state.columns else 0
            st.metric("📚 Avg Literacy Rate", f"{avg_literacy:.1f}%")
        
        st.divider()
        
        # Quick Stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Information")
            st.write(f"**Total Records:** {len(df):,}")
            st.write(f"**Total Columns:** {len(df.columns)}")
            st.write(f"**Administrative Levels:** {df['Level'].nunique()}")
            
            # Level distribution
            level_counts = df['Level'].value_counts()
            fig = px.pie(values=level_counts.values, names=level_counts.index, 
                        title="Records by Administrative Level")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏛️ Top 10 States by Population")
            top_states = df_state.nlargest(10, 'TOT_P')[['Name', 'TOT_P']].copy()
            top_states['Population (Millions)'] = top_states['TOT_P'] / 1e6
            
            fig = px.bar(top_states, x='Name', y='Population (Millions)',
                        color='Population (Millions)', color_continuous_scale='Blues')
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # ============== PAGE: DISTRIBUTION ANALYSIS ==============
    elif page == "📈 Distribution Analysis":
        st.markdown('<h2 class="section-header">Distribution Analysis</h2>', unsafe_allow_html=True)
        
        # Feature Selection
        numeric_cols = ['Sex_Ratio', 'Literacy_Rate', 'Male_Literacy_Rate', 'Female_Literacy_Rate',
                       'Child_Population_Ratio', 'Work_Participation_Rate', 'SC_Population_Percent', 
                       'ST_Population_Percent', 'Gender_Literacy_Gap', 'TOT_P']
        available_cols = [c for c in numeric_cols if c in df_state.columns]
        
        selected_feature = st.selectbox("Select Feature to Analyze", available_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            st.subheader(f"📊 Distribution of {selected_feature}")
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df_state[selected_feature].dropna()
            
            ax.hist(data, bins=15, color='steelblue', edgecolor='white', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Box Plot
            st.subheader(f"📦 Box Plot of {selected_feature}")
            fig = px.box(df_state, y=selected_feature, points="all",
                        hover_data=['Name'] if 'Name' in df_state.columns else None)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Descriptive Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{data.mean():.2f}")
            st.metric("Std Dev", f"{data.std():.2f}")
        with col2:
            st.metric("Median", f"{data.median():.2f}")
            st.metric("Range", f"{data.max() - data.min():.2f}")
        with col3:
            st.metric("Min", f"{data.min():.2f}")
            st.metric("Max", f"{data.max():.2f}")
        with col4:
            st.metric("Skewness", f"{data.skew():.3f}")
            st.metric("Kurtosis", f"{data.kurtosis():.3f}")
    
    # ============== PAGE: CORRELATION ANALYSIS ==============
    elif page == "🔗 Correlation Analysis":
        st.markdown('<h2 class="section-header">Correlation Analysis</h2>', unsafe_allow_html=True)
        
        # Select features for correlation
        numeric_cols = ['Sex_Ratio', 'Literacy_Rate', 'Male_Literacy_Rate', 'Female_Literacy_Rate',
                       'Child_Population_Ratio', 'Work_Participation_Rate', 'SC_Population_Percent', 
                       'ST_Population_Percent', 'Gender_Literacy_Gap']
        available_cols = [c for c in numeric_cols if c in df_state.columns]
        
        selected_features = st.multiselect("Select Features", available_cols, default=available_cols[:6])
        
        if len(selected_features) >= 2:
            # Correlation Heatmap
            st.subheader("🔥 Correlation Heatmap")
            corr_matrix = df_state[selected_features].corr()
            
            fig = px.imshow(corr_matrix, 
                           labels=dict(color="Correlation"),
                           x=selected_features, y=selected_features,
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1,
                           text_auto='.2f')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter Plot
            st.subheader("📍 Scatter Plot Analysis")
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X-axis", selected_features, index=0)
            with col2:
                y_var = st.selectbox("Y-axis", selected_features, index=1 if len(selected_features) > 1 else 0)
            
            fig = px.scatter(df_state, x=x_var, y=y_var, 
                            hover_data=['Name'] if 'Name' in df_state.columns else None,
                            color='Literacy_Rate' if 'Literacy_Rate' in df_state.columns else None,
                            size='TOT_P' if 'TOT_P' in df_state.columns else None,
                            trendline='ols')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            corr_val = df_state[x_var].corr(df_state[y_var])
            st.info(f"**Correlation between {x_var} and {y_var}:** {corr_val:.3f}")
        else:
            st.warning("Please select at least 2 features for correlation analysis.")
    
    # ============== PAGE: STATE COMPARISON ==============
    elif page == "🗺️ State Comparison":
        st.markdown('<h2 class="section-header">State-wise Comparison</h2>', unsafe_allow_html=True)
        
        # Feature selection
        compare_features = ['Literacy_Rate', 'Sex_Ratio', 'Work_Participation_Rate', 
                           'Child_Population_Ratio', 'SC_Population_Percent', 'ST_Population_Percent']
        available_features = [f for f in compare_features if f in df_state.columns]
        
        selected_metric = st.selectbox("Select Metric to Compare", available_features)
        
        # Bar chart - All states
        st.subheader(f"📊 {selected_metric} Across All States")
        df_sorted = df_state.sort_values(selected_metric, ascending=True)
        
        fig = px.bar(df_sorted, x=selected_metric, y='Name', orientation='h',
                    color=selected_metric, color_continuous_scale='Viridis')
        fig.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top and Bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 5 States")
            top5 = df_state.nlargest(5, selected_metric)[['Name', selected_metric]]
            st.dataframe(top5, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⚠️ Bottom 5 States")
            bottom5 = df_state.nsmallest(5, selected_metric)[['Name', selected_metric]]
            st.dataframe(bottom5, use_container_width=True, hide_index=True)
    
    # ============== PAGE: DEEP EDA ==============
    elif page == "📊 Deep EDA":
        st.markdown('<h2 class="section-header">Deep Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox("Select Analysis Type", 
                                     ["Univariate Analysis", "Bivariate Analysis", 
                                      "Multivariate Analysis", "Outlier Detection"])
        
        if analysis_type == "Univariate Analysis":
            st.subheader("📈 Univariate Analysis - All Key Features")
            
            features = ['Literacy_Rate', 'Sex_Ratio', 'Work_Participation_Rate', 'Child_Population_Ratio']
            available = [f for f in features if f in df_state.columns]
            
            # Create grid of distributions
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(available[:4]):
                data = df_state[feature].dropna()
                axes[i].hist(data, bins=12, color='steelblue', edgecolor='white', alpha=0.7)
                axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
                axes[i].set_title(f'{feature}', fontweight='bold')
                axes[i].set_xlabel(feature)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Summary Statistics Table
            st.subheader("📋 Summary Statistics")
            stats_df = df_state[available].describe().T
            stats_df['skewness'] = df_state[available].skew()
            stats_df['kurtosis'] = df_state[available].kurtosis()
            st.dataframe(stats_df.round(2), use_container_width=True)
        
        elif analysis_type == "Bivariate Analysis":
            st.subheader("🔗 Bivariate Relationships")
            
            col1, col2 = st.columns(2)
            
            # Literacy vs Sex Ratio
            with col1:
                if 'Literacy_Rate' in df_state.columns and 'Sex_Ratio' in df_state.columns:
                    fig = px.scatter(df_state, x='Sex_Ratio', y='Literacy_Rate',
                                    hover_data=['Name'], trendline='ols',
                                    title="Literacy Rate vs Sex Ratio")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Work Participation vs Literacy
            with col2:
                if 'Literacy_Rate' in df_state.columns and 'Work_Participation_Rate' in df_state.columns:
                    fig = px.scatter(df_state, x='Literacy_Rate', y='Work_Participation_Rate',
                                    hover_data=['Name'], trendline='ols',
                                    title="Work Participation vs Literacy Rate")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Gender Literacy Gap Analysis
            if 'Male_Literacy_Rate' in df_state.columns and 'Female_Literacy_Rate' in df_state.columns:
                st.subheader("👫 Gender Literacy Comparison")
                
                df_gender = df_state[['Name', 'Male_Literacy_Rate', 'Female_Literacy_Rate']].melt(
                    id_vars=['Name'], var_name='Gender', value_name='Literacy Rate'
                )
                df_gender['Gender'] = df_gender['Gender'].replace({
                    'Male_Literacy_Rate': 'Male', 'Female_Literacy_Rate': 'Female'
                })
                
                fig = px.bar(df_gender, x='Name', y='Literacy Rate', color='Gender',
                            barmode='group', title="Male vs Female Literacy Rate by State")
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Multivariate Analysis":
            st.subheader("🎯 Multivariate Analysis")
            
            # Pair Plot using Plotly
            features = ['Literacy_Rate', 'Sex_Ratio', 'Work_Participation_Rate', 'Child_Population_Ratio']
            available = [f for f in features if f in df_state.columns]
            
            if len(available) >= 2:
                fig = px.scatter_matrix(df_state, dimensions=available,
                                       hover_data=['Name'] if 'Name' in df_state.columns else None,
                                       title="Scatter Matrix of Key Features")
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            # Parallel Coordinates
            st.subheader("📊 Parallel Coordinates Plot")
            if len(available) >= 3:
                fig = px.parallel_coordinates(df_state, dimensions=available,
                                             color='Literacy_Rate' if 'Literacy_Rate' in df_state.columns else None,
                                             title="Parallel Coordinates - State Profiles")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Outlier Detection":
            st.subheader("🔍 Outlier Detection")
            
            features = ['Literacy_Rate', 'Sex_Ratio', 'Work_Participation_Rate', 'Child_Population_Ratio']
            available = [f for f in features if f in df_state.columns]
            
            selected = st.selectbox("Select Feature for Outlier Analysis", available)
            
            data = df_state[selected].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_state[(df_state[selected] < lower_bound) | (df_state[selected] > upper_bound)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                fig = px.box(df_state, y=selected, points="all",
                            hover_data=['Name'] if 'Name' in df_state.columns else None,
                            title=f"Box Plot: {selected}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write(f"**IQR Method Results:**")
                st.write(f"- Q1: {Q1:.2f}")
                st.write(f"- Q3: {Q3:.2f}")
                st.write(f"- IQR: {IQR:.2f}")
                st.write(f"- Lower Bound: {lower_bound:.2f}")
                st.write(f"- Upper Bound: {upper_bound:.2f}")
                st.write(f"- **Outliers Found: {len(outliers)}**")
                
                if len(outliers) > 0:
                    st.write("**Outlier States:**")
                    st.dataframe(outliers[['Name', selected]], use_container_width=True, hide_index=True)
    
    # ============== PAGE: ML CLUSTERING ==============
    elif page == "🤖 ML Clustering":
        st.markdown('<h2 class="section-header">Machine Learning - Clustering Analysis</h2>', unsafe_allow_html=True)
        
        # Feature Selection for Clustering
        st.subheader("🎯 Feature Selection")
        
        clustering_features = ['Literacy_Rate', 'Sex_Ratio', 'Work_Participation_Rate', 
                              'Child_Population_Ratio', 'SC_Population_Percent', 'ST_Population_Percent']
        available_features = [f for f in clustering_features if f in df_state.columns]
        
        selected_features = st.multiselect(
            "Select Features for Clustering",
            available_features,
            default=available_features[:4]
        )
        
        if len(selected_features) >= 2:
            # Prepare data for clustering
            cluster_data = df_state[['Name'] + selected_features].dropna()
            X = cluster_data[selected_features].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            st.divider()
            
            # Analysis Type Selection
            analysis_type = st.selectbox(
                "Select Analysis",
                ["🔍 Find Optimal Clusters", "📊 Run Clustering", "📈 Cluster Insights"]
            )
            
            if analysis_type == "🔍 Find Optimal Clusters":
                st.subheader("🔍 Optimal Number of Clusters")
                
                max_k = st.slider("Maximum K to test", 3, 10, 8)
                
                if st.button("Run Analysis", type="primary"):
                    k_range = range(2, max_k + 1)
                    inertias = []
                    silhouette_scores = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, k in enumerate(k_range):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(X_scaled)
                        inertias.append(kmeans.inertia_)
                        silhouette_scores.append(silhouette_score(X_scaled, labels))
                        progress_bar.progress((i + 1) / len(k_range))
                    
                    progress_bar.empty()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Elbow Method
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                                name='Inertia', line=dict(color='blue', width=2),
                                                marker=dict(size=10)))
                        fig.update_layout(title='Elbow Method', xaxis_title='Number of Clusters (k)',
                                         yaxis_title='Inertia', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Silhouette Score
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                                                name='Silhouette Score', line=dict(color='green', width=2),
                                                marker=dict(size=10)))
                        fig.update_layout(title='Silhouette Score', xaxis_title='Number of Clusters (k)',
                                         yaxis_title='Silhouette Score', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    optimal_k = k_range[np.argmax(silhouette_scores)]
                    st.success(f"**Recommended number of clusters: {optimal_k}** (highest silhouette score: {max(silhouette_scores):.3f})")
            
            elif analysis_type == "📊 Run Clustering":
                st.subheader("📊 K-Means Clustering")
                
                n_clusters = st.slider("Number of Clusters", 2, 8, 3)
                
                if st.button("Run Clustering", type="primary"):
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Add cluster labels to data
                    cluster_data['Cluster'] = cluster_labels
                    
                    # Calculate silhouette score
                    sil_score = silhouette_score(X_scaled, cluster_labels)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Clusters", n_clusters)
                    with col2:
                        st.metric("Silhouette Score", f"{sil_score:.3f}")
                    with col3:
                        st.metric("States Analyzed", len(cluster_data))
                    
                    st.divider()
                    
                    # PCA Visualization
                    st.subheader("🎨 Cluster Visualization (PCA)")
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    cluster_data['PC1'] = X_pca[:, 0]
                    cluster_data['PC2'] = X_pca[:, 1]
                    
                    fig = px.scatter(cluster_data, x='PC1', y='PC2', color='Cluster',
                                    hover_data=['Name'] + selected_features,
                                    title=f'Clusters in PCA Space (Variance Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)',
                                    color_continuous_scale='viridis')
                    fig.update_traces(marker=dict(size=12))
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster Distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Cluster Distribution")
                        cluster_counts = cluster_data['Cluster'].value_counts().sort_index()
                        fig = px.pie(values=cluster_counts.values, names=[f'Cluster {i}' for i in cluster_counts.index],
                                    title='States per Cluster')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🏛️ States by Cluster")
                        for c in sorted(cluster_data['Cluster'].unique()):
                            states = cluster_data[cluster_data['Cluster'] == c]['Name'].tolist()
                            with st.expander(f"Cluster {c} ({len(states)} states)"):
                                st.write(", ".join(states))
                    
                    # Cluster Characteristics
                    st.subheader("📈 Cluster Characteristics")
                    cluster_means = cluster_data.groupby('Cluster')[selected_features].mean()
                    
                    fig = px.imshow(cluster_means.T, 
                                   labels=dict(x="Cluster", y="Feature", color="Mean Value"),
                                   x=[f'Cluster {i}' for i in cluster_means.index],
                                   y=selected_features,
                                   color_continuous_scale='RdYlBu_r',
                                   text_auto='.1f')
                    fig.update_layout(height=400, title='Average Feature Values by Cluster')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download Results
                    csv = cluster_data.to_csv(index=False)
                    st.download_button("📥 Download Cluster Results", csv, "cluster_results.csv", "text/csv")
            
            elif analysis_type == "📈 Cluster Insights":
                st.subheader("📈 Cluster Comparison & Insights")
                
                n_clusters = st.slider("Number of Clusters", 2, 8, 3, key="insight_clusters")
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                cluster_data['Cluster'] = cluster_labels
                
                # Feature comparison across clusters
                st.subheader("📊 Feature Comparison Across Clusters")
                
                selected_compare = st.selectbox("Select Feature to Compare", selected_features)
                
                fig = px.box(cluster_data, x='Cluster', y=selected_compare, 
                            color='Cluster', points='all',
                            hover_data=['Name'],
                            title=f'{selected_compare} Distribution by Cluster')
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                
                # Radar Chart for Cluster Profiles
                st.subheader("🎯 Cluster Profiles (Radar Chart)")
                
                # Normalize data for radar chart
                cluster_means = cluster_data.groupby('Cluster')[selected_features].mean()
                cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
                
                fig = go.Figure()
                colors = px.colors.qualitative.Set1[:n_clusters]
                
                for i, cluster in enumerate(cluster_means_norm.index):
                    values = cluster_means_norm.loc[cluster].tolist()
                    values.append(values[0])  # Close the radar
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=selected_features + [selected_features[0]],
                        fill='toself',
                        name=f'Cluster {cluster}',
                        line_color=colors[i]
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title='Normalized Cluster Profiles',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Statistics
                st.subheader("📋 Cluster Summary Statistics")
                summary_stats = cluster_data.groupby('Cluster')[selected_features].agg(['mean', 'std', 'min', 'max'])
                st.dataframe(summary_stats.round(2), use_container_width=True)
        
        else:
            st.warning("Please select at least 2 features for clustering analysis.")
    
    # ============== PAGE: DATA EXPLORER ==============
    elif page == "📋 Data Explorer":
        st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
        
        data_level = st.radio("Select Data Level", ["State Level", "District Level", "Raw Data"], horizontal=True)
        
        if data_level == "State Level":
            st.subheader("🏛️ State-Level Data")
            display_cols = ['Name', 'TOT_P', 'Sex_Ratio', 'Literacy_Rate', 'Work_Participation_Rate',
                          'SC_Population_Percent', 'ST_Population_Percent']
            available_display = [c for c in display_cols if c in df_state.columns]
            st.dataframe(df_state[available_display].round(2), use_container_width=True, hide_index=True)
            
            # Download button
            csv = df_state.to_csv(index=False)
            st.download_button("📥 Download State Data", csv, "state_data.csv", "text/csv")
        
        elif data_level == "District Level":
            st.subheader("🏘️ District-Level Data")
            
            display_cols = ['Name', 'TOT_P', 'Sex_Ratio', 'Literacy_Rate', 'Work_Participation_Rate']
            available_display = [c for c in display_cols if c in df_district.columns]
            
            st.dataframe(df_district[available_display].head(100).round(2), 
                        use_container_width=True, hide_index=True)
            st.info(f"Showing first 100 of {len(df_district)} districts")
        
        else:
            st.subheader("📄 Raw Dataset")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            st.info("Showing first 100 rows")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🇮🇳 India Census 2011 EDA Dashboard | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
