# 🇮🇳 India Census 2011 - EDA & Clustering Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-orange.svg)](https://scikit-learn.org)

A comprehensive **Exploratory Data Analysis (EDA)** and **Machine Learning Clustering** project on the India Census 2011 dataset. This project includes an interactive Streamlit dashboard for visualizing demographic patterns across Indian states and districts.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Features & Techniques](#-features--techniques)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dashboard Guide](#-dashboard-guide)
- [Technical Implementation](#-technical-implementation)
- [Results & Insights](#-results--insights)

---

## 🎯 Project Overview

This project performs detailed exploratory data analysis and unsupervised learning on the India Census 2011 dataset to:

- **Understand demographic patterns** across Indian states and districts
- **Identify key factors** influencing literacy, population growth, and social development
- **Apply clustering algorithms** to group similar regions based on demographic characteristics
- **Provide actionable insights** for policy-making and resource allocation
- **Visualize data interactively** through a modern Streamlit dashboard

---

## 📊 Dataset Description

### Source
- **Dataset**: India Census 2011
- **File**: `2011-IndiaStateDistSbDist-0000.xlsx - Data.csv`
- **Records**: 20,000+ records across multiple administrative levels

### Key Variables

| Category | Variables |
|----------|-----------|
| **Population** | `TOT_P` (Total), `TOT_M` (Male), `TOT_F` (Female), `P_06` (Children 0-6) |
| **Literacy** | `P_LIT` (Literate), `M_LIT` (Male Literate), `F_LIT` (Female Literate) |
| **Work Force** | `TOT_WORK_P`, `TOT_WORK_M`, `TOT_WORK_F`, `NON_WORK_P` |
| **Social Groups** | `P_SC` (Scheduled Caste), `P_ST` (Scheduled Tribe) |
| **Administrative** | `State`, `District`, `Level`, `TRU` (Total/Rural/Urban) |

### Engineered Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `Sex_Ratio` | (Female / Male) × 1000 | Females per 1000 males |
| `Literacy_Rate` | (Literate / Total) × 100 | Overall literacy percentage |
| `Male_Literacy_Rate` | (Male Literate / Male) × 100 | Male literacy percentage |
| `Female_Literacy_Rate` | (Female Literate / Female) × 100 | Female literacy percentage |
| `Gender_Literacy_Gap` | Male Rate - Female Rate | Literacy disparity |
| `Work_Participation_Rate` | (Workers / Total) × 100 | Workforce participation |
| `Child_Population_Ratio` | (Children 0-6 / Total) × 100 | Young population ratio |
| `SC_Population_Percent` | (SC / Total) × 100 | SC population share |
| `ST_Population_Percent` | (ST / Total) × 100 | ST population share |

---

## 📁 Project Structure

```
EDA-Project/
│
├── 📂 data/
│   └── 2011-IndiaStateDistSbDist-0000.xlsx - Data.csv
│
├── 📂 notebooks/
│   └── 01_India_Census_EDA_Clustering.ipynb    # Main analysis notebook
│
├── 📂 src/
│   ├── data_preprocessing.py      # Data loading & cleaning functions
│   ├── eda_visualization.py       # Visualization functions
│   ├── clustering_model.py        # ML clustering functions
│   └── utils.py                   # Utility functions
│
├── 📂 dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│
├── 📂 outputs/
│   ├── cleaned_dataset.csv        # Processed data
│   ├── cluster_results.csv        # Clustering results
│   └── analysis_summary.txt       # Summary report
│
├── 📂 docs/
│   └── documentation files
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # MIT License
```

---

## ✨ Features & Techniques

### 📈 Exploratory Data Analysis (EDA)

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Univariate Analysis** | Distribution of individual features | Histograms, Box plots, Density plots |
| **Bivariate Analysis** | Relationships between two variables | Scatter plots, Trend lines |
| **Multivariate Analysis** | Multiple variable relationships | Scatter matrix, Parallel coordinates |
| **Correlation Analysis** | Feature correlations | Heatmaps, Correlation coefficients |
| **Outlier Detection** | Identify anomalies | IQR method, Box plots |
| **Descriptive Statistics** | Summary metrics | Mean, Median, Std, Skewness, Kurtosis |

### 🤖 Machine Learning Techniques

| Technique | Purpose | Library |
|-----------|---------|---------|
| **K-Means Clustering** | Group similar states | scikit-learn |
| **Elbow Method** | Find optimal K | Custom implementation |
| **Silhouette Analysis** | Evaluate cluster quality | scikit-learn |
| **PCA** | Dimensionality reduction & visualization | scikit-learn |
| **StandardScaler** | Feature normalization | scikit-learn |

### 📊 Visualization Libraries

| Library | Usage |
|---------|-------|
| **Matplotlib** | Static plots, Histograms, Box plots |
| **Seaborn** | Statistical visualizations, Heatmaps |
| **Plotly** | Interactive charts, Scatter plots, 3D visualizations |
| **Streamlit** | Web dashboard framework |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/india-census-eda.git
cd india-census-eda
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
streamlit>=1.28.0
plotly>=5.0.0
jupyter>=1.0.0
notebook>=6.4.0
joblib>=1.1.0
statsmodels>=0.13.0
```

---

## 💻 Usage

### Option 1: Run Streamlit Dashboard (Recommended)
```bash
cd dashboard
streamlit run app.py
```
Then open your browser at `http://localhost:8501`

### Option 2: Run Jupyter Notebook
```bash
jupyter notebook notebooks/01_India_Census_EDA_Clustering.ipynb
```

### Option 3: Use as Python Module
```python
from src.data_preprocessing import load_dataset, create_features
from src.clustering_model import perform_kmeans_clustering

# Load data
df = load_dataset("data/2011-IndiaStateDistSbDist-0000.xlsx - Data.csv")

# Create features
df_features = create_features(df)

# Run clustering
results = perform_kmeans_clustering(df_features, n_clusters=4)
```

---

## 📱 Dashboard Guide

### Navigation Pages

#### 🏠 Overview
- Dataset summary and key metrics
- Total population, states count, districts count
- Administrative level distribution (pie chart)
- Top 10 states by population (bar chart)

#### 📈 Distribution Analysis
- Select any numeric feature for analysis
- **Histogram** with mean and median markers
- **Box plot** with individual data points
- Complete **descriptive statistics** (mean, median, std, skewness, kurtosis)

#### 🔗 Correlation Analysis
- Multi-select features for correlation matrix
- **Interactive heatmap** with correlation values
- **Scatter plot** with trend line (OLS regression)
- Correlation coefficient display

#### 🗺️ State Comparison
- Compare all states on selected metric
- **Horizontal bar chart** sorted by value
- **Top 5** and **Bottom 5** performers
- Color-coded by metric value

#### 📊 Deep EDA
Four analysis types:
1. **Univariate Analysis**: Grid of distributions for key features with summary statistics
2. **Bivariate Analysis**: Scatter plots with trendlines, Gender literacy comparison bar chart
3. **Multivariate Analysis**: Scatter matrix (pair plot), Parallel coordinates plot
4. **Outlier Detection**: IQR method with box plots and identified outlier states

#### 🤖 ML Clustering
Three analysis modes:

**1. Find Optimal Clusters**
- Elbow method plot (Inertia vs K)
- Silhouette score plot
- Automatic recommendation for optimal K

**2. Run Clustering**
- K-Means clustering with user-selected K
- PCA 2D visualization of clusters
- Cluster distribution pie chart
- States grouped by cluster (expandable)
- Cluster characteristics heatmap
- Download cluster results as CSV

**3. Cluster Insights**
- Box plot comparison across clusters
- Radar chart for cluster profiles (normalized)
- Summary statistics table (mean, std, min, max)

#### 📋 Data Explorer
- View data at different levels (State, District, Raw)
- Download processed data as CSV
- Interactive data tables with sorting

---

## 🔧 Technical Implementation

### Data Pipeline

```
Raw Data → Load → Filter by Level → Clean → Feature Engineering → Analysis/ML
```

### Clustering Pipeline

```
Select Features → Standardize (Z-score) → Find Optimal K → K-Means → PCA → Visualize
```

### Key Algorithms

#### K-Means Clustering
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

#### Optimal K Selection
```python
# Elbow Method - measures within-cluster sum of squares
inertias = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(2, 11)]

# Silhouette Score - measures cluster cohesion and separation
from sklearn.metrics import silhouette_score
scores = [silhouette_score(X, KMeans(n_clusters=k).fit_predict(X)) for k in range(2, 11)]
optimal_k = range(2, 11)[np.argmax(scores)]
```

#### PCA Visualization
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
variance_explained = pca.explained_variance_ratio_.sum()
```

#### Statistical Measures (No SciPy - Using Pandas)
```python
# Skewness - measure of asymmetry
skewness = data.skew()

# Kurtosis - measure of tail heaviness  
kurtosis = data.kurtosis()

# IQR for outlier detection
Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
```

---

## 📊 Results & Insights

### Key Findings

1. **Literacy Patterns**
   - Southern states show higher literacy rates (Kerala: 94%, Tamil Nadu: 80%)
   - Significant gender literacy gap in northern states (Bihar, Rajasthan, UP)
   - Strong positive correlation between female literacy and sex ratio

2. **Demographic Clusters** (Example K=4)
   - **Cluster 0 - High Development**: Kerala, Goa, Mizoram (High literacy, balanced sex ratio)
   - **Cluster 1 - Moderate Development**: Karnataka, Maharashtra, Tamil Nadu
   - **Cluster 2 - Developing**: Bihar, UP, Jharkhand (Lower literacy, high child population)
   - **Cluster 3 - Tribal States**: Northeast states (High ST population, moderate literacy)

3. **Work Participation**
   - Higher in northeastern states and Himachal Pradesh
   - Significant gender gap in workforce participation across all states
   - Negative correlation with child population ratio

4. **Social Indicators**
   - ST population concentrated in central/eastern India and Northeast
   - SC population distributed across northern states (Punjab, UP, Bihar)
   - Both indicators show correlation with literacy patterns

### Cluster Characteristics Summary

| Metric | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|--------|-----------|-----------|-----------|-----------|
| Literacy Rate | High (>80%) | Moderate (70-80%) | Low (<65%) | Moderate (65-75%) |
| Sex Ratio | High (>980) | Moderate (920-960) | Low (<920) | High (>970) |
| Work Rate | Moderate | High | Low | High |
| Child Pop | Low | Moderate | High | Moderate |
| Profile | Developed | Industrial | Developing | Tribal/NE |

---

## 🛠️ Future Enhancements

- [ ] Add predictive models for demographic forecasting
- [ ] Include time-series analysis comparing Census 2001 and 2011
- [ ] Add geographic visualization with choropleth maps
- [ ] Implement DBSCAN and Hierarchical clustering algorithms
- [ ] Add automated PDF report generation
- [ ] Include district-level clustering analysis
- [ ] Add rural vs urban comparison analysis

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Census of India for providing the dataset
- Streamlit team for the amazing dashboard framework
- scikit-learn contributors for ML algorithms

---

<div align="center">
  <p>Made with ❤️ for Data Science</p>
  <p>🇮🇳 <b>India Census 2011 EDA & Clustering Project</b></p>
</div>
