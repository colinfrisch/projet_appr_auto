# Weld Quality Analysis and Prediction
## Machine Learning Project Report

**Project**: Steel Weld Database Analysis  
**Date**: November 6, 2025  
**Objective**: Predict weld quality and identify distinct welding process patterns through regression and clustering

---

## Executive Summary

This project analyzes a comprehensive steel weld database containing 1,652 welding samples with 44 features including chemical composition, welding parameters, and mechanical properties. The main objectives were to:

1. **Predict weld quality** using machine learning regression models
2. **Identify distinct welding patterns** through clustering analysis
3. **Understand key factors** influencing weld quality

**Key Results:**
- Achieved **74.7% R² score** in predicting weld quality using XGBoost
- Identified **6 distinct welding clusters** with unique characteristics
- Discovered that **electrical parameters** (current, voltage, power input) and **impurities** (sulfur, phosphorus) are the most important factors

---

## 1. Data Preprocessing

### 1.1 Dataset Overview

The welddb dataset contains:
- **1,652 welding samples** from various steel welding processes
- **44 features** spanning multiple categories:
  - Chemical composition (12 features): Carbon, Silicon, Manganese, Sulfur, Phosphorus, etc.
  - Welding parameters (6 features): Current, Voltage, Heat Input, Temperature, etc.
  - Mechanical properties (9 features): Tensile Strength, Toughness, Elongation, Hardness, etc.
  - Microstructure properties (5 features): Ferrite phases, Martensite, etc.
  - Process variables (3 categorical): AC/DC, Electrode Polarity, Weld Type

### 1.2 Data Cleaning Strategy

**Challenge**: Significant missing data and ambiguous values (e.g., "<0.002", ">500")

**Solutions implemented:**
1. **Ambiguous value handling**: 
   - Values with "<" → multiplied by 0.5 (half the detection limit)
   - Values with ">" → multiplied by 1.5 (1.5× the upper limit)

2. **Missing data strategy**:
   - Removed columns with >50% missing values (22 columns eliminated)
   - Removed rows with >30% missing values
   - Retained **31 features** for full dataset analysis

3. **Outlier treatment**:
   - Z-score method (threshold = 3)
   - Marked 681 outliers as NaN
   - Applied **KNN imputation** (k=5) to fill missing values

4. **Categorical encoding**:
   - Created **12 dummy variables** for categorical features
   - Avoided multicollinearity by dropping first category

**Result**: Clean dataset with 1,652 complete samples and 43 features (31 numeric + 12 categorical)

### 1.3 Feature Engineering

Nine engineered features were created based on metallurgical principles:

**Impurity Features:**
- `impurities_index` = Sulfur + Phosphorus
- `sulphur_phosphorus_ratio` = Sulfur / Phosphorus

**Electrical Features:**
- `power_input_kw` = (Current × Voltage) / 1000
- `power_efficiency` = Heat Input / (Current × Voltage)
- `current_density_proxy` = Current / Voltage

**Carbon/Manganese Features:**
- `hardenability_index` = Carbon + (Manganese / 6)  *(Carbon Equivalent)*
- `carbon_manganese_ratio` = Carbon / Manganese

**Thermal Features:**
- `cooling_rate_proxy` = Heat Input / (Interpass Temp + 273)
- `thermal_cycle_intensity` = Heat Input / (Interpass Temp + 1)

These features capture complex physical relationships that simple linear models might miss.

### 1.4 Dimensionality Reduction (PCA)

**Principal Component Analysis** was applied to reduce dimensionality while preserving information:

- **11 principal components** selected (optimal based on scree plot)
- **88.19% variance explained** by these 11 components
- Component breakdown:
  - PC1 (27.81%): Dominated by electrical parameters (power, current, voltage)
  - PC2 (12.67%): Chemical composition (manganese, silicon)
  - PC3-PC11 (47.71%): Mixed thermal, chemical, and process parameters

**Key findings from PCA loadings:**
- PC1 strongly correlates with engineered features: `power_input_kw` (0.94), `current_density_proxy` (0.85), `cooling_rate_proxy` (0.88)
- PC2 captures chemical composition: `manganese` (0.74), `hardenability_index` (0.64)

---

## 2. Regression Analysis: Predicting Weld Quality

### 2.1 Target Variable Definition

A composite **quality score** was created from four mechanical properties:

**Formula:**
```
Quality Score = 0.33 × (UTS_normalized) + 0.33 × (Toughness_normalized) + 0.34 × [(Elongation_normalized + Reduction_normalized) / 2]
```

Where:
- **UTS**: Ultimate Tensile Strength (MPa) - measures maximum stress before failure
- **Toughness**: Charpy Impact Toughness (J) - measures shock absorption
- **Elongation**: Elongation (%) - measures ductility/stretchability
- **Reduction**: Reduction of Area (%) - measures ductility/deformation capacity

**Rationale**: This weighted score balances strength (33%), toughness (33%), and ductility (34%), representing overall weld quality.

### 2.2 Dataset Reduction

**Challenge**: Many samples lacked sufficient target values for quality score calculation.

**Solution**: 
- Retained only samples with **≥2 of 4 target values**
- Applied KNN imputation to complete missing target values
- **Final regression dataset**: 720 samples (43.6% of original data)

Quality score statistics:
- Mean: 0.585
- Median: 0.631  
- Std Dev: 0.098

### 2.3 Model Comparison

Three models were evaluated using 5-fold cross-validation:

| Model | Cross-Val R² | Test R² | Notes |
|-------|-------------|---------|-------|
| Ridge Regression | 0.544 ± 0.055 | - | Baseline linear model |
| Random Forest | 0.646 ± 0.043 | 0.703 | Good performance, robust |
| Gradient Boosting | 0.653 ± 0.060 | 0.727 | Best traditional ensemble |
| **XGBoost** | **-** | **0.747** | **Best overall performance** |

### 2.4 Hyperparameter Optimization

**XGBoost - Best Configuration:**
```python
{
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.7
}
```

**Performance Metrics (Test Set):**
- R² Score: 0.747
- Mean Squared Error: 0.00290
- Mean Absolute Error: 0.0357

This means the model explains **74.7% of variance** in weld quality, which is excellent for this complex domain.

### 2.5 Feature Importance for Regression

Top predictive features (from Gradient Boosting analysis):

1. **Principal Components** (PC1-PC11): Capture complex interactions
2. **Categorical Variables**: Weld type significantly impacts quality
3. **Original Features** (when analyzed):
   - Electrical parameters dominate
   - Chemical impurities (S, P) strongly influence quality
   - Thermal cycle parameters moderately important

---

## 3. Clustering Analysis: Identifying Welding Patterns

### 3.1 Clustering Approach

**Data**: PCA-transformed features (11 components) from full dataset (1,652 samples)

**Methods Evaluated:**

| Method | N Clusters | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|--------|-----------|-----------------|----------------|-------------------|
| **K-Means** | **6** | **0.206** | **1.550** | **353.7** |
| Hierarchical | 6 | 0.184 | 1.625 | 334.9 |
| Spectral | 6 | 0.236 | 1.515 | 171.9 |
| Mini-Batch K-Means | 6 | 0.188 | 1.657 | 341.5 |
| DBSCAN | 5 | 0.443* | 0.579* | 228.6* |

*DBSCAN excluded 42% of data as noise, limiting practical utility.

**Selection Criteria**: Multi-metric composite score (40% Silhouette + 30% Davies-Bouldin + 30% Calinski-Harabasz)

**Winner**: **K-Means with K=6** (Composite Score: 0.769)

### 3.2 Optimal Number of Clusters

K=6 was selected based on:
- **Elbow method**: Inertia curve flattens after K=6
- **Silhouette analysis**: K=6 shows good balance (0.206)
- **Physical interpretation**: Six clusters represent distinct welding regimes

### 3.3 Cluster Characteristics

#### **Cluster 0** (649 samples, 39.3%) - *"Standard High-Manganese Welds"*
- **Largest cluster**, representing typical welding practices
- **High**: Manganese content (1.49% vs 1.20% overall)
- **Low**: Current (171A vs 262A), heat input, power
- **Quality**: Average toughness (85.4J), low test temperature (-47.8°C)
- **Interpretation**: Low-current, manganese-rich steels for low-temperature applications

#### **Cluster 1** (168 samples, 10.2%) - *"High-Efficiency, High-Impurity Welds"*
- **High**: Power efficiency, phosphorus (0.015% vs 0.010%), thermal cycle intensity
- **Moderate**: Current (228A), power (5.8kW)
- **Quality**: Lower toughness (72.3J), moderate test temperature (-31.6°C)
- **Interpretation**: Energy-efficient processes, but impurities reduce toughness

#### **Cluster 2** (146 samples, 8.8%) - *"Vanadium-Rich, Heat-Treated Welds"*
- **High**: Vanadium (0.063% vs 0.011%), PWHT temperature (465°C), nitrogen
- **Low**: Current density
- **Quality**: **Lowest toughness** (52.2J), warm test temperature (-1.4°C)
- **Interpretation**: High-strength alloy steels with post-weld heat treatment, sacrificing toughness for strength

#### **Cluster 3** (343 samples, 20.8%) - *"Low-Manganese, Low-Hardenability Welds"*
- **Low**: Manganese (0.78% vs 1.20%), hardenability index
- **Similar to Cluster 0** in electrical parameters
- **Quality**: Good toughness (85.3J), low test temperature (-45.9°C)
- **Interpretation**: Soft steels with lower alloy content, good low-temperature performance

#### **Cluster 4** (101 samples, 6.1%) - *"High Carbon-Ratio, Premium Welds"*
- **Smallest cluster** with unique properties
- **High**: Carbon/Manganese ratio, current density, PWHT temperature
- **Quality**: **Highest toughness** (110.6J), warmest test temperature (-14.2°C)
- **Interpretation**: Carefully balanced composition with optimized heat treatment → superior quality

#### **Cluster 5** (245 samples, 14.8%) - *"High-Power, High-Current Welds"*
- **Highest**: Power input (13.5kW vs 7.1kW), current (424A vs 262A), current density
- **Moderate**: All other parameters
- **Quality**: Good toughness (89.6J), moderate test temperature (-29.0°C)
- **Interpretation**: High-energy welding processes (e.g., submerged arc welding)

### 3.4 Key Discriminating Features (ANOVA F-test)

Features that **best distinguish clusters** (ranked by F-statistic):

1. **Current** (F=6817) - Most important differentiator
2. **Power input** (F=5398)
3. **Current density** (F=2833)
4. **Heat input** (F=1579)
5. **Cooling rate** (F=1078)
6. **Voltage** (F=807)
7. **Vanadium content** (F=527)
8. **Manganese content** (F=499)
9. **Impurities index** (F=412)
10. **PWHT temperature** (F=285)

**Insight**: Electrical parameters dominate cluster separation, followed by chemical composition and thermal treatment.

### 3.5 Cluster Quality Analysis

**Within-cluster homogeneity** (lower std dev = more homogeneous):

- **Most homogeneous**: Cluster 0 (0.70) and Cluster 3 (0.97)
- **Least homogeneous**: Cluster 1 (7.28) and Cluster 5 (7.00)

**Most similar clusters**: Cluster 0 ↔ Cluster 3 (distance: 2.90)  
**Most different clusters**: Cluster 2 ↔ Cluster 5 (distance: 7.87)

---

## 4. Key Findings and Insights

### 4.1 Critical Success Factors for Weld Quality

1. **Electrical Parameters are Paramount**
   - Current, voltage, and power input are the strongest predictors and cluster differentiators
   - Optimal balance varies by material and application

2. **Impurities Significantly Reduce Quality**
   - Sulfur and phosphorus (impurities_index) strongly correlate with reduced toughness
   - Cluster 1 (high impurities) shows 15% lower toughness than average

3. **Engineered Features Add Value**
   - Features like `power_efficiency`, `cooling_rate_proxy`, and `hardenability_index` improve both prediction and interpretation
   - Validate the importance of domain knowledge in feature engineering

4. **Trade-offs Exist**
   - Cluster 2 (high vanadium, heat-treated) sacrifices toughness for strength
   - High current/power doesn't always guarantee best quality (Cluster 5)

### 4.2 Practical Recommendations

**For High-Quality Welds** (inspired by Cluster 4):
- Optimize carbon/manganese ratio
- Apply appropriate post-weld heat treatment
- Control current density carefully
- Target test for specific temperature ranges

**For Low-Temperature Applications** (Clusters 0, 3):
- Prioritize manganese content
- Use lower current settings
- Minimize impurities (S, P)
- Expect good toughness at -45°C or lower

**For High-Strength Applications** (Cluster 2):
- Consider vanadium additions
- Implement post-weld heat treatment (PWHT)
- Accept trade-off in impact toughness
- Suitable for ambient temperature service

---

## 5. Methodology Strengths and Limitations

### 5.1 Strengths

✓ **Comprehensive preprocessing**: Handled ambiguous values, missing data, and outliers systematically  
✓ **Domain-informed feature engineering**: Created physically meaningful features  
✓ **Multiple model comparison**: Evaluated various algorithms before selecting best performer  
✓ **Thorough clustering analysis**: Compared 5 different clustering methods with multiple metrics  
✓ **Statistical rigor**: Used ANOVA, t-tests, and multiple validation techniques  

### 5.2 Limitations

✗ **Reduced dataset for regression**: Only 720/1,652 samples (43.6%) had sufficient target data  
✗ **Imputation uncertainty**: KNN imputation may introduce bias, especially for outliers  
✗ **PCA interpretability**: While effective, principal components are less intuitive than original features  
✗ **Moderate silhouette scores**: Cluster overlap suggests some ambiguity in welding regime boundaries  
✗ **Limited microstructure data**: Many microstructure features removed due to missing values  

---

## 6. Conclusions

This comprehensive analysis of 1,652 steel welds demonstrates the power of machine learning in understanding complex manufacturing processes:

1. **Predictive Success**: XGBoost achieved 74.7% R² in predicting weld quality, enabling:
   - Quality prediction before physical testing
   - Parameter optimization for target quality
   - Cost reduction through fewer failed welds

2. **Pattern Discovery**: K-Means clustering revealed 6 distinct welding regimes:
   - Each cluster represents a coherent strategy (e.g., high-power vs. low-current)
   - Clear trade-offs between strength, toughness, and ductility
   - Practical guidance for process selection

3. **Critical Factors Identified**:
   - **Electrical parameters** (current, voltage, power) dominate both prediction and clustering
   - **Chemical impurities** (S, P) critically impact quality
   - **Heat treatment** (PWHT) enables high-strength applications
   - **Engineered features** capture complex physical relationships

4. **Actionable Insights**:
   - Cluster 4 characteristics (high C/Mn ratio, optimized heat treatment) produce best quality
   - Managing impurities is essential for high-toughness applications
   - Different applications require different welding strategies (clusters)

### Future Work

- **Incorporate microstructure data**: Impute or collect complete ferrite/martensite measurements
- **Time-series analysis**: If available, analyze temporal patterns in welding processes
- **Cost optimization**: Build cost models to balance quality vs. expenses
- **Transfer learning**: Apply models to new steel grades or welding methods
- **Explainable AI**: Use SHAP or LIME for instance-level explanations
- **Real-time monitoring**: Deploy models for in-process quality prediction

---

## 7. Technical Appendix

### 7.1 Software and Libraries

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Statistical Analysis**: scipy

### 7.2 Data Files Generated

1. `processed_welddb.csv` - Cleaned data with engineered features and PCA components
2. `clustered_welddb.csv` - Full dataset with cluster labels
3. `cluster_summary_statistics.csv` - Statistical summaries by cluster
4. `cluster_feature_importance.csv` - ANOVA F-statistics for feature importance
5. `clustering_comparison.csv` - Performance metrics for all clustering methods
6. `pca_model.pkl`, `scaler_model.pkl` - Saved preprocessing models

### 7.3 Reproducibility

All analysis is fully reproducible through the provided Jupyter notebooks:
1. `preprocessing.ipynb` - Data cleaning, feature engineering, PCA
2. `regression.ipynb` - Quality score definition, model training, evaluation
3. `clustering.ipynb` - Clustering algorithm comparison and selection
4. `clusters_analysis.ipynb` - Detailed cluster characterization

Random seeds were set for all stochastic processes (random_state=42).

---

**Report prepared by**: Machine Learning Analysis Team  
**Project Code**: Available in project repository  
**Contact**: For questions or collaborations regarding this analysis

