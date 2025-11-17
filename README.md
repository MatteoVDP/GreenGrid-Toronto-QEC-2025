<div align="left">
  <img src="frontend/Greengrid_Toronto_Logo.png" alt="GreenGrid Toronto Logo" width="300">
  
  **AI-Powered Urban Forestry Planning System**
</div>

An intelligent tree planting prioritization system that uses machine learning to identify optimal locations for urban tree planting across Toronto's 3,702 neighborhoods, balancing heat mitigation, environmental justice, and community access.

View the live demo below!

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://jcub05.github.io/GreenGrid-Toronto-QEC-2025/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Data Sources](#data-sources)
- [Model Performance](#model-performance)
- [Usage Guide](#usage-guide)
- [Technologies](#technologies)
- [Team](#team)

---

## 🌍 Overview

Toronto faces increasing urban heat island effects and unequal access to green space. GreenGrid Toronto solves this by analyzing multiple environmental and social factors to recommend where tree planting would have the greatest impact.

**The Problem:**
- Urban heat disproportionately affects vulnerable populations
- Limited resources require strategic tree planting decisions
- Complex interplay of heat, tree coverage, demographics, and foot traffic

**Our Solution:**
- AI model trained on 4 diverse datasets (heat maps, tree coverage, vulnerability indices, traffic patterns)
- 98.3% prediction accuracy using Random Forest regression
- Interactive web map for easy visualization and decision-making

---

## ✨ Features

- 🎯 **Smart Prioritization**: ML model identifies 1,613 high-priority areas (43.6% of Toronto)
- 🗺️ **Interactive Map**: Click any neighborhood to see detailed priority scores and contributing factors
- 📊 **Multi-Factor Analysis**: Considers heat exposure, tree coverage, population vulnerability, and foot traffic
- 🔄 **Dynamic Visualization**: Switch between different scoring metrics (priority, heat, trees, vulnerability, footfall)
- 📈 **High Accuracy**: 98.3% R² score with robust cross-validation
- 🌐 **Web-Based**: No installation needed - accessible from any browser

---

## 🚀 Quick Start

### View the Live Demo

Visit the interactive map: [https://jcub05.github.io/GreenGrid-Toronto-QEC-2025/](https://jcub05.github.io/GreenGrid-Toronto-QEC-2025/)

### Run Locally

**Prerequisites:**
- Python 3.13+
- Git

**Installation:**

```bash
# Clone the repository
git clone https://github.com/Jcub05/QEC-2025-Team2.git
cd QEC-2025-Team2

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing (optional - pre-processed data included)
python data_preprocessing/preprocess_data.py

# Train model (optional - trained model included)
python ml_model/train_model.py

# View results
python visualization/visualize_map.py
```

**Open the Frontend:**
```bash
# Navigate to frontend folder
cd frontend

# Open index.html in your browser
# Or use a local server:
python -m http.server 8000
# Then visit: http://localhost:8000
```

---

## 🧠 How It Works

### 1. Data Collection & Integration

We integrate four diverse datasets covering all 3,702 Toronto neighborhoods:

- **Heat Vulnerability Index** (University of Toronto)
  - Heat exposure measurements
  - Tree canopy coverage percentages
  - Population vulnerability scores

- **Traffic Data** (City of Toronto Open Data)
  - 6,318 intersection measurement points
  - Daily pedestrian and vehicle counts
  - Spatially aggregated to neighborhood level

- **Census Boundaries** (Statistics Canada)
  - Geographic framework for analysis
  - Dissemination Area (DA) polygons

### 2. Data Preprocessing

**Key Steps:**
- **Spatial Alignment**: Convert all datasets to consistent coordinate system (EPSG:32617)
- **Missing Value Handling**: Use K-Nearest Neighbors (k=5) to impute missing traffic data for 1,248 neighborhoods
- **Skewness Correction**: Apply log transformation to traffic data (increased model importance from 1% → 17.6%)
- **Normalization**: Scale all features to 0-1 range for fair comparison

### 3. Ground Truth Generation

Since no historical tree planting data exists, we create synthetic labels using domain expert weights:

```
Priority Score = 0.3×heat + 0.3×footfall + 0.2×(1-trees) + 0.2×vulnerability
```

### 4. Machine Learning Model

**Algorithm**: Random Forest Regression
- **Architecture**: 200 decision trees, max depth 15
- **Training**: 80/20 train-test split (2,961 train, 741 test)
- **Validation**: 5-fold cross-validation

**Model Discovery:**
The model learned that vulnerability is actually the strongest predictor (44%), adjusting initial expert weights:

| Factor | Initial Weight | Learned Importance | Change |
|--------|---------------|-------------------|--------|
| Vulnerability | 20% | **44.0%** | +24% ↑ |
| Tree Coverage | 20% | 19.9% | ±0% |
| Heat | 30% | 18.5% | -11.5% ↓ |
| Footfall | 30% | 17.6% | -12.4% ↓ |

### 5. Web Visualization

Interactive map built with Leaflet.js:
- Color-coded neighborhoods (yellow = low priority, orange/brown = high priority)
- Click any area for detailed scores
- Switch between different metrics
- Responsive design for mobile/desktop

---

## 📊 Data Sources

### Primary Datasets

1. **Toronto Heat Vulnerability Study**
   - Source: University of Toronto Actuarial Science
   - Repository: [GitHub](https://github.com/UofTActuarial/toronto-heat-vulnerability)
   - Coverage: 3,702 Dissemination Areas
   - Variables: Heat exposure index, tree canopy %, vulnerability score

2. **Traffic Volumes at Intersections**
   - Source: City of Toronto Open Data
   - Link: [Open Data Portal](https://open.toronto.ca/dataset/traffic-volumes-at-intersections/)
   - Records: 6,318 intersection points
   - Variables: Daily vehicle/pedestrian counts

3. **Census Geographic Boundaries**
   - Source: Statistics Canada (2021 Census)
   - Format: Shapefile/GeoPackage
   - Purpose: Spatial framework for aggregation

### Data Characteristics

- **Spatial Resolution**: Dissemination Area (DA) level (~500 people each)
- **Coverage**: Complete City of Toronto
- **Coordinate System**: EPSG:32617 (UTM Zone 17N) → EPSG:4326 (WGS84) for web

---

## 🎯 Model Performance

### Accuracy Metrics

| Metric | Training Set | Test Set | Interpretation |
|--------|-------------|----------|----------------|
| **R² Score** | 99.57% | **98.27%** | Model explains 98.3% of variance |
| **MAE** | 0.0057 | **0.0127** | Average error of 1.27% |
| **RMSE** | 0.0100 | **0.0210** | Typical error of 2.1% |
| **CV Score (5-fold)** | — | **98.25% ± 0.5%** | Highly consistent across data splits |

### Feature Importance

The model discovered these factors drive tree planting priority:

1. **Vulnerability** (44.0%) - Most important predictor
   - Captures compound effects of heat + socioeconomic factors
   - Prioritizes environmental justice

2. **Tree Coverage** (19.9%) - Inverted (low trees = higher priority)
   - Identifies gaps in existing canopy

3. **Heat Exposure** (18.5%) - Climate adaptation factor
   - Urban heat island mitigation

4. **Foot Traffic** (17.6%) - Public benefit factor
   - Maximizes community access to shade

### Results Summary

- **Total Areas Analyzed**: 3,702 neighborhoods
- **High Priority (0.67-1.0)**: 1,613 DAs (43.6%)
- **Medium Priority (0.33-0.67)**: 1,927 DAs (52.1%)
- **Low Priority (0.0-0.33)**: 162 DAs (4.4%)

**Key Finding**: Nearly half of Toronto requires high-priority tree planting intervention based on combined environmental and social factors.

---

## 📖 Usage Guide

### For City Planners

1. **Explore the Map**: Visit the live demo and zoom to your area of interest
2. **Identify Priorities**: Dark orange/brown areas need trees most urgently
3. **View Details**: Click any neighborhood to see:
   - Overall priority score (0-1 scale)
   - Heat exposure level
   - Current tree coverage
   - Population vulnerability
   - Pedestrian traffic volume
4. **Switch Metrics**: Use legend controls to view individual factors
5. **Export Data**: Download `all_predictions.csv` for bulk analysis

### For Researchers

**Reproduce Results:**
```bash
# 1. Preprocess data
python data_preprocessing/preprocess_data.py
# Output: features.csv, target.csv, toronto_simple.geojson

# 2. Train model
python ml_model/train_model.py
# Output: tree_priority_model.pkl, feature_importance.csv, metrics

# 3. Generate visualizations
python visualization/visualize_map.py
# Output: toronto_priority_map.png (high-res static map)
```

**Modify the Model:**
Edit `ml_model/train_model.py` to adjust:
- Number of trees: `n_estimators=200`
- Tree depth: `max_depth=15`
- Feature weights: Modify synthetic target formula

**Add New Data:**
Extend `data_preprocessing/preprocess_data.py` to integrate additional datasets (e.g., soil quality, air pollution)

---

## 🛠️ Technologies

### Backend & Data Processing
- **Python 3.13** - Core programming language
- **geopandas 0.14.1** - Spatial data manipulation
- **scikit-learn 1.7.2** - Machine learning (Random Forest, KNN imputer)
- **pandas 2.1.3** - Tabular data processing
- **numpy 1.26.2** - Numerical computing

### Machine Learning
- **Random Forest Regressor** - Ensemble learning algorithm
- **K-Nearest Neighbors Imputation** - Missing value handling
- **5-Fold Cross-Validation** - Model robustness testing

### Visualization & Frontend
- **Leaflet.js** - Interactive web mapping
- **OpenStreetMap** - Base map tiles
- **matplotlib 3.8.2** - Static map generation
- **HTML/CSS/JavaScript** - Web interface

### Geospatial
- **EPSG:32617** - UTM Zone 17N for processing
- **EPSG:4326** - WGS84 for web delivery
- **GeoJSON** - Web-friendly spatial data format

---

## 👥 Team

**QEC Team 2 - Queens Engineering Competition 2025**

This project was developed for the Queens Engineering Competition, demonstrating the intersection of data science, environmental planning, and social impact.

---

## 🙏 Acknowledgments

- **University of Toronto** - Toronto Heat Vulnerability Study dataset
- **City of Toronto** - Open data portal and traffic datasets
- **Statistics Canada** - Census geographic boundaries
- **OpenStreetMap contributors** - Base map tiles
- **Queens Engineering Competition** - Platform and motivation
