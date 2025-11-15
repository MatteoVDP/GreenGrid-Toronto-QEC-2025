"""
Data Preprocessing Pipeline for Toronto Tree Planting Recommendation System
Integrates: Heat maps, Tree coverage, Vulnerability, Traffic data
Output: ML-ready datasets for regression model
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TORONTO TREE PLANTING - DATA PREPROCESSING PIPELINE")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_PATH = r"c:\Users\jcube\OneDrive\Desktop\Jacob\School\Queens\Year 5\Extra Curricular\QEC\Toronto_Heat_Vulnerability"
TRAFFIC_PATH = r"c:\Users\jcube\OneDrive\Desktop\Jacob\School\Queens\Year 5\Extra Curricular\QEC\traffic_count_by_intersection_all_modes.csv"
OUTPUT_DIR = "data/processed"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: LOAD ALL DATASETS
# =============================================================================

print("\n[1/10] Loading datasets...")

# Load Toronto DA boundaries
boundaries = gpd.read_file(f"{BASE_PATH}/input_data/Toronto_DA_boundary.zip")
print(f"  ✓ Loaded {len(boundaries)} Dissemination Areas")

# Load heat/temperature data
heat = gpd.read_file(f"{BASE_PATH}/input_data/exposure(degree_day_20).zip")
print(f"  ✓ Loaded heat exposure data")

# Load tree coverage data
trees = gpd.read_file(f"{BASE_PATH}/input_data/canopy_cover.zip")
print(f"  ✓ Loaded tree canopy data")

# Load vulnerability index
vulnerability = gpd.read_file(f"{BASE_PATH}/Results/pca_vuln_index.zip")
print(f"  ✓ Loaded vulnerability index")

# Load traffic data
traffic = pd.read_csv(TRAFFIC_PATH)
print(f"  ✓ Loaded {len(traffic)} traffic intersection points")

# =============================================================================
# STEP 2: PROCESS TRAFFIC DATA (POINTS → POLYGONS)
# =============================================================================

print("\n[2/10] Processing traffic data...")

# Convert traffic CSV to GeoDataFrame
traffic_gdf = gpd.GeoDataFrame(
    traffic,
    geometry=gpd.points_from_xy(traffic['longitude'], traffic['latitude']),
    crs='EPSG:4326'  # Lat/lon coordinate system
)

# Convert to same CRS as DAs (EPSG:32617 - UTM Zone 17N)
traffic_gdf = traffic_gdf.to_crs('EPSG:32617')
print(f"  ✓ Converted traffic data to UTM projection")

# Spatial join: assign each intersection to its DA
traffic_by_da = gpd.sjoin(
    traffic_gdf, 
    boundaries[['DAUID', 'geometry']], 
    how='left', 
    predicate='within'
)
print(f"  ✓ Assigned intersections to DAs")

# Aggregate traffic per DA (sum all intersection counts)
traffic_summary = traffic_by_da.groupby('DAUID').agg({
    'total_vehicle': 'sum',
    'total_bike': 'sum',
    'total_pedestrian': 'sum'
}).reset_index()

# Create total traffic metric (weighted: pedestrians matter more for tree benefit)
traffic_summary['total_traffic'] = (
    traffic_summary['total_vehicle'] * 0.3 +      # Cars benefit less from trees
    traffic_summary['total_bike'] * 0.3 +         # Bikes benefit moderately
    traffic_summary['total_pedestrian'] * 0.4     # Pedestrians benefit most
)

print(f"  ✓ Aggregated traffic to {len(traffic_summary)} DAs")

# =============================================================================
# STEP 3: MERGE ALL DATASETS
# =============================================================================

print("\n[3/10] Merging datasets...")

# Start with boundaries (geometry)
combined = boundaries[['DAUID', 'geometry']].copy()

# Extract needed columns and rename
heat_data = heat[['DAUID', 'SUM_temper']].rename(columns={'SUM_temper': 'heat'})
tree_data = trees[['DAUID', 'canopy_per']].rename(columns={'canopy_per': 'tree_coverage'})
vuln_data = vulnerability[['DAUID', 'std_pc2']].rename(columns={'std_pc2': 'vulnerability'})
traffic_data = traffic_summary[['DAUID', 'total_traffic']].rename(columns={'total_traffic': 'footfall'})

# Convert DAUID to string for consistent merging
combined['DAUID'] = combined['DAUID'].astype(str)
heat_data['DAUID'] = heat_data['DAUID'].astype(str)
tree_data['DAUID'] = tree_data['DAUID'].astype(str)
vuln_data['DAUID'] = vuln_data['DAUID'].astype(str)
traffic_data['DAUID'] = traffic_data['DAUID'].astype(str)

# Merge all datasets on DAUID
combined = combined.merge(heat_data, on='DAUID', how='left')
combined = combined.merge(tree_data, on='DAUID', how='left')
combined = combined.merge(vuln_data, on='DAUID', how='left')
combined = combined.merge(traffic_data, on='DAUID', how='left')

print(f"  ✓ Merged all datasets: {len(combined)} DAs")
print(f"  Columns: {combined.columns.tolist()}")

# =============================================================================
# STEP 4: HANDLE MISSING VALUES
# =============================================================================

print("\n[4/10] Handling missing values...")

# Check missing values
missing_before = combined[['heat', 'tree_coverage', 'vulnerability', 'footfall']].isnull().sum()
print(f"\nMissing values before processing:")
print(missing_before)

# Fill heat, trees, vulnerability with 0 or median
combined['heat'] = combined['heat'].fillna(combined['heat'].median())
combined['tree_coverage'] = combined['tree_coverage'].fillna(0)
combined['vulnerability'] = combined['vulnerability'].fillna(combined['vulnerability'].median())

print(f"  ✓ Filled heat, trees, vulnerability")

# For footfall: use KNN imputation based on spatial proximity
print(f"\n  Processing footfall with KNN imputation...")
missing_footfall = combined['footfall'].isna().sum()
print(f"  DAs with missing footfall: {missing_footfall}")

if missing_footfall > 0:
    # Get centroids for spatial distance
    combined['centroid_x'] = combined.geometry.centroid.x
    combined['centroid_y'] = combined.geometry.centroid.y
    
    # Prepare data for KNN (coordinates + footfall)
    coords_footfall = combined[['centroid_x', 'centroid_y', 'footfall']].values
    
    # KNN imputation (k=5 nearest neighbors)
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(coords_footfall)
    
    # Update footfall with imputed values
    combined['footfall'] = imputed_data[:, 2]
    
    print(f"  ✓ Imputed footfall using 5 nearest neighbors")
else:
    print(f"  ✓ No missing footfall values")

# Verify no missing values remain
missing_after = combined[['heat', 'tree_coverage', 'vulnerability', 'footfall']].isnull().sum()
print(f"\nMissing values after processing:")
print(missing_after)

# =============================================================================
# STEP 5: NORMALIZE FEATURES (0-1 for visualization)
# =============================================================================

print("\n[5/10] Normalizing features (0-1)...")

def normalize(series):
    """Min-max normalization to 0-1 range"""
    return (series - series.min()) / (series.max() - series.min())

# Normalize all features
combined['heat_norm'] = normalize(combined['heat'])
combined['tree_norm'] = normalize(combined['tree_coverage'])
combined['vuln_norm'] = normalize(combined['vulnerability'])

# Log transform footfall before normalizing (handles skewed traffic data)
combined['footfall_log'] = np.log1p(combined['footfall'])  # log(1+x) to handle zeros
combined['footfall_norm'] = normalize(combined['footfall_log'])

print(f"  ✓ Normalized: heat, tree_coverage, vulnerability")
print(f"  ✓ Log-transformed and normalized: footfall (reduces skewness)")

# =============================================================================
# STEP 6: CREATE SYNTHETIC TARGET VARIABLE
# =============================================================================

print("\n[6/10] Creating synthetic target variable...")

def create_target(row, w_heat=0.3, w_footfall=0.3, w_tree=0.2, w_vuln=0.2):
    """
    Create synthetic target for supervised learning
    Based on expert weights and domain knowledge
    """
    # Invert tree coverage (less trees = higher priority for planting)
    tree_inverted = 1 - row['tree_norm']
    
    score = (
        w_heat * row['heat_norm'] +           # High heat = high priority
        w_footfall * row['footfall_norm'] +   # High traffic = high priority
        w_tree * tree_inverted +              # Low trees = high priority
        w_vuln * row['vuln_norm']             # High vulnerability = high priority
    )
    return score

# Create target
combined['target_priority'] = combined.apply(create_target, axis=1)

# Normalize target to 0-1
combined['target_priority'] = normalize(combined['target_priority'])

print(f"  ✓ Created target variable (priority score)")
print(f"  Target range: {combined['target_priority'].min():.3f} to {combined['target_priority'].max():.3f}")

# =============================================================================
# STEP 7: STANDARDIZE FEATURES FOR ML (mean=0, std=1)
# =============================================================================

print("\n[7/10] Standardizing features for ML...")

# Select feature columns
feature_cols = ['heat', 'tree_coverage', 'vulnerability', 'footfall']
X = combined[feature_cols].copy()

# Standardize (mean=0, std=1) for ML models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add scaled features to dataframe
for i, col in enumerate(feature_cols):
    combined[f'{col}_scaled'] = X_scaled[:, i]

# Save scaler for production use
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')
print(f"  ✓ Saved scaler to {OUTPUT_DIR}/scaler.pkl")

# =============================================================================
# STEP 8: FEATURE ENGINEERING (Optional enhancements)
# =============================================================================

print("\n[8/10] Engineering additional features...")

# Interaction features
combined['heat_x_vuln'] = combined['heat_norm'] * combined['vuln_norm']
combined['low_trees_high_heat'] = (1 - combined['tree_norm']) * combined['heat_norm']
combined['high_traffic_low_trees'] = combined['footfall_norm'] * (1 - combined['tree_norm'])

# Spatial features
combined['centroid_lat'] = combined.geometry.centroid.y
combined['centroid_lon'] = combined.geometry.centroid.x

print(f"  ✓ Created interaction and spatial features")

# =============================================================================
# STEP 9: SAVE PROCESSED DATA
# =============================================================================

print("\n[9/10] Saving processed datasets...")

# 1. Full dataset in original CRS (EPSG:32617) for backend
combined.to_file(f'{OUTPUT_DIR}/toronto_data_utm.gpkg', driver='GPKG')
print(f"  ✓ Saved: {OUTPUT_DIR}/toronto_data_utm.gpkg")

# 2. Convert to WGS84 (EPSG:4326) for web mapping
combined_web = combined.to_crs(epsg=4326)
combined_web.to_file(f'{OUTPUT_DIR}/toronto_data_web.geojson', driver='GeoJSON')
print(f"  ✓ Saved: {OUTPUT_DIR}/toronto_data_web.geojson")

# 3. ML features and target (CSV for easy loading)
ml_features = combined[feature_cols + ['heat_norm', 'tree_norm', 'vuln_norm', 'footfall_norm']]
ml_features.to_csv(f'{OUTPUT_DIR}/features.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR}/features.csv")

combined[['DAUID', 'target_priority']].to_csv(f'{OUTPUT_DIR}/target.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR}/target.csv")

# 4. Lightweight version for frontend (only essential columns)
columns_for_web = [
    'DAUID', 'heat_norm', 'tree_norm', 'vuln_norm', 'footfall_norm',
    'target_priority', 'geometry'
]
combined_simple = combined_web[columns_for_web].copy()
combined_simple.to_file(f'{OUTPUT_DIR}/toronto_simple.geojson', driver='GeoJSON')
print(f"  ✓ Saved: {OUTPUT_DIR}/toronto_simple.geojson")

# 5. Summary statistics
stats = combined[['heat', 'tree_coverage', 'vulnerability', 'footfall', 'target_priority']].describe()
stats.to_csv(f'{OUTPUT_DIR}/summary_statistics.csv')
print(f"  ✓ Saved: {OUTPUT_DIR}/summary_statistics.csv")

# =============================================================================
# STEP 10: GENERATE SUMMARY REPORT
# =============================================================================

print("\n" + "="*70)
print("PREPROCESSING SUMMARY")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"  Total Dissemination Areas: {len(combined)}")
print(f"  CRS: {boundaries.crs} → EPSG:4326 (web)")
print(f"  Bounding Box: {combined.total_bounds}")

print(f"\n📈 Feature Statistics:")
print(combined[['heat', 'tree_coverage', 'vulnerability', 'footfall']].describe().round(2))

print(f"\n🎯 Target Variable (Priority Score):")
print(f"  Min:    {combined['target_priority'].min():.3f}")
print(f"  Max:    {combined['target_priority'].max():.3f}")
print(f"  Mean:   {combined['target_priority'].mean():.3f}")
print(f"  Median: {combined['target_priority'].median():.3f}")
print(f"  Std:    {combined['target_priority'].std():.3f}")

print(f"\n📁 Output Files Created:")
print(f"  1. toronto_data_utm.gpkg      - Full dataset (original CRS)")
print(f"  2. toronto_data_web.geojson   - Full dataset (web-friendly)")
print(f"  3. toronto_simple.geojson     - Lightweight (for frontend)")
print(f"  4. features.csv               - ML training features")
print(f"  5. target.csv                 - ML training target")
print(f"  6. scaler.pkl                 - Feature scaler (for production)")
print(f"  7. summary_statistics.csv     - Data summary")

print("\n✅ PREPROCESSING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Train ML model using features.csv and target.csv")
print("  2. Use toronto_simple.geojson for frontend visualization")
print("  3. Load scaler.pkl for production predictions")
print("="*70)
