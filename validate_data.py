"""
Validation script to verify data preprocessing results
Checks that all 4 features were properly integrated
"""

import pandas as pd
import geopandas as gpd
import numpy as np

print("="*70)
print("DATA PREPROCESSING VALIDATION")
print("="*70)

# =============================================================================
# 1. CHECK CSV FILES
# =============================================================================

print("\n[1] Checking CSV files...")

# Load features and target
features = pd.read_csv('data/processed/features.csv')
target = pd.read_csv('data/processed/target.csv')

print(f"\n✓ Features shape: {features.shape}")
print(f"  Expected: (3702, 8) - 4 raw + 4 normalized")
print(f"  Columns: {list(features.columns)}")

print(f"\n✓ Target shape: {target.shape}")
print(f"  Expected: (3702, 2) - DAUID + target_priority")
print(f"  Columns: {list(target.columns)}")

# Check for missing values
print(f"\n✓ Missing values in features:")
print(features.isnull().sum())

print(f"\n✓ Missing values in target:")
print(target.isnull().sum())

# =============================================================================
# 2. CHECK GEOJSON FILES
# =============================================================================

print("\n[2] Checking GeoJSON files...")

# Load simple geojson
simple = gpd.read_file('data/processed/toronto_simple.geojson')

print(f"\n✓ GeoJSON shape: {simple.shape}")
print(f"  Expected: (3702, 7) - DAUID + 4 normalized + target + geometry")
print(f"  Columns: {list(simple.columns)}")

# Check CRS
print(f"\n✓ CRS: {simple.crs}")
print(f"  Expected: EPSG:4326 (WGS84 for web)")

# =============================================================================
# 3. VALIDATE FEATURE RANGES
# =============================================================================

print("\n[3] Validating feature ranges...")

# Check normalized features are in 0-1 range
norm_cols = ['heat_norm', 'tree_norm', 'vuln_norm', 'footfall_norm']

print(f"\nNormalized features (should be 0-1):")
for col in norm_cols:
    min_val = simple[col].min()
    max_val = simple[col].max()
    mean_val = simple[col].mean()
    print(f"  {col:15s}: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")
    
    if min_val < 0 or max_val > 1:
        print(f"    ⚠️  WARNING: {col} is not in [0,1] range!")

# Check target
print(f"\nTarget priority:")
print(f"  Min:  {simple['target_priority'].min():.3f}")
print(f"  Max:  {simple['target_priority'].max():.3f}")
print(f"  Mean: {simple['target_priority'].mean():.3f}")

# =============================================================================
# 4. CHECK DATA INTEGRATION
# =============================================================================

print("\n[4] Checking data source integration...")

# Sample a few DAs and show all features
sample_das = simple.sample(5, random_state=42)

print("\nSample of 5 DAs with all features:")
print(sample_das[['DAUID', 'heat_norm', 'tree_norm', 'vuln_norm', 'footfall_norm', 'target_priority']].to_string())

# Check if any DA has all zeros (would indicate missing data)
all_zero = simple[(simple['heat_norm'] == 0) & 
                  (simple['tree_norm'] == 0) & 
                  (simple['vuln_norm'] == 0) & 
                  (simple['footfall_norm'] == 0)]

if len(all_zero) > 0:
    print(f"\n⚠️  WARNING: {len(all_zero)} DAs have all features = 0")
else:
    print(f"\n✓ No DAs with all features = 0")

# =============================================================================
# 5. CHECK FOOTFALL IMPUTATION
# =============================================================================

print("\n[5] Checking footfall imputation...")

# Load full dataset to check imputation
full = gpd.read_file('data/processed/toronto_data_web.geojson')

# Count DAs with very low footfall (likely imputed)
low_footfall = full[full['footfall_norm'] < 0.05]
print(f"\nDAs with footfall_norm < 0.05: {len(low_footfall)} ({len(low_footfall)/len(full)*100:.1f}%)")
print(f"  These were likely imputed via KNN from neighbors")

# Show distribution
print(f"\nFootfall distribution:")
print(full['footfall_norm'].describe())

# =============================================================================
# 6. FEATURE CORRELATION CHECK
# =============================================================================

print("\n[6] Checking feature correlations...")

correlation = simple[norm_cols].corr()
print("\nCorrelation matrix (values close to 1 or -1 indicate high correlation):")
print(correlation.round(3))

# Flag high correlations (>0.7)
print("\nHigh correlations (>0.7):")
high_corr = []
for i in range(len(norm_cols)):
    for j in range(i+1, len(norm_cols)):
        corr_val = abs(correlation.iloc[i, j])
        if corr_val > 0.7:
            high_corr.append((norm_cols[i], norm_cols[j], corr_val))

if high_corr:
    for feat1, feat2, corr in high_corr:
        print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
else:
    print("  ✓ No high correlations found (features are independent)")

# =============================================================================
# 7. SUMMARY
# =============================================================================

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

checks_passed = 0
total_checks = 7

print("\n✓ Data loaded successfully")
checks_passed += 1

if features.shape[0] == 3702 and target.shape[0] == 3702:
    print("✓ All datasets have 3702 DAs")
    checks_passed += 1
else:
    print("✗ Dataset size mismatch")

if features.isnull().sum().sum() == 0 and target.isnull().sum().sum() == 0:
    print("✓ No missing values")
    checks_passed += 1
else:
    print("✗ Missing values detected")

if all(simple[col].min() >= 0 and simple[col].max() <= 1 for col in norm_cols):
    print("✓ All features properly normalized (0-1)")
    checks_passed += 1
else:
    print("✗ Some features not in [0,1] range")

if simple.crs.to_string() == 'EPSG:4326':
    print("✓ Correct CRS for web mapping")
    checks_passed += 1
else:
    print("✗ Wrong CRS")

if len(all_zero) == 0:
    print("✓ All DAs have feature data")
    checks_passed += 1
else:
    print(f"✗ {len(all_zero)} DAs missing all features")

if len(high_corr) == 0:
    print("✓ Features are independent (low correlation)")
    checks_passed += 1
else:
    print(f"⚠️  {len(high_corr)} high feature correlations detected")

print(f"\n{'='*70}")
print(f"RESULT: {checks_passed}/{total_checks} checks passed")

if checks_passed == total_checks:
    print("✅ ALL CHECKS PASSED - Data is ready for ML training!")
else:
    print("⚠️  Some issues detected - review warnings above")

print("="*70)
