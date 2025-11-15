"""
Add ML predictions to GeoJSON for frontend visualization
"""
import geopandas as gpd
import pandas as pd
import json

print("="*70)
print("ADDING ML PREDICTIONS TO GEOJSON")
print("="*70)

# Load GeoJSON and predictions
print("\n[1/4] Loading data...")
gdf = gpd.read_file('data/processed/toronto_simple.geojson')
predictions = pd.read_csv('ml_model/all_predictions.csv')
print(f"  ✓ Loaded {len(gdf)} DAs from GeoJSON")
print(f"  ✓ Loaded {len(predictions)} predictions")

# Merge predictions into GeoJSON
print("\n[2/4] Merging predictions...")
gdf_with_predictions = gdf.merge(predictions, on='DAUID', how='left')
print(f"  ✓ Merged successfully")
print(f"  Columns: {gdf_with_predictions.columns.tolist()}")

# Save updated GeoJSON
print("\n[3/4] Saving updated GeoJSON...")
output_path = 'data/processed/toronto_with_predictions.geojson'
gdf_with_predictions.to_file(output_path, driver='GeoJSON')
print(f"  ✓ Saved to: {output_path}")

# Verify
print("\n[4/4] Verification...")
file_size_mb = round(len(open(output_path).read()) / 1024 / 1024, 2)
print(f"  File size: {file_size_mb} MB")
print(f"  ML scores range: {gdf_with_predictions['ml_priority_score'].min():.3f} - {gdf_with_predictions['ml_priority_score'].max():.3f}")

print("\n" + "="*70)
print("✅ GEOJSON READY FOR FRONTEND!")
print("="*70)
print(f"\nFile location: {output_path}")
print("\nFrontend can now:")
print("  1. Load this GeoJSON")
print("  2. Color DAs by 'ml_priority_score' property")
print("  3. Show tooltips with heat, tree_coverage, vulnerability, footfall")
