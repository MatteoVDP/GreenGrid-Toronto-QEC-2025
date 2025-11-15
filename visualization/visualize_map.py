"""
Visualize Toronto tree planting priority map with color-coded DAs
Green gradient: Light green = low priority, Dark green = high priority
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("="*70)
print("GENERATING TORONTO TREE PLANTING PRIORITY MAP")
print("="*70)

# Load GeoJSON with predictions
print("\n[1/4] Loading data...")
gdf = gpd.read_file('data/processed/toronto_with_predictions.geojson')
print(f"  Loaded {len(gdf)} DAs")
print(f"  ML priority scores: {gdf['ml_priority_score'].min():.3f} - {gdf['ml_priority_score'].max():.3f}")

# Create figure
print("\n[2/4] Creating visualization...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Plot with green gradient
gdf.plot(
    column='ml_priority_score',
    cmap='Greens',  # Light green to dark green
    linewidth=0.3,
    edgecolor='white',
    legend=True,
    ax=ax,
    legend_kwds={
        'label': "Tree Planting Priority Score",
        'orientation': "vertical",
        'shrink': 0.6
    }
)

# Styling
ax.set_title('Toronto Tree Planting Priority - ML Model Predictions', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_aspect('equal')

# Add text annotations
textstr = '\n'.join([
    'Color intensity = Priority level',
    'Light green = Low priority',
    'Dark green = High priority',
    '',
    f'Total areas: {len(gdf):,}',
    f'Score range: {gdf["ml_priority_score"].min():.3f} - {gdf["ml_priority_score"].max():.3f}'
])
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Save
print("\n[3/4] Saving map...")
output_path = 'visualization/toronto_priority_map.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  Saved to: {output_path}")

# Display statistics
print("\n[4/4] Priority Statistics...")
print(f"  Mean priority: {gdf['ml_priority_score'].mean():.3f}")
print(f"  Median priority: {gdf['ml_priority_score'].median():.3f}")
print(f"  Std dev: {gdf['ml_priority_score'].std():.3f}")

# Count by priority level
low = (gdf['ml_priority_score'] < 0.33).sum()
medium = ((gdf['ml_priority_score'] >= 0.33) & (gdf['ml_priority_score'] < 0.67)).sum()
high = (gdf['ml_priority_score'] >= 0.67).sum()

print(f"\n  Priority Distribution:")
print(f"    Low (0.0-0.33):    {low:4d} DAs ({low/len(gdf)*100:.1f}%)")
print(f"    Medium (0.33-0.67): {medium:4d} DAs ({medium/len(gdf)*100:.1f}%)")
print(f"    High (0.67-1.0):   {high:4d} DAs ({high/len(gdf)*100:.1f}%)")

print("\n" + "="*70)
print("MAP VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nOpen the map at: {output_path}")
print("\nDark green areas = Highest priority for tree planting")
print("Light green areas = Lower priority")
