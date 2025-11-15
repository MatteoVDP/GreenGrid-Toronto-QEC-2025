"""
Random Forest Regression Model for Toronto Tree Planting Priority
Trains model using preprocessed features and synthetic ground truth labels
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt

print("="*70)
print("RANDOM FOREST REGRESSION MODEL TRAINING")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("\n[1/8] Loading preprocessed data...")

features = pd.read_csv('data/processed/features.csv')
target = pd.read_csv('data/processed/target.csv')

print(f"  ✓ Features: {features.shape}")
print(f"  ✓ Target: {target.shape}")

# =============================================================================
# STEP 2: PREPARE FEATURES
# =============================================================================

print("\n[2/8] Preparing features...")

# Use normalized features
X = features[['heat_norm', 'tree_norm', 'vuln_norm', 'footfall_norm']].copy()

# Invert tree_norm so lower tree coverage = higher priority
X['tree_norm_inv'] = 1 - X['tree_norm']
X = X.drop('tree_norm', axis=1)

# Reorder columns
X = X[['heat_norm', 'tree_norm_inv', 'vuln_norm', 'footfall_norm']]

y = target['target_priority'].values

print(f"  ✓ Feature matrix: {X.shape}")
print(f"  ✓ Features: {list(X.columns)}")
print(f"  ✓ Target range: [{y.min():.3f}, {y.max():.3f}]")

# =============================================================================
# STEP 3: TRAIN/TEST SPLIT
# =============================================================================

print("\n[3/8] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  ✓ Training set: {X_train.shape[0]} samples")
print(f"  ✓ Test set: {X_test.shape[0]} samples")

# =============================================================================
# STEP 4: TRAIN RANDOM FOREST MODEL
# =============================================================================

print("\n[4/8] Training Random Forest Regressor...")

# Initialize model
rf_model = RandomForestRegressor(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Max depth of trees
    min_samples_split=5,   # Min samples to split node
    min_samples_leaf=2,    # Min samples in leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

# Train model
rf_model.fit(X_train, y_train)

print(f"  ✓ Model trained with {rf_model.n_estimators} trees")

# =============================================================================
# STEP 5: EVALUATE MODEL
# =============================================================================

print("\n[5/8] Evaluating model performance...")

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Metrics - Training
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Metrics - Test
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n  Training Performance:")
print(f"    R² Score:  {train_r2:.4f}")
print(f"    MAE:       {train_mae:.4f}")
print(f"    RMSE:      {train_rmse:.4f}")

print(f"\n  Test Performance:")
print(f"    R² Score:  {test_r2:.4f}")
print(f"    MAE:       {test_mae:.4f}")
print(f"    RMSE:      {test_rmse:.4f}")

# Cross-validation
print(f"\n  Cross-validation (5-fold)...")
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"    CV R² Scores: {cv_scores}")
print(f"    CV Mean:      {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# STEP 6: FEATURE IMPORTANCE
# =============================================================================

print("\n[6/8] Analyzing feature importance...")

importances = rf_model.feature_importances_
feature_names = X.columns.tolist()

print(f"\n  Feature Importance:")
for name, importance in sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True):
    print(f"    {name:20s}: {importance:.4f} ({importance*100:.1f}%)")

# =============================================================================
# STEP 7: SAVE MODEL AND RESULTS
# =============================================================================

print("\n[7/8] Saving model and results...")

# Create output directory
os.makedirs('ml_model', exist_ok=True)

# Save trained model
joblib.dump(rf_model, 'ml_model/tree_priority_model.pkl')
print(f"  ✓ Model saved: ml_model/tree_priority_model.pkl")

# Save feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
importance_df.to_csv('ml_model/feature_importance.csv', index=False)
print(f"  ✓ Feature importance saved: ml_model/feature_importance.csv")

# Save test predictions
test_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'error': y_test - y_test_pred
})
test_results.to_csv('ml_model/test_predictions.csv', index=False)
print(f"  ✓ Test predictions saved: ml_model/test_predictions.csv")

# Save model metrics
metrics = {
    'train_r2': train_r2,
    'train_mae': train_mae,
    'train_rmse': train_rmse,
    'test_r2': test_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'cv_mean_r2': cv_scores.mean(),
    'cv_std_r2': cv_scores.std()
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('ml_model/model_metrics.csv', index=False)
print(f"  ✓ Metrics saved: ml_model/model_metrics.csv")

# =============================================================================
# STEP 8: GENERATE PREDICTIONS FOR ALL DAs
# =============================================================================

print("\n[8/8] Generating predictions for all DAs...")

# Predict on entire dataset
all_predictions = rf_model.predict(X)

# Save predictions
predictions_df = pd.DataFrame({
    'DAUID': target['DAUID'],
    'ml_priority_score': all_predictions,
    'baseline_priority_score': y
})
predictions_df.to_csv('ml_model/all_predictions.csv', index=False)
print(f"  ✓ All predictions saved: ml_model/all_predictions.csv")

# Compare ML vs baseline
comparison = predictions_df[['ml_priority_score', 'baseline_priority_score']].describe()
print(f"\n  Score comparison (ML vs Baseline):")
print(comparison)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\n📊 Model Performance:")
print(f"  Test R² Score: {test_r2:.4f}")
print(f"  Test MAE:      {test_mae:.4f}")
print(f"  Test RMSE:     {test_rmse:.4f}")

if test_r2 > 0.95:
    print(f"\n✅ EXCELLENT - Model explains {test_r2*100:.1f}% of variance")
elif test_r2 > 0.85:
    print(f"\n✅ GOOD - Model explains {test_r2*100:.1f}% of variance")
elif test_r2 > 0.7:
    print(f"\n⚠️  FAIR - Model explains {test_r2*100:.1f}% of variance")
else:
    print(f"\n❌ POOR - Model only explains {test_r2*100:.1f}% of variance")

print(f"\n🎯 Top Feature: {feature_names[importances.argmax()]} ({importances.max()*100:.1f}%)")

print(f"\n📁 Output Files:")
print(f"  1. tree_priority_model.pkl    - Trained model (use for predictions)")
print(f"  2. feature_importance.csv     - Feature importance rankings")
print(f"  3. test_predictions.csv       - Test set predictions vs actual")
print(f"  4. model_metrics.csv          - Performance metrics")
print(f"  5. all_predictions.csv        - ML predictions for all 3,702 DAs")

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70)

print("\nNext steps:")
print("  1. Review feature_importance.csv to understand what drives priority")
print("  2. Use tree_priority_model.pkl in backend API for predictions")
print("  3. Compare ml_priority_score vs baseline_priority_score")
print("  4. Integrate with frontend for visualization")

print("="*70)
