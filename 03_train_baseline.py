"""
Step 3: Train XGBoost Baseline
Days 5-6 of implementation

Trains gradient boosting baseline with:
- Hyperparameter tuning (Optuna)
- Cross-validation
- Feature importance analysis
- Performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_recall_fscore_support)
import xgboost as xgb
import optuna
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create directories
Path('models').mkdir(exist_ok=True)
Path('visualizations').mkdir(exist_ok=True)

print("="*80)
print("STEP 3: TRAIN XGBOOST BASELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n📂 Loading processed features...")
X = pd.read_csv('data/X_features.csv')
y = pd.read_csv('data/y_target.csv')['credit_score']

print(f"✅ Loaded: {len(X):,} samples × {len(X.columns)} features")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

# 80-20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set:  {len(X_test):,} samples")

print("\nTrain distribution:")
for label, count in y_train.value_counts().sort_index().items():
    label_name = {0: 'Poor', 1: 'Standard', 2: 'Good'}[label]
    print(f"   {label_name:10s}: {count:,}")

print("\nTest distribution:")
for label, count in y_test.value_counts().sort_index().items():
    label_name = {0: 'Poor', 1: 'Standard', 2: 'Good'}[label]
    print(f"   {label_name:10s}: {count:,}")

# ============================================================================
# 3. BASELINE MODEL (No Tuning)
# ============================================================================

print("\n" + "="*80)
print("BASELINE MODEL (Default Parameters)")
print("="*80)

# Calculate class weights for imbalance
class_counts = y_train.value_counts().sort_index()
total = len(y_train)
class_weights = {i: total / (len(class_counts) * count) for i, count in class_counts.items()}

print(f"\nClass weights (for imbalance): {class_weights}")

# Train baseline
print("\n🔄 Training baseline XGBoost...")
baseline_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='mlogloss'
)

baseline_model.fit(X_train, y_train)

# Evaluate
y_pred_baseline = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted')

print(f"\n✅ Baseline Results:")
print(f"   Accuracy: {baseline_acc:.4f}")
print(f"   F1-Score: {baseline_f1:.4f}")

# ============================================================================
# 4. HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================================

print("\n" + "="*80)
print("HYPERPARAMETER TUNING (Optuna)")
print("="*80)

def objective(trial):
    """Optuna objective function"""
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        
        # Hyperparameters to tune
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
    }
    
    # Cross-validation
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()

print("\n🔍 Running Optuna optimization...")
print("   Target: 50 trials (adjust based on time)")
print("   This will take 5-15 minutes...\n")

# Create study
study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')

# Run optimization
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n✅ Optimization complete!")
print(f"\nBest F1-Score: {study.best_value:.4f}")
print(f"\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"   {key:20s}: {value}")

# ============================================================================
# 5. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("TRAINING FINAL MODEL")
print("="*80)

# Build final model
final_params = study.best_params.copy()
final_params.update({
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42
})

print("\n🔄 Training final XGBoost model...")
final_model = xgb.XGBClassifier(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ============================================================================
# 6. EVALUATE FINAL MODEL
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Predictions
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Per-class metrics
precision, recall, f1_per_class, support = precision_recall_fscore_support(y_test, y_pred)

print(f"\n📊 Overall Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   F1-Score:  {f1:.4f}")

print(f"\n📊 Per-Class Metrics:")
class_names = ['Poor', 'Standard', 'Good']
for i, name in enumerate(class_names):
    print(f"\n   {name}:")
    print(f"      Precision: {precision[i]:.4f}")
    print(f"      Recall:    {recall[i]:.4f}")
    print(f"      F1-Score:  {f1_per_class[i]:.4f}")
    print(f"      Support:   {support[i]}")

# AUC-ROC (multi-class, one-vs-rest)
try:
    auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    auc_str = f"{auc_ovr:.4f}" if 'auc_ovr' in locals() else "N/A"
    print(f"\n   AUC-ROC (weighted): {auc_str}")
except:
    print("\n   AUC-ROC: Could not calculate")

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================================
# 7. CONFUSION MATRIX
# ============================================================================

print("="*80)
print("CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_test, y_pred)
print("\n", cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('visualizations/03_confusion_matrix_xgboost.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: visualizations/03_confusion_matrix_xgboost.png")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
for i, row in feature_importance.head(20).iterrows():
    print(f"   {row['feature']:40s} {row['importance']:.4f}")

# Visualize top 20 features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance')
plt.title('Top 20 Feature Importance - XGBoost', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/03_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: visualizations/03_feature_importance.png")

# ============================================================================
# 9. SAVE MODEL AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL AND RESULTS")
print("="*80)

# Save model
joblib.dump(final_model, 'models/xgboost_model.pkl')
print("✅ Saved: models/xgboost_model.pkl")

# Save feature importance
feature_importance.to_csv('models/xgboost_feature_importance.csv', index=False)
print("✅ Saved: models/xgboost_feature_importance.csv")

# Save metrics
metrics = {
    'model': 'XGBoost',
    'accuracy': accuracy,
    'f1_weighted': f1,
    'f1_poor': f1_per_class[0],
    'f1_standard': f1_per_class[1],
    'f1_good': f1_per_class[2],
    'precision_poor': precision[0],
    'precision_standard': precision[1],
    'precision_good': precision[2],
    'recall_poor': recall[0],
    'recall_standard': recall[1],
    'recall_good': recall[2],
}

pd.DataFrame([metrics]).to_csv('models/xgboost_metrics.csv', index=False)
print("✅ Saved: models/xgboost_metrics.csv")

# Save best parameters
with open('models/xgboost_best_params.txt', 'w') as f:
    f.write("XGBoost Best Parameters\n")
    f.write("="*50 + "\n\n")
    for key, value in study.best_params.items():
        f.write(f"{key:20s}: {value}\n")
    f.write(f"\nBest CV F1-Score: {study.best_value:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test F1-Score: {f1:.4f}\n")
print("✅ Saved: models/xgboost_best_params.txt")

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("XGBoost BASELINE SUMMARY")
print("="*80)

print(f"""
✅ XGBoost Model Training Complete!

Performance Metrics:
  - Accuracy:  {accuracy:.4f}
  - F1-Score:  {f1:.4f}
  - AUC-ROC:   {auc_str}

Per-Class F1-Scores:
  - Poor:      {f1_per_class[0]:.4f}
  - Standard:  {f1_per_class[1]:.4f}
  - Good:      {f1_per_class[2]:.4f}

Model Details:
  - Best params found via Optuna (50 trials)
  - Trained on {len(X_train):,} samples
  - Tested on {len(X_test):,} samples
  - {len(X.columns)} features used

Top 3 Most Important Features:
  1. {feature_importance.iloc[0]['feature']}
  2. {feature_importance.iloc[1]['feature']}
  3. {feature_importance.iloc[2]['feature']}

Next Steps:
  1. Run: python 04_train_dendrites.py
  2. Train dendritic neural network
  3. Compare with XGBoost baseline
""")

print("="*80)
print("STEP 3 COMPLETE! ✅")
print("="*80)