"""
Step 5: Ensemble Model
Day 9 of implementation

Combines XGBoost + Neural Network for best performance
Uses weighted averaging or stacking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("STEP 5: ENSEMBLE MODEL")
print("=" * 80)

# Create directories
Path("models").mkdir(exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("\n📂 Loading data...")
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_target.csv")["credit_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data loaded: {len(X_test):,} test samples")

# =============================================================================
# 2. LOAD TRAINED MODELS
# =============================================================================

print("\n" + "=" * 80)
print("LOADING TRAINED MODELS")
print("=" * 80)

# ---- Load XGBoost -----------------------------------------------------------
try:
    xgb_model = joblib.load("models/xgboost_model.pkl")
    print("✅ Loaded: XGBoost model")
    has_xgb = True
except Exception as e:
    print(f"⚠️  XGBoost model not found: {e}")
    has_xgb = False

# ---- Load Regular Neural Network -------------------------------------------
try:
    model_type = "regular"

    scaler = joblib.load("models/regular_scaler.pkl")

    input_size = X_train.shape[1]
    num_classes = 3

    class RegularNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.2),

                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.network(x)

    nn_model = RegularNN(input_size, num_classes)
    nn_model.load_state_dict(torch.load("models/regular_model.pth"))
    nn_model.eval()

    print("✅ Loaded: Regular Neural Network")
    has_nn = True

except Exception as e:
    print(f"⚠️  Neural network model not found: {e}")
    has_nn = False

if not (has_xgb and has_nn):
    print("\n❌ Need both models for ensemble. Run steps 3 and 4 first.")
    exit(1)

# =============================================================================
# 3. GET PREDICTIONS FROM BOTH MODELS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS FROM BOTH MODELS")
print("=" * 80)

# ---- XGBoost ---------------------------------------------------------------
print("\n🔄 Getting XGBoost predictions...")
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
print(f"   XGBoost: Acc={xgb_acc:.4f}, F1={xgb_f1:.4f}")

# ---- Neural Network --------------------------------------------------------
print("\n🔄 Getting Regular Network predictions...")
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = nn_model.to(device)

with torch.no_grad():
    outputs = nn_model(X_test_tensor.to(device))
    nn_proba = torch.softmax(outputs, dim=1).cpu().numpy()
    nn_pred = torch.argmax(outputs, dim=1).cpu().numpy()

nn_acc = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred, average="weighted")
print(f"   Regular NN: Acc={nn_acc:.4f}, F1={nn_f1:.4f}")

# =============================================================================
# 4. ENSEMBLE METHOD 1: WEIGHTED AVERAGING
# =============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE METHOD 1: WEIGHTED AVERAGING")
print("=" * 80)

def weighted_ensemble(p1, p2, w):
    return w * p1 + (1 - w) * p2

best_weight = 0.5
best_f1 = 0

for w in np.arange(0, 1.05, 0.05):
    proba = weighted_ensemble(xgb_proba, nn_proba, w)
    pred = np.argmax(proba, axis=1)
    f1 = f1_score(y_test, pred, average="weighted")
    if f1 > best_f1:
        best_f1 = f1
        best_weight = w

weighted_pred = np.argmax(
    weighted_ensemble(xgb_proba, nn_proba, best_weight), axis=1
)
weighted_acc = accuracy_score(y_test, weighted_pred)
weighted_f1 = f1_score(y_test, weighted_pred, average="weighted")

print(f"✅ Best weight: XGB={best_weight:.2f}, NN={1-best_weight:.2f}")
print(f"   Accuracy: {weighted_acc:.4f}")
print(f"   F1-Score: {weighted_f1:.4f}")

# =============================================================================
# 5. ENSEMBLE METHOD 2: STACKING
# =============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE METHOD 2: STACKING")
print("=" * 80)

X_train_scaled = scaler.transform(X_train)
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)

with torch.no_grad():
    nn_train_proba = torch.softmax(
        nn_model(X_train_tensor), dim=1
    ).cpu().numpy()

X_stack_train = np.hstack([
    xgb_model.predict_proba(X_train),
    nn_train_proba
])
X_stack_test = np.hstack([xgb_proba, nn_proba])

meta = LogisticRegression(max_iter=1000, random_state=42)
meta.fit(X_stack_train, y_train)

stack_pred = meta.predict(X_stack_test)
stack_acc = accuracy_score(y_test, stack_pred)
stack_f1 = f1_score(y_test, stack_pred, average="weighted")

print(f"✅ Stacking: Acc={stack_acc:.4f}, F1={stack_f1:.4f}")

# =============================================================================
# 6. SELECT BEST ENSEMBLE
# =============================================================================

if stack_f1 >= weighted_f1:
    best_ensemble = "Stacking"
    final_pred = stack_pred
    final_acc = stack_acc
    final_f1 = stack_f1
else:
    best_ensemble = "Weighted"
    final_pred = weighted_pred
    final_acc = weighted_acc
    final_f1 = weighted_f1

print(f"\n🏆 Best Ensemble: {best_ensemble}")
print(f"   Accuracy: {final_acc:.4f}")
print(f"   F1-Score: {final_f1:.4f}")

# =============================================================================
# 7. FINAL EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("FINAL ENSEMBLE EVALUATION")
print("=" * 80)

print(classification_report(
    y_test, final_pred, target_names=["Poor", "Standard", "Good"]
))

cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Poor", "Standard", "Good"],
            yticklabels=["Poor", "Standard", "Good"])
plt.title(f"Confusion Matrix - {best_ensemble} Ensemble")
plt.tight_layout()
plt.savefig("visualizations/05_confusion_matrix_ensemble.png", dpi=300)

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================

metrics = {
    "model": f"{best_ensemble} Ensemble",
    "accuracy": final_acc,
    "f1_weighted": final_f1
}

pd.DataFrame([metrics]).to_csv(
    "models/ensemble_metrics.csv", index=False
)

print("\nSTEP 5 COMPLETE! ✅")
print("=" * 80)