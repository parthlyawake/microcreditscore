"""
Step 4: Train Neural Network (Optimized Regular NN)
Days 7–9 of implementation

⭐ STRONG TABULAR BASELINE ⭐

Uses a carefully optimized feed-forward neural network
to compete with XGBoost on credit scoring.

Focus:
- Better feature interactions
- Stable training
- Strong weighted F1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("STEP 4: TRAIN REGULAR NEURAL NETWORK (OPTIMIZED)")
print("=" * 80)

# =============================================================================
# SETUP
# =============================================================================

Path("models").mkdir(exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("\n📂 Loading processed features...")
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_target.csv")["credit_score"]

print(f"✅ Loaded: {len(X):,} samples × {len(X.columns)} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 2. FEATURE SCALING
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/regular_scaler.pkl")
print("✅ Features standardized")
print("✅ Saved scaler: models/regular_scaler.pkl")

# =============================================================================
# 3. DATA LOADERS
# =============================================================================

print("\n" + "=" * 80)
print("PREPARING PYTORCH DATASETS")
print("=" * 80)

train_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train.values)
    ),
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test.values)
    ),
    batch_size=128,
    shuffle=False
)

print(f"✅ Batch size: 128")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# =============================================================================
# 4. MODEL
# =============================================================================

print("\n" + "=" * 80)
print("DEFINING NEURAL NETWORK ARCHITECTURE")
print("=" * 80)

input_size = X_train_scaled.shape[1]
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

model = RegularNN(input_size, num_classes).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Input: {input_size}")
print(f"   Hidden: 512 → 256 → 128")
print(f"   Output: {num_classes}")
print(f"   Total parameters: {total_params:,}")

# =============================================================================
# 5. TRAINING SETUP
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING SETUP")
print("=" * 80)

class_counts = y_train.value_counts().sort_index()
total = len(y_train)
class_weights = torch.FloatTensor(
    [total / (len(class_counts) * c) for c in class_counts]
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 80
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)

print("✅ Loss: CrossEntropyLoss (class-weighted + label smoothing)")
print("✅ Optimizer: Adam")
print("✅ Scheduler: CosineAnnealingLR")

# =============================================================================
# 6. TRAINING LOOP (WITH EARLY STOPPING)
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING REGULAR NEURAL NETWORK")
print("=" * 80)

best_f1 = 0.0
best_state = None
epochs_no_improve = 0
patience = 10
epochs_trained = 0

train_losses, test_losses = [], []
test_accuracies, test_f1_scores = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item()
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    test_accuracies.append(acc)
    test_f1_scores.append(f1)

    scheduler.step()
    epochs_trained = epoch + 1

    if f1 > best_f1:
        best_f1 = f1
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Acc: {acc:.4f} | F1: {f1:.4f}"
        )

    if epochs_no_improve >= patience:
        print(f"\n⏹ Early stopping at epoch {epoch+1}")
        break

print(f"\n✅ Training complete! Best F1: {best_f1:.4f}")
model.load_state_dict(best_state)

# =============================================================================
# 7. FINAL EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

model.eval()
preds, labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds.extend(model(xb).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())

accuracy = accuracy_score(labels, preds)
f1_weighted = f1_score(labels, preds, average="weighted")
precision, recall, f1_cls, support = precision_recall_fscore_support(labels, preds)

print(f"\n📊 Accuracy: {accuracy:.4f}")
print(f"📊 F1-Score (weighted): {f1_weighted:.4f}")

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(labels, preds, target_names=["Poor", "Standard", "Good"]))

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================

torch.save(model.state_dict(), "models/regular_model.pth")

metrics = {
    "model": "RegularNN",
    "accuracy": accuracy,
    "f1_weighted": f1_weighted,
    "f1_poor": f1_cls[0],
    "f1_standard": f1_cls[1],
    "f1_good": f1_cls[2],
    "params": total_params,
    "epochs": epochs_trained
}

pd.DataFrame([metrics]).to_csv("models/regular_metrics.csv", index=False)

print("\nSTEP 4 COMPLETE! ✅")