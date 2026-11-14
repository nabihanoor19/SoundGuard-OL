import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
import json
import random
from pathlib import Path

import dagshub
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.cnn import AcousticCNN

# -----------------------------
# 1. Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# -----------------------------
# 2. Load processed dataset
# -----------------------------
X = np.load("data/processed/features.npy")
y = np.load("data/processed/labels.npy")
folds = np.load("data/processed/folds.npy")

# Add channel dimension for CNN
X = X[:, np.newaxis, :, :].astype(np.float32)

# Normalize each audio sample
mean = X.mean(axis=(2, 3), keepdims=True)
std = X.std(axis=(2, 3), keepdims=True)
X = (X - mean) / (std + 1e-6)

# ESC-50: folds 1-4 for training, fold 5 for testing
train_mask = folds != 5
test_mask = folds == 5

X_train = torch.tensor(X[train_mask], dtype=torch.float32)
y_train = torch.tensor(y[train_mask], dtype=torch.long)

X_test = torch.tensor(X[test_mask], dtype=torch.float32)
y_test = torch.tensor(y[test_mask], dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -----------------------------
# 3. Initialize CNN
# -----------------------------
model = AcousticCNN(num_classes=50).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# 4. Train model
# -----------------------------
print("MAIN CNN TRAINING STARTED")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# -----------------------------
# 5. Evaluate on Fold 5
# -----------------------------
model.eval()
predictions = []

with torch.no_grad():

    for start in range(0, len(X_test), BATCH_SIZE):

        batch = X_test[start:start + BATCH_SIZE].to(DEVICE)

        outputs = model(batch)
        predicted = torch.argmax(outputs, dim=1)

        predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_test.numpy(), predictions)

macro_f1 = f1_score(
    y_test.numpy(),
    predictions,
    average="macro",
    zero_division=0
)

print("\nMAIN CNN EVALUATION")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")


# -----------------------------
# 6. Save model and metrics
# -----------------------------
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

model_path = "models/main_cnn.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_classes": 50,
        "seed": SEED
    },
    model_path
)

metrics = {
    "model": "AcousticCNN",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "epochs": EPOCHS,
    "test_fold": 5
}

with open("reports/main_cnn_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# -----------------------------
# 7. Log experiment to MLflow
# -----------------------------
dagshub.init(
    repo_owner="nabihanoorkhalique",
    repo_name="SoundGuard-OL",
    mlflow=True
)

mlflow.set_experiment("SoundGuard-OL")

with mlflow.start_run(run_name="Main_CNN"):

    mlflow.log_params({
        "model": "AcousticCNN",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "training_folds": "1,2,3,4",
        "testing_fold": 5,
        "seed": SEED
    })

    mlflow.log_metrics({
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1)
    })

    mlflow.log_artifact(model_path)
    mlflow.log_artifact("reports/main_cnn_metrics.json")

print("\nMAIN CNN PIPELINE COMPLETED SUCCESSFULLY")
