import os
import time
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, efficientnet_b0, vit_b_32
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    ViT_B_32_Weights,
)

from sklearn.model_selection import StratifiedShuffleSplit

# -----------------------------------------
# Argument parser for selecting the model
# -----------------------------------------
# Removed the argument parser as we are running all models at once

# Get absolute path of src/ directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(SRC_DIR, "metrics.csv")
ARCHIVE_FOLDER = os.path.join(SRC_DIR, "ISIC_Archive")

# Ensure data is available inside src/
if not os.path.exists(METRICS_FILE) or not os.path.exists(ARCHIVE_FOLDER):
    print("Data missing, downloading now...")
    subprocess.run(
        ["python3", os.path.join(SRC_DIR, "utils/download_data.py")], check=True
    )


# -----------------------------------------
# EarlyStopping Implementation
# -----------------------------------------
class EarlyStopping:
    # Early stops the training if validation loss doesn't improve after a given patience.

    def __init__(
        self,
        patience=3,
        verbose=False,
        delta=0,
        path="best_model.pth",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # Using np.inf (lowercase) for NumPy
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# -----------------------------------------
# 1. Load & Prepare Data
# -----------------------------------------
df = pd.read_csv(METRICS_FILE)
# Create a bin column for stratified splitting using the 'dc_mean' column (dividing into 5 bins: [0,0.2], (0.2,0.4], etc.)
df["d_bin"] = pd.cut(
    df["dc_mean"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False, include_lowest=True
)

# Stratified split into train+val and test
test_size = 0.15
split1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
for train_val_idx, test_idx in split1.split(df, df["d_bin"]):
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

# Split train+val into train and validation (~17.65% of train+val for validation to achieve ~15% overall)
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
for train_idx, val_idx in split2.split(df_train_val, df_train_val["d_bin"]):
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)


# -----------------------------------------
# 2. Dataset Definition
# -----------------------------------------
class ImageRegressionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["img_name"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["dc_mean"], dtype=torch.float32)
        return image, label


# -----------------------------------------
# 3. Define Transforms & Create DataLoaders
# -----------------------------------------
# Training transform with data augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)
# Validation and test transform (no augmentation)
val_test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

img_dir = ARCHIVE_FOLDER  # 'ISIC_Archive' directory (downloaded & extracted)
batch_size = 32

train_dataset = ImageRegressionDataset(df_train, img_dir, transform=train_transform)
val_dataset = ImageRegressionDataset(df_val, img_dir, transform=val_test_transform)
test_dataset = ImageRegressionDataset(df_test, img_dir, transform=val_test_transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


# -----------------------------------------
# 4. Model Selection Function
# -----------------------------------------
def get_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
    elif model_name == "vit_b_32":
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, 1)
    else:
        raise ValueError("Model not recognized.")
    return model.to(device)


# -----------------------------------------
# 5. Training & Evaluation Functions (with error standard deviation)
# -----------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    mse = np.mean((all_preds - all_labels) ** 2)
    mae = np.mean(np.abs(all_preds - all_labels))
    std_mse = np.std((all_preds - all_labels) ** 2)
    std_mae = np.std(np.abs(all_preds - all_labels))
    return epoch_loss, mse, mae, std_mse, std_mae


# -----------------------------------------
# 6. Training Function with Early Stopping & LR Scheduler
# -----------------------------------------
def train_model(model, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.1
    )
    early_stopping = EarlyStopping(patience=3, verbose=True)

    total_start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mse, val_mae, val_std_mse, val_std_mae = evaluate(
            model, val_loader, criterion, device
        )
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f} (std: {val_std_mse:.4f}), "
            f"MAE: {val_mae:.4f} (std: {val_std_mae:.4f})"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print(f"\nTotal training time: {time.time() - total_start_time:.2f}s")
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_mse, test_mae, test_std_mse, test_std_mae = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"\nTest Loss: {test_loss:.4f}, MSE: {test_mse:.4f} (std: {test_std_mse:.4f}), "
        f"MAE: {test_mae:.4f} (std: {test_std_mae:.4f})"
    )


# -----------------------------------------
# 7. Run Training for All Models
# -----------------------------------------
if __name__ == "__main__":
    models_to_run = ["resnet18", "efficientnet_b0", "vit_b_32"]  # List of models to run

    for model_name in models_to_run:
        print(f"Running {model_name} model...")

        # Get the model
        model = get_model(model_name)  # Create the model

        # Train the model (using the model specified in the loop)
        train_model(model, num_epochs=10)
