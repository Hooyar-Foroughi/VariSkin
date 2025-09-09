# Library imports
import os
import json
import time
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    vit_b_32,
    ViT_B_32_Weights,
)

# Local module imports
from utils import gradcam_utils
from utils.download_data import download_data
from analysis.histograms import dice_mean_histograms
from analysis.diagnosis_correlation import correlation
from analysis.scatter.plot_scatter import plot_scatter
from analysis.experiment import MAE_normalization, MSE_normalization
from analysis.error_visuals.display_error import display_extreme_error_images


# Load the configuration from a YAML file. This allows the user to toggle features and set paths.
def load_config(config_path="config.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_full_path = os.path.join(script_dir, config_path)
    with open(config_full_path, "r") as f:
        return yaml.safe_load(f)


# SECTION: EarlyStopping Implementation
# Implements early stopping to halt training when validation loss stops improving.
# Saves the best model based on validation loss.
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
        # Saves model when validation loss decreases.
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# SECTION: Load & Prepare Data
# Stratified binning of 'dc_mean' into 5 bins ensures consistent distribution across train/val/test
def load_and_split_data(metrics_file):
    df = pd.read_csv(metrics_file)
    # Create a bin column for stratified splitting using the 'dc_mean' column (dividing into 5 bins: [0,0.2], (0.2,0.4], etc.)
    df["d_bin"] = pd.cut(
        df["dc_mean"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=False,
        include_lowest=True,
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
    return df_train, df_val, df_test


# SECTION: Dataset Definition
# Custom PyTorch Dataset for loading dermatological images and their corresponding disagreement scores.
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


# SECTION: Training & Evaluation Functions


# Define a new head that produces a spatial map and then aggregates it for regression
# Predicts a spatial heatmap which is then aggregated (mean-pooled) to get a scalar regression output.
# Useful for interpretability and compatibility with Grad-CAM.
class MultiTaskHead(nn.Module):
    def __init__(self, in_channels):
        super(MultiTaskHead, self).__init__()
        # 1x1 convolution produces a 1-channel output preserving spatial dimensions
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, in_channels, H, W]
        spatial_map = self.conv(x)  # shape: [B, 1, H, W]
        # Apply sigmoid to force values between 0 and 1
        spatial_map = self.sigmoid(spatial_map)
        # Global average pooling to produce a scalar output per sample
        output = spatial_map.view(spatial_map.size(0), spatial_map.size(1), -1).mean(
            dim=2
        )
        return output, spatial_map


# Load the model based on the specified architecture.
# Modified for ResNet-18 to use the multi-task head.
def get_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "resnet18":
        # Load ResNet-18 without the last two layers (avgpool and fc)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc to keep spatial dimensions (take layers up to layer4)
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
        in_channels = 512  # For ResNet-18, output channels of layer4 is 512
        head = MultiTaskHead(in_channels)
        # Define the new model as a sequential module: backbone -> multi-task head
        model = nn.Sequential(backbone, head)
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


# In training, you'll use only the regression output.
# For example, modify the training function as follows:
# model(images) returns (regression_output, spatial_map); we only use regression_output here
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        # Our model returns a tuple: (regression_output, spatial_map)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
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
            # Unpack the tuple from the model's forward pass:
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
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


# SECTION: Training Function
# Train the model with early stopping and learning rate scheduling.
# Returns the best model based on validation loss.
def train_model(model, train_loader, val_loader, test_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.1
    )
    early_stopping = EarlyStopping(
        patience=3, verbose=True, path=os.path.join(SRC_DIR, "best_model.pth")
    )

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
    model.load_state_dict(torch.load(os.path.join(SRC_DIR, "best_model.pth")))
    test_loss, test_mse, test_mae, test_std_mse, test_std_mae = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"\nTest Loss: {test_loss:.4f}, MSE: {test_mse:.4f} (Std: {test_std_mse:.4f}), "
        f"MAE: {test_mae:.4f} (Std: {test_std_mae:.4f})"
    )
    return model


# --- Helper function to load image name lists from json files ---
def load_name_list(json_file):
    with open(json_file, "r") as f:
        names = json.load(f)
    return names


def remove_extension(name):
    return os.path.splitext(name)[0]


def create_dataloaders(
    df_train, df_val, df_test, img_dir, batch_size, train_transform, val_test_transform
):
    train_dataset = ImageRegressionDataset(df_train, img_dir, transform=train_transform)
    val_dataset = ImageRegressionDataset(df_val, img_dir, transform=val_test_transform)
    test_dataset = ImageRegressionDataset(
        df_test, img_dir, transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader, test_dataset


# Evaluates the model on benign and malignant subsets using image lists from config JSONs.
def evaluate_subsets(
    model, df_test, img_dir, batch_size, transform, criterion, device, config
):
    benign_list = load_name_list(config["benign_json"])
    malignant_list = load_name_list(config["malignant_json"])

    df_test["img_base"] = df_test["img_name"].apply(remove_extension)
    benign_df = df_test[df_test["img_base"].isin(benign_list)]
    malignant_df = df_test[df_test["img_base"].isin(malignant_list)]

    print(
        f"Found {len(benign_df)} benign images and {len(malignant_df)} malignant images in the test set."
    )

    benign_dataset = ImageRegressionDataset(benign_df, img_dir, transform=transform)
    malignant_dataset = ImageRegressionDataset(
        malignant_df, img_dir, transform=transform
    )

    benign_loader = DataLoader(
        benign_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    malignant_loader = DataLoader(
        malignant_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    benign_eval = evaluate(model, benign_loader, criterion, device)
    malignant_eval = evaluate(model, malignant_loader, criterion, device)

    print("\nBenign subset evaluation:")
    print(f"Loss: {benign_eval[0]:.4f}")
    print(f"MSE: {benign_eval[1]:.4f} (Std: {benign_eval[3]:.4f})")
    print(f"MAE: {benign_eval[2]:.4f} (Std: {benign_eval[4]:.4f})")

    print("\nMalignant subset evaluation:")
    print(f"Loss: {malignant_eval[0]:.4f}")
    print(f"MSE: {malignant_eval[1]:.4f} (Std: {malignant_eval[3]:.4f})")
    print(f"MAE: {malignant_eval[2]:.4f} (Std: {malignant_eval[4]:.4f})")
    print()
    return benign_eval, malignant_eval


# Run Grad-CAM visualization on a batch of images defined in the config file.
def run_gradcam_batch(model, model_name, image_filenames, archive_dir, metrics_file):
    if model_name == "vit_b_32":
        print(
            "Skipping Grad-CAM for ViT. Grad-CAM requires convolutional layers which ViT does not have."
        )
        return

    for image_filename in image_filenames:
        image_path = os.path.join(archive_dir, image_filename)
        print(f"Running GradCAM on: {image_path}")
        gradcam_utils.run_gradcam(model_name, model, image_path, metrics_file)


# Returns the image transformations to be applied to the training, validation, and test datasets
def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )
    val_test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, val_test_transform


# SECTION: Main Function
# Main entry point for training, evaluation, and visualizations based on config toggles.
def main():
    try:
        config = load_config()
    except FileNotFoundError:
        raise RuntimeError("Missing config.yaml file. Please check your setup.")

    REQUIRED_KEYS = [
        "model_name",
        "num_epochs",
        "batch_size",
        "metrics_file",
        "image_archive_dir",
        "enable_scatter_plot",
        "enable_error_visuals",
        "enable_subset_eval",
        "enable_gradcam",
        "gradcam_images",
        "benign_json",
        "malignant_json",
        "enable_correlation_analysis",
        "enable_mae_normalization",
        "enable_mse_normalization",
        "enable_dice_histograms",
    ]
    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing required key in config: '{key}'")

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    METRICS_FILE = os.path.join(SRC_DIR, config["metrics_file"])
    ARCHIVE_FOLDER = os.path.join(SRC_DIR, config["image_archive_dir"])
    BENIGN_JSON_PATH = os.path.join(SRC_DIR, config["benign_json"])
    MALIGNANT_JSON_PATH = os.path.join(SRC_DIR, config["malignant_json"])
    config["benign_json"] = BENIGN_JSON_PATH
    config["malignant_json"] = MALIGNANT_JSON_PATH
    img_dir = ARCHIVE_FOLDER
    batch_size = config["batch_size"]

    # Download data if not already present by calling utils/download_data.py
    download_data()

    if not os.path.exists(METRICS_FILE):
        raise FileNotFoundError(f"Metrics file not found at: {METRICS_FILE}")
    if not os.path.exists(ARCHIVE_FOLDER):
        raise FileNotFoundError(f"Image archive folder not found at: {ARCHIVE_FOLDER}")

    df_train, df_val, df_test = load_and_split_data(METRICS_FILE)

    model = get_model(config["model_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_test_transform = get_transforms()

    train_loader, val_loader, test_loader, test_dataset = create_dataloaders(
        df_train,
        df_val,
        df_test,
        img_dir,
        batch_size,
        train_transform,
        val_test_transform,
    )

    if config.get("enable_training", True):
        best_model = train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=config["num_epochs"],
        )
    else:
        if os.path.exists(os.path.join(SRC_DIR, "best_model.pth")):
            try:
                model.load_state_dict(
                    torch.load(os.path.join(SRC_DIR, "best_model.pth"))
                )
                best_model = model
                print("Loaded pretrained model from best_model.pth")
            except RuntimeError:
                print(
                    "The saved model does not match the specified architecture in the config file."
                )
                print(
                    "Please set enable_training: true in config.yaml to retrain the model."
                )
                exit(1)
        else:
            raise FileNotFoundError(
                "No trained model found. To train a model, go to config.yaml and set enable_training: true"
            )

    if config["enable_scatter_plot"]:
        plot_scatter(best_model, test_loader, device)

    if config["enable_error_visuals"]:
        display_extreme_error_images(best_model, test_dataset, device)

    if config["enable_subset_eval"]:
        criterion = nn.MSELoss()
        evaluate_subsets(
            best_model,
            df_test,
            img_dir,
            config["batch_size"],
            val_test_transform,
            criterion,
            device,
            config,
        )

    if config["enable_gradcam"]:
        run_gradcam_batch(
            best_model,
            config["model_name"],
            config["gradcam_images"],
            ARCHIVE_FOLDER,
            METRICS_FILE,
        )

    if config.get("enable_dice_histograms", False):
        dice_mean_histograms.main()

    if config.get("enable_correlation_analysis", False):
        correlation.main()

    if config.get("enable_mae_normalization", False):
        MAE_normalization.main()

    if config.get("enable_mse_normalization", False):
        MSE_normalization.main()


if __name__ == "__main__":
    main()
