

import os
import matplotlib.pyplot as plt
import torch

# Scatter plot comparing ground truth vs predicted dc_mean to visualize performance spread.
def plot_scatter(model, dataloader, device):
    model.eval()
    gt_values = []
    pred_values = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            gt_values.extend(labels.cpu().numpy().flatten().tolist())
            pred_values.extend(outputs.cpu().numpy().flatten().tolist())

    plt.figure(figsize=(8, 6))
    plt.scatter(gt_values, pred_values, color="blue", alpha=0.6)
    plt.xlabel("Ground Truth dc_mean")
    plt.ylabel("Predicted dc_mean")
    plt.title("Scatter Plot: Ground Truth vs Predicted dc_mean")
    min_val = min(min(gt_values), min(pred_values))
    max_val = max(max(gt_values), max(pred_values))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y=x)")
    plt.legend()
    plt.show()
    # Save the scatter plot for headless environments in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "scatter_plot.png")
    plt.savefig(save_path)
