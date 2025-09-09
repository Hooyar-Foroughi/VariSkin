import numpy as np
from scipy.stats import boxcox, zscore
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    # MAE values
    resnet_scores = [
        0.1505,
        0.1367,
        0.1507,
        0.1472,
        0.1435,
        0.1452,
        0.1467,
        0.1561,
        0.1436,
        0.1413,
    ]
    efficientnet_scores = [
        0.1451,
        0.1461,
        0.1455,
        0.1456,
        0.1523,
        0.1451,
        0.1483,
        0.1459,
        0.1455,
        0.1499,
    ]
    vit_scores = [
        0.1428,
        0.1390,
        0.1473,
        0.1425,
        0.1481,
        0.1406,
        0.1370,
        0.1469,
        0.1415,
        0.1370,
    ]

    # Combine data into a NumPy array
    data = {
        "ResNet18": np.array(resnet_scores),
        "EfficientNet-B0": np.array(efficientnet_scores),
        "ViT-B/32": np.array(vit_scores),
    }

    # Apply Box-Cox transformation (add a small constant if needed)
    transformed_data = {}
    for key, values in data.items():
        values_positive = values - min(values) + 1e-6  # Ensure positivity
        transformed_data[key], _ = boxcox(values_positive)

    # Standardize the transformed data (Z-score normalization)
    standardized_data = {
        key: zscore(values) for key, values in transformed_data.items()
    }

    # Create subplots for original and transformed distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original distributions
    for key, values in data.items():
        sns.kdeplot(values, label=f"{key}", ax=axes[0], fill=True)
    axes[0].set_title("MAE Original Distributions")
    axes[0].legend()
    axes[0].set_xlabel("MAE")

    # Plot transformed and standardized distributions
    for key, values in standardized_data.items():
        sns.kdeplot(values, label=f"{key}", ax=axes[1], fill=True)
    axes[1].set_title("MAE Transformed and Standardized Distributions")
    axes[1].legend()
    axes[1].set_xlabel("Transformed and Standardized MAE")

    # Save both histograms in one PNG file
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "mae_distributions.png"))
    plt.close()

    # Print standardized data
    print("\n\n[MAE] Standardized Scores:\n")
    for key, values in standardized_data.items():
        print(f"{key}: {values}")
    print("\nplot saved here: ../src/analysis/experiment/mae_distributions.png\n")

    if __name__ != "__main__":
        from . import MAE_Kruskal_Wallis

        MAE_Kruskal_Wallis.main()


if __name__ == "__main__":
    main()
