import numpy as np
from scipy.stats import boxcox, zscore
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    # MSE values
    resnet_scores = [
        0.0413,
        0.0388,
        0.0405,
        0.0386,
        0.0369,
        0.0383,
        0.0383,
        0.0436,
        0.0375,
        0.0356,
    ]
    efficientnet_scores = [
        0.0365,
        0.0362,
        0.0377,
        0.0356,
        0.0380,
        0.0359,
        0.0402,
        0.0374,
        0.0362,
        0.0382,
    ]
    vit_scores = [
        0.0415,
        0.0431,
        0.0421,
        0.0406,
        0.0421,
        0.0406,
        0.0421,
        0.0421,
        0.0415,
        0.0413,
    ]

    # Combine data into a NumPy array
    data = {
        "ResNet18": np.array(resnet_scores),
        "EfficientNet-B0": np.array(efficientnet_scores),
        "ViT-B/32": np.array(vit_scores),
    }

    # Apply Box-Cox transformation
    transformed_data = {}
    for key, values in data.items():
        values_positive = values - min(values) + 1e-6  # Ensure positivity
        transformed_data[key], _ = boxcox(values_positive)

    # Standardize the data (Z-score normalization)
    standardized_data = {
        key: zscore(values) for key, values in transformed_data.items()
    }

    # Create a single figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot original distributions
    for key, values in data.items():
        sns.kdeplot(values, label=f"{key}", fill=True, ax=axes[0])
    axes[0].set_title("MSE Original Distributions")
    axes[0].legend()
    axes[0].set_xlabel("MSE")

    # Plot transformed and standardized distributions
    for key, values in standardized_data.items():
        sns.kdeplot(values, label=f"{key}", fill=True, ax=axes[1])
    axes[1].set_title("MSE Transformed and Standardized Distributions")
    axes[1].legend()
    axes[1].set_xlabel("Transformed and Standardized MSE")

    # Save and show the figure
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "mse_distributions.png")
    plt.savefig(output_path)
    plt.close()

    # Display the standardized data
    print("\n\n[MSE] Standardized Scores:\n")
    for key, values in standardized_data.items():
        print(f"{key}: {values}")
    print("\nplot saved here: ../src/analysis/experiment/mse_distributions.png\n")

    if __name__ != "__main__":
        from . import MSE_Kruskal_Wallis

        MSE_Kruskal_Wallis.main()


if __name__ == "__main__":
    main()
