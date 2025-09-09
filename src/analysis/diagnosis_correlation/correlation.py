import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from utils.download_data import get_metadata


# This script analyzes the relationship between image annotation variability
# and diagnosis labels (benign vs. malignant) using the ISIC dataset of medical images
# and their associated metadata.
def main():
    # Define paths
    SRC_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )  # project root
    METRICS_FILE = os.path.join(SRC_DIR, "metrics.csv")
    METADATA_DIR = os.path.join(SRC_DIR, "ISIC_Archive", "metadata")

    get_metadata()

    if not os.path.exists(METRICS_FILE):
        print(f"Error: {METRICS_FILE} not found.")
        return

    # Step 1: Load metrics.csv
    print("Loading metrics.csv...")
    metrics_df = pd.read_csv(METRICS_FILE)

    if not os.path.exists(METADATA_DIR):
        print(f"Error: Metadata directory '{METADATA_DIR}' not found.")
        return

    # Step 2: Read metadata JSON files and extract diagnosis
    data = []
    for filename in os.listdir(METADATA_DIR):
        if filename.endswith(".json"):
            isic_id = filename.replace(".json", "")  # e.g., ISIC_0000004
            json_path = os.path.join(METADATA_DIR, filename)

            # Read the JSON file
            with open(json_path, "r") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode JSON from {filename}")
                    continue

            # Extract benign_malignant from clinical section
            if "clinical" in metadata and "benign_malignant" in metadata["clinical"]:
                diagnosis = metadata["clinical"]["benign_malignant"]
                if diagnosis in ["benign", "malignant"]:  # Ensure valid values
                    data.append({"img_name": f"{isic_id}.jpg", "diagnosis": diagnosis})
                else:
                    print(f"Skipping {isic_id}: Invalid diagnosis value '{diagnosis}'")
            else:
                print(f"Skipping {isic_id}: Missing clinical or benign_malignant field")

    # Create a DataFrame from metadata
    metadata_df = pd.DataFrame(data)
    metadata_df.to_csv(
        os.path.join(os.path.dirname(__file__), "metadata_output.csv"), index=False
    )

    # Step 3: Merge metrics and metadata on img_name
    print("Merging data...")
    merged_df = pd.merge(
        metrics_df[["img_name", "dc_mean"]],
        metadata_df,
        on="img_name",
        how="inner",  # Only keep rows where both datasets have data
    )

    if merged_df.empty:
        print("Error: No overlapping entries found in metrics and metadata.")
        return

    # Step 4: Calculate correlation
    # Convert diagnosis to binary (0 for benign, 1 for malignant)
    merged_df["diagnosis_binary"] = merged_df["diagnosis"].map(
        {"benign": 0, "malignant": 1}
    )
    correlation, p_value = pointbiserialr(
        merged_df["diagnosis_binary"], merged_df["dc_mean"]
    )
    print()
    print(f"Point-Biserial Correlation: {correlation:.3f}, p-value: {p_value:.4f}")
    print()

    # Step 5: Plot the correlation using a box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="diagnosis",
        y="dc_mean",
        hue="diagnosis",
        data=merged_df,
        palette="Set2",
        legend=False,
    )
    plt.title(
        f"Distribution of dc_mean by Diagnosis\nCorrelation: {correlation:.3f}, p-value: {p_value:.4f}"
    )
    plt.xlabel("Diagnosis")
    plt.ylabel("dc_mean")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "correlation_plot.png")
    )  # Save for OverLeaf
    plt.show()


if __name__ == "__main__":
    main()
