import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():  # Load dataset
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    METRICS_FILE = os.path.join(SRC_DIR, "../../metrics.csv")
    df = pd.read_csv(METRICS_FILE)

    # Split into train and test using the same stratified bins
    df["d_bin"] = pd.cut(
        df["dc_mean"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=False,
        include_lowest=True,
    )

    # Train-test split
    from sklearn.model_selection import StratifiedShuffleSplit

    test_size = 0.15
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in split.split(df, df["d_bin"]):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

    # Plot histograms
    plt.figure(figsize=(12, 5))

    # Training data histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df_train["dc_mean"], bins=20, kde=True, color="blue")
    plt.xlabel("Dice Score")
    plt.ylabel("Frequency")
    plt.title("Training Set Dice Score Distribution")

    # Testing data histogram
    plt.subplot(1, 2, 2)
    sns.histplot(df_test["dc_mean"], bins=20, kde=True, color="red")
    plt.xlabel("Dice Score")
    plt.ylabel("Frequency")
    plt.title("Testing Set Dice Score Distribution")

    # Show plots
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "dice_mean_histograms.png")
    )  # Saves the plot as a PNG file
    plt.close()  # Closes the figure to free up memory


if __name__ == "__main__":
    main()
