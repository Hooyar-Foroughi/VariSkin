from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd


def main():
    # Normalized MAE values for each model
    resnet_scores = [
        0.79219979,
        -2.36171116,
        0.81949341,
        0.31592327,
        -0.29807954,
        -0.00260027,
        0.23877101,
        1.50557454,
        -0.27992822,
        -0.72964284,
    ]

    efficientnet_scores = [
        -1.62723686,
        0.19442358,
        -0.24097125,
        -0.14332289,
        1.51470637,
        -1.62723686,
        0.89865766,
        0.07970564,
        -0.24097125,
        1.19224585,
    ]

    vit_scores = [
        0.40749936,
        -0.37685499,
        0.98936847,
        0.36006422,
        1.07534831,
        0.01518769,
        -1.80251656,
        0.94485893,
        0.18956112,
        -1.80251656,
    ]
    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(resnet_scores, efficientnet_scores, vit_scores)

    print(f"Kruskal-Wallis test statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < 0.05:
        print("There is a statistically significant difference between the models.")
    else:
        print("No statistically significant difference between the models.")

    # perform Dunn's test if there is a significant difference (to see pairwise comparisons)
    if p_value < 0.05:
        # Combine scores into a DataFrame for Dunn's test
        data = pd.DataFrame(
            {
                "DiceScore": resnet_scores + efficientnet_scores + vit_scores,
                "Model": (["ResNet18"] * len(resnet_scores))
                + (["EfficientNet-B0"] * len(efficientnet_scores))
                + (["ViT-B/32"] * len(vit_scores)),
            }
        )

        # Perform Dunn's test with Bonferroni correction
        dunn_results = sp.posthoc_dunn(
            data, val_col="DiceScore", group_col="Model", p_adjust="bonferroni"
        )

        print()
        print("\nDunn's test results (p-values):")
        print(dunn_results)
        print()


if __name__ == "__main__":
    main()
