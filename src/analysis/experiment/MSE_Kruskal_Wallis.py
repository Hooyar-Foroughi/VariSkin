from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd


def main():
    # Normalized MSE values for each model
    resnet_scores = [
        0.99796612,
        0.13624485,
        0.74595548,
        0.05540673,
        -0.77483817,
        -0.07071808,
        -0.07071808,
        1.64255485,
        -0.44354695,
        -2.21830676,
    ]

    efficientnet_scores = [
        -0.22740283,
        -0.52465344,
        0.58459158,
        -2.01708171,
        0.74109215,
        -0.92868353,
        1.64305341,
        0.41453062,
        -0.52465344,
        0.83920719,
    ]

    vit_scores = [
        0.04282742,
        1.33522286,
        0.60682795,
        -1.83111586,
        0.60682795,
        -1.83111586,
        0.60682795,
        0.60682795,
        0.04282742,
        -0.18595778,
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
