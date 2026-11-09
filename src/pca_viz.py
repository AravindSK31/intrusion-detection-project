"""
2D PCA visualization: projects the (already preprocessed) feature space
down to 2 components and plots normal vs. each attack category, to see
whether attacks are visually separable from normal traffic.

PCA is fit on the FULL model-training data (normal + attacks) for this
visualization specifically, since the goal here is seeing class separation,
not anomaly scoring -- a different purpose from the ablation in Part 2.
"""

import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from load_nslkdd import load_split
from preprocess import fit_feature_pipeline, transform_features
from evaluation import make_normal_validation_split

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15


if __name__ == "__main__":
    train_path = sys.argv[1]
    train_df = load_split(train_path)

    model_train_df, _ = make_normal_validation_split(train_df, VALIDATION_FRACTION, RANDOM_STATE)

    pipeline = fit_feature_pipeline(model_train_df)
    X = transform_features(model_train_df, pipeline)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    print(f"Variance explained by 2 components: {pca.explained_variance_ratio_.sum():.2%}")

    plt.figure(figsize=(8, 7))
    categories = model_train_df["attack_category"].unique()
    for cat in categories:
        mask = model_train_df["attack_category"].values == cat
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cat, alpha=0.4, s=8)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("NSL-KDD Training Data: 2D PCA Projection by Attack Category")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig("pca_2d_visualization.png", dpi=150)
    print("Saved to pca_2d_visualization.png")