from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from src.data import load_data


def _confused_pairs(cm, limit=4):
    """Return the most common true/predicted pairs, highest count first."""
    pairs = []
    for true_label in range(cm.shape[0]):
        for pred_label in range(cm.shape[1]):
            count = int(cm[true_label, pred_label])
            if true_label != pred_label and count > 0:
                pairs.append((count, int(true_label), int(pred_label)))
    pairs.sort(reverse=True)
    return pairs[:limit]


def _error_analysis(cm):
    """Describe the digit pairs the model actually mixes up on this run."""
    pairs = _confused_pairs(cm)
    total_errors = int(cm.sum() - np.trace(cm))
    if not pairs:
        return (
            "Evaluated the trained CNN on the MNIST test split from `load_data()`. "
            "The model made no mistakes on this split.\n"
        )

    named = ", ".join(
        f"{true_label} read as {pred_label} ({count} images)"
        for count, true_label, pred_label in pairs
    )
    return (
        "Evaluated the trained CNN on the 10,000-image MNIST test split from "
        "`load_data()`, the same split training uses. "
        f"The model misclassified {total_errors} test images. "
        f"The most common confusions are {named}. "
        "These digits share similar strokes, so a small change in a loop or a curve "
        "is enough to flip the prediction.\n"
    )


def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the trained model and the MNIST test split
    model = load_model("models/digit_cnn.keras")
    _, _, (x_test, y_test) = load_data()

    # 2. Predict digit probabilities and convert to predicted class (0-9)
    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # 3. Save confusion matrix image
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(docs_dir / "confusion_matrix.png")
    plt.close(fig)

    # 4. Save 4x4 grid of up to 16 wrong predictions
    wrong_idx = np.where(y_pred != y_test)[0]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(wrong_idx):
            idx = wrong_idx[i]
            img = x_test[idx].squeeze()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"true {y_test[idx]} / pred {y_pred[idx]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(docs_dir / "misclassified.png")
    plt.close(fig)

    # 5. Save test accuracy and analysis to docs/evaluation.md
    accuracy = float(np.mean(y_pred == y_test))
    report = (
        "# Evaluation Report\n\n"
        f"- **Test Accuracy:** {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "## Error Analysis\n"
        f"{_error_analysis(cm)}"
    )
    (docs_dir / "evaluation.md").write_text(report, encoding="utf-8")
    print(f"Done. Accuracy: {accuracy:.4f}. Files saved to {docs_dir}/")


if __name__ == "__main__":
    main()