from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from src.data import fake_data


def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load placeholder model and dummy test data
    model = load_model("models/digit_cnn.keras")
    _, _, (x_test, y_test) = fake_data()

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
        "Evaluated on placeholder data with the initial model, predictions are close to random noise "
        "across digits 0 through 9. In handwritten digit recognition on real data, visually similar "
        "pairs such as 4/9 and 3/5 commonly cause confusion due to loop closure and stroke curves. "
        "Once the trained model and real test set are merged, re-running `python -m src.evaluate` will "
        "refresh these metrics and charts with actual performance.\n"
    )
    (docs_dir / "evaluation.md").write_text(report, encoding="utf-8")
    print(f"Done. Accuracy: {accuracy:.4f}. Files saved to {docs_dir}/")


if __name__ == "__main__":
    main()