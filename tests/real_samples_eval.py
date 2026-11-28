import base64
from collections import defaultdict
from pathlib import Path
import numpy as np

from api.main import decode_image, load_model, load_preprocess

SAMPLES_DIR = Path("samples")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_label_from_filename(filename: str) -> int:
    """Extracts the leading digit label (0-9) from '<label>_<person>_<n>.ext'."""
    first_char = filename.strip()[0]
    if not first_char.isdigit():
        raise ValueError(f"Filename does not start with a digit 0-9: {filename}")
    return int(first_char)


def evaluate():
    if not SAMPLES_DIR.exists():
        print(f"Error: '{SAMPLES_DIR}' directory not found.")
        return

    sample_files = [
        f for f in sorted(SAMPLES_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not sample_files:
        print(f"No sample images found in '{SAMPLES_DIR}'.")
        return

    model, model_loaded = load_model()
    preprocess_fn, preprocess_kind = load_preprocess()

    print(f"Model status:      {'real' if model_loaded else 'placeholder/dummy'}")
    print(f"Preprocess status: {preprocess_kind}")
    print(f"Samples found:     {len(sample_files)}\n")

    correct_per_digit = defaultdict(int)
    total_per_digit = defaultdict(int)
    total_correct = 0

    for img_path in sample_files:
        true_label = parse_label_from_filename(img_path.name)

        # Decode image to grayscale (H, W) uint8 array using the API's exact helper
        b64_str = base64.b64encode(img_path.read_bytes()).decode("ascii")
        gray = decode_image(b64_str)

        # Preprocess and predict
        x = preprocess_fn(gray)
        x = np.asarray(x, dtype=np.float32).reshape(1, 28, 28, 1)
        probs = np.asarray(model.predict(x, verbose=0)).reshape(-1)
        pred_label = int(np.argmax(probs))

        total_per_digit[true_label] += 1
        if pred_label == true_label:
            correct_per_digit[true_label] += 1
            total_correct += 1

    print(f"{'Digit':<8} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 42)

    for digit in range(10):
        tot = total_per_digit[digit]
        corr = correct_per_digit[digit]
        acc = (corr / tot * 100) if tot > 0 else 0.0
        print(f"{digit:<8} | {corr:<8} | {tot:<8} | {acc:>6.2f}%")

    overall_acc = (total_correct / len(sample_files)) * 100
    print("-" * 42)
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{len(sample_files)})")

    if overall_acc >= 80.0:
        print("\nPASS: Accuracy meets the >= 80% target.")
    else:
        print("\nBELOW TARGET: Expected ~10% on placeholder model; >= 80% required for real model.")


if __name__ == "__main__":
    evaluate()
