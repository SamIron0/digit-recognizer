# Evaluation Report

- **Test Accuracy:** 0.9920 (99.20%)

## Error Analysis
Evaluated the trained CNN on the 10,000-image MNIST test split from `load_data()`, the same split training uses. The model misclassified 80 test images. The most common confusions are 2 read as 7 (7 images), 6 read as 0 (6 images), 9 read as 5 (5 images), 9 read as 4 (5 images). These digits share similar strokes, so a small change in a loop or a curve is enough to flip the prediction.
