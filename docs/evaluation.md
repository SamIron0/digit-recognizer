# Evaluation Report

- **Test Accuracy:** 0.0900 (9.00%)

## Error Analysis
Evaluated on placeholder data with the initial model, predictions are close to random noise across digits 0 through 9. In handwritten digit recognition on real data, visually similar pairs such as 4/9 and 3/5 commonly cause confusion due to loop closure and stroke curves. Once the trained model and real test set are merged, re-running `python -m src.evaluate` will refresh these metrics and charts with actual performance.
