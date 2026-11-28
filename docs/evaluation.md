# Evaluation Report

- **Test Accuracy:** 0.1400 (14.00%)

## Error Analysis
Evaluated on the current placeholder dataset (`fake_data()`), the model predicts digit `8` for the vast majority of samples across all true classes, with a few `7`s predicted as `2` and occasional `1`s/`2`s predicted as `2` or `5`s as `3`. Once the real test dataset replaces `fake_data()`, re-running `python -m src.evaluate` will refresh these metrics and charts with real handwritten digit performance.