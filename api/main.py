"""
Work package 8 — Backend API
Owner: teammate assigned to this package.

Deliverable: FastAPI app with POST /predict and GET /health.
Load models/digit_cnn.keras once at startup. A dummy model is fine until package 5 lands.

Request:  { "image": "<base64 PNG string>" }
Response: { "digit": 7, "confidence": 0.97, "probabilities": [10 values] }
Error:    { "error": "message" } with an appropriate HTTP status.

Depends on: packages 5 and 7.
Tests live in tests/test_api.py.

Replace this placeholder with the implementation.
"""
