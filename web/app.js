/*
Work package 10 — Frontend integration
Owner: teammate assigned to this package.

Deliverable: export the canvas as a PNG, call POST /predict, and show the
digit, confidence, a loading state, and a readable error.
Depends on: packages 8 and 9. A mocked API response is fine until the API exists.
*/

const USE_MOCK = true;
const API_URL = "http://localhost:8000";
// While USE_MOCK is true, set this to "result", "unsure", or "error"
// to demo each Predict state without the API. Loading shows during the delay.
const MOCK_STATE = "result";

const predictBtn = document.getElementById("predict");
const loadingMessage = document.getElementById("loadingMessage");
const resultMessage = document.getElementById("resultMessage");
const errorMessage = document.getElementById("errorMessage");

function delay(ms) {
    return new Promise(function (resolve) {
        setTimeout(resolve, ms);
    });
}

function mockResponse(digit, confidence) {
    const probabilities = [];
    const rest = (1 - confidence) / 9;
    for (let i = 0; i < 10; i++) {
        probabilities.push(i === digit ? confidence : rest);
    }
    return { digit: digit, confidence: confidence, probabilities: probabilities };
}

function canvasImage() {
    if (typeof getCanvasPNG === "function") {
        return getCanvasPNG();
    }
    return document.getElementById("drawingCanvas").toDataURL("image/png");
}

function showResult(data) {
    const confidence = Number(data.confidence);
    if (!Number.isFinite(confidence)) {
        throw new Error("The server response did not include a confidence score.");
    }
    if (confidence < 0.5) {
        resultMessage.textContent = "I'm not sure — try redrawing?";
        return;
    }
    const percent = Math.round(confidence * 100);
    resultMessage.textContent = "Prediction: " + data.digit + " (confidence " + percent + "%)";
}

async function requestPrediction(image) {
    if (USE_MOCK) {
        await delay(500);
        if (MOCK_STATE === "error") {
            throw new Error("Could not reach the prediction service.");
        }
        if (MOCK_STATE === "unsure") {
            return mockResponse(4, 0.42);
        }
        return mockResponse(7, 0.97);
    }

    let response;
    try {
        response = await fetch(API_URL + "/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: image })
        });
    } catch (error) {
        throw new Error("Could not reach the prediction service. Check that the API is running.");
    }

    let data;
    try {
        data = await response.json();
    } catch (error) {
        throw new Error("The server returned an unreadable response.");
    }
    if (!response.ok || data.error) {
        throw new Error(data.error || "Prediction failed.");
    }
    return data;
}

predictBtn.addEventListener("click", async function () {
    loadingMessage.hidden = false;
    predictBtn.disabled = true;
    resultMessage.textContent = "";
    errorMessage.textContent = "";

    try {
        const data = await requestPrediction(canvasImage());
        showResult(data);
    } catch (error) {
        errorMessage.textContent = error.message || "Something went wrong. Please try again.";
    } finally {
        loadingMessage.hidden = true;
        predictBtn.disabled = false;
    }
});
