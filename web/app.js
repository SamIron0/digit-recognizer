/*
Work package 10 — Frontend integration
Owner: teammate assigned to this package.

Deliverable: export the canvas as a PNG, call POST /predict, and show the
digit, confidence, a loading state, and a readable error.
Depends on: packages 8 and 9. A mocked API response is fine until the API exists.

Replace this placeholder with the implementation.
*/

const API_URL = "http://localhost:8000/predict";

//Getting elements from the HTML page
const drawingCanvas = document.getElementById("drawingCanvas");
const predictBtn = document.getElementById("predict");
const digit = document.getElementById("digit");
const confidence = document.getElementById("confidence");
const loadingMessage = document.getElementById("loadingMessage");
const errorMessage = document.getElementById("errorMessage");

//Adding an event listener for the predict button
predictBtn.addEventListener("click", async function() {
    loadingMessage.hidden = false;
    predictBtn.disabled = true;
    errorMessage.textContent = "";
    digit.textContent = "-";
    confidence.textContent = "-";

    try {
        const imageData = drawingCanvas.toDataURL("image/png");
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageData
        })
    });
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
    }

    digit.textContent = data.digit;
    confidence.textContent = (data.confidence * 100).toFixed(2) + "%"; 

    } catch(error) {
        errorMessage.textContent = "ERROR! Could not make prediction: " + error.message;
    }
    
    finally {
        loadingMessage.hidden = true;
        predictBtn.disabled = false;
    }
});