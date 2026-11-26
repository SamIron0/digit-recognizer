/*
Work package 9 — Frontend canvas
Owner: teammate assigned to this package.

Deliverable: drawing with mouse and touch, plus a Clear button.
Canvas export used by package 10: PNG, black ink on white background, 280x280.
Depends on: package 1.

Replace this placeholder with the implementation.
*/
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let coord = { x: 0, y: 0 };//keeps track of the mouse position on the canvas

document.addEventListener("mousedown", start);//start drawing when mouse is pressed
document.addEventListener("mouseup", stop);//stop drawing when mouse is released

function reposition(event) { //updates the mouse coordinates when moved
  coord.x = event.clientX - canvas.offsetLeft;
  coord.y = event.clientY - canvas.offsetTop;
}
function start(event) { 
  document.addEventListener("mousemove", draw);
  reposition(event);
}
function stop() {
  document.removeEventListener("mousemove", draw);
}
function draw(event) { //This is what actually draws the line on the canvas
  ctx.beginPath();
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";
  ctx.moveTo(coord.x, coord.y);
  reposition(event);
  ctx.lineTo(coord.x, coord.y);
  ctx.stroke();
}

const clearButton = document.getElementById("clear");
clearButton.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);//clears the canvas when the Clear button is clicked
});

const predictButton = document.getElementById("predict");
predictButton.addEventListener("click", () => {
  const imageData = canvas.toDataURL("image/png");//converts the canvas content to a PNG image
  // Do something with the image data, e.g., send it to the server for prediction
});