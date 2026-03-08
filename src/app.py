from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
from model_helper import HotdogPredictor

app = FastAPI()

# create predictor class once
predictor = HotdogPredictor()

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hotdog or Not?</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: 'DM Sans', sans-serif;
                background: #0E0E12;
                margin: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 40px 20px;
                position: relative;
                overflow: hidden;
            }

            body::before {
                content: '';
                position: fixed;
                width: 700px; height: 700px;
                background: radial-gradient(circle, rgba(201,168,76,0.07) 0%, transparent 65%);
                top: -200px; left: -200px;
                pointer-events: none;
            }

            body::after {
                content: '';
                position: fixed;
                width: 500px; height: 500px;
                background: radial-gradient(circle, rgba(201,168,76,0.04) 0%, transparent 65%);
                bottom: -100px; right: -100px;
                pointer-events: none;
            }

            .container {
                background: #16161D;
                border-radius: 24px;
                box-shadow:
                    0 40px 100px rgba(0,0,0,0.7),
                    0 0 0 1px rgba(201,168,76,0.15),
                    inset 0 1px 0 rgba(255,255,255,0.04);
                max-width: 720px;
                width: 100%;
                display: flex;
                flex-direction: row;
                align-items: stretch;
                overflow: hidden;
                animation: fadeUp 0.8s cubic-bezier(0.16,1,0.3,1) both;
            }

            @keyframes fadeUp {
                from { opacity: 0; transform: translateY(28px); }
                to   { opacity: 1; transform: translateY(0); }
            }

            /* LEFT PANEL */
            .panel-left {
                width: 240px;
                min-width: 240px;
                padding: 48px 36px;
                background: linear-gradient(160deg, #1A1410 0%, #0E0A06 100%);
                border-right: 1px solid rgba(201,168,76,0.15);
                display: flex;
                flex-direction: column;
                justify-content: center;
            }

            .panel-left h1 {
                font-family: 'Playfair Display', serif;
                font-size: 1.75em;
                font-weight: 700;
                color: #F0EDE6;
                line-height: 1.25;
                text-align: left;
            }

            .panel-left p {
                font-size: 0.82em;
                font-weight: 300;
                color: #5A5A6C;
                margin-top: 12px;
                line-height: 1.6;
                text-align: left;
            }

            .divider {
                width: 36px;
                height: 2px;
                background: linear-gradient(to right, #C9A84C, transparent);
                margin: 20px 0;
            }

            /* RIGHT PANEL */
            .panel-right {
                flex: 1;
                padding: 44px 40px;
                display: flex;
                flex-direction: column;
                gap: 14px;
                justify-content: center;
            }

            input[type="file"] {
                width: 100%;
                padding: 13px 16px;
                border-radius: 10px;
                border: 1px solid rgba(201,168,76,0.2);
                background: #1E1E28;
                color: #7A7A8C;
                font-size: 0.88em;
                font-family: 'DM Sans', sans-serif;
                outline: none;
                cursor: pointer;
                transition: border-color 0.3s, box-shadow 0.3s;
            }

            input[type="file"]:hover,
            input[type="file"]:focus {
                border-color: rgba(201,168,76,0.5);
                box-shadow: 0 0 0 3px rgba(201,168,76,0.07);
            }

            #preview {
                width: 100%;
                border-radius: 12px;
                display: none;
                border: 1px solid rgba(201,168,76,0.15);
                box-shadow: 0 12px 40px rgba(0,0,0,0.5);
                object-fit: cover;
                max-height: 220px;
            }

            #preview.visible {
                display: block;
                animation: fadeUp 0.4s ease both;
            }

            button {
                background: linear-gradient(135deg, #C9A84C 0%, #A8863A 100%);
                color: #0E0A06;
                border: none;
                padding: 13px 28px;
                font-size: 0.85em;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                border-radius: 10px;
                cursor: pointer;
                font-family: 'DM Sans', sans-serif;
                box-shadow: 0 8px 24px rgba(201,168,76,0.28);
                transition: transform 0.2s, box-shadow 0.2s;
                width: 100%;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 14px 32px rgba(201,168,76,0.38);
            }

            button:active {
                transform: translateY(0);
                opacity: 0.9;
            }

            #result {
                font-weight: 500;
                font-size: 1em;
                color: #E8C97A;
                letter-spacing: 0.5px;
                min-height: 1.4em;
                text-align: left;
            }

            @media (max-width: 580px) {
                .container { flex-direction: column; }
                .panel-left {
                    width: 100%;
                    min-width: unset;
                    border-right: none;
                    border-bottom: 1px solid rgba(201,168,76,0.15);
                    padding: 36px 32px;
                }
                .panel-right { padding: 32px; }
            }
        </style>
    </head>
    <body>
        <div class="container">

            <!-- LEFT PANEL -->
            <div class="panel-left">
                <h1>🌭 Hotdog or Not Hotdog?</h1>
                <div class="divider"></div>
                <p>Upload an image and let the model decide.</p>
            </div>

            <!-- RIGHT PANEL -->
            <div class="panel-right">
                <input type="file" id="imageInput" accept="image/*">
                <img id="preview" src="">
                <button onclick="uploadImage()">Predict!</button>
                <div id="result"></div>
            </div>

        </div>

        <script>
            const imageInput = document.getElementById('imageInput');
            const preview = document.getElementById('preview');

            imageInput.onchange = evt => {
                const [file] = imageInput.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.classList.add('visible');
                }
            }

            async function uploadImage() {
                const file = imageInput.files[0];
                if (!file) { alert("Please select a photo first!"); return; }

                const formData = new FormData();
                formData.append("file", file);

                document.getElementById('result').innerText = "Analyzing...";

                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();
                const resultElement = document.getElementById('result');
                resultElement.innerText = `${data.label} \nScore: ${data.score.toFixed(4)} \nConfidence: ${data.confidence}`;
                if (data.label === "NOT HOT DOG") {
                    resultElement.style.color = "red";   // NOT HOT DOG → kırmızı
                } else if (data.label === "HOT DOG") {
                    resultElement.style.color = "green"; // HOT DOG → yeşil
                } else {
                    resultElement.style.color = "black"; // fallback
                }


            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    result = predictor.predict_image(contents)
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
