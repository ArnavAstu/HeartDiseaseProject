from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model/ecg_model.h5"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "ecg_image" not in request.files:
        return "No image uploaded"

    file = request.files["ecg_image"]

    if file.filename == "":
        return "No image selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    image = Image.open(filepath).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Model prediction
    probability = float(model.predict(image_array, verbose=0)[0][0])

    # IMPORTANT:
    # 0 = ABNORMAL
    # 1 = NORMAL
    if probability >= 0.5:
        result = "NORMAL"
        confidence = probability * 100
    else:
        result = "ABNORMAL"
        confidence = (1 - probability) * 100

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        image_path="/" + filepath.replace("\\", "/")
    )


if __name__ == "__main__":
    app.run(debug=True)