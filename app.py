from flask import Flask, render_template, request, send_from_directory
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


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


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
    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # Model output:
    # 0 = ABNORMAL
    # 1 = NORMAL

    normal_probability = probability * 100
    abnormal_probability = (1 - probability) * 100

    if probability >= 0.5:
        result = "NORMAL"
        confidence = normal_probability
    else:
        result = "ABNORMAL"
        confidence = abnormal_probability

    if result == "NORMAL":
        interpretation = (
            "The AI model classified the uploaded ECG image "
            "as NORMAL based on the learned ECG image patterns."
        )
    else:
        interpretation = (
            "The AI model classified the uploaded ECG image "
            "as ABNORMAL based on the learned ECG image patterns."
        )

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        normal_probability=round(normal_probability, 2),
        abnormal_probability=round(abnormal_probability, 2),
        interpretation=interpretation,
        image_path="/uploads/" + file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)