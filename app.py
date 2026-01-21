from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

IMG_SIZE = 64

models = {
    "svm": joblib.load("svm_model.pkl"),
    "rf": joblib.load("rf_model.pkl"),
    "lr": joblib.load("lr_model.pkl"),
    "kmeans": joblib.load("kmeans_model.pkl")
}

def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.flatten()
    return np.array(image).reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        model_name = request.form["model"]

        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        image = cv2.imread(img_path)
        processed = preprocess_image(image)

        model = models[model_name]
        prediction = model.predict(processed)[0]

        if prediction == 0:
            result = " Cat"
        else:
            result = " Dog"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
