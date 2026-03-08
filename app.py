from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
# Load your trained model
model = tf.keras.models.load_model("digit_model.h5")

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Optional: invert colors if needed (MNIST = white digit on black)
    image = ImageOps.invert(image)

    # Convert to array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    return image

@app.route("/", methods=["POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        processed_image = preprocess_image(image)

        preds = model.predict(processed_image)
        prediction = np.argmax(preds)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
