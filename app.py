from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
# Load your trained model
# model = tf.keras.models.load_model("digit_model.h5")
# model = tf.keras.models.load_model("15epochs.h5")
# model = tf.keras.models.load_model("8epochs.h5")
# model = tf.keras.models.load_model("10epochs.h5")
model = tf.keras.models.load_model("11epochs.h5")


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

@app.route("/", methods=["GET", "POST"])
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


# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from flask_cors import CORS
# import cv2

# app = Flask(__name__)
# CORS(app)

# # Load trained model
# model = tf.keras.models.load_model("digit_model.h5")


# def preprocess_image(image):
#     # Convert PIL → OpenCV
#     image = np.array(image)

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Blur to remove noise
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Threshold (digit white, background black)
#     _, thresh = cv2.threshold(
#         blur, 0, 255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     # Find digit pixels
#     coords = np.column_stack(np.where(thresh > 0))

#     if coords.size == 0:
#         return None

#     # Bounding box
#     x, y, w, h = cv2.boundingRect(coords)
#     digit = thresh[y:y+h, x:x+w]

#     # 🔥 FIXED PART (safe resizing + centering)
#     h, w = digit.shape

#     if h > w:
#         new_h = 20
#         new_w = int(w * (20 / h))
#     else:
#         new_w = 20
#         new_h = int(h * (20 / w))

#     resized_digit = cv2.resize(digit, (new_w, new_h))

#     # Create 28x28 black image
#     final_img = np.zeros((28, 28), dtype=np.uint8)

#     # Center digit
#     x_offset = (28 - new_w) // 2
#     y_offset = (28 - new_h) // 2

#     final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_digit

#     # Normalize
#     normalized = final_img / 255.0

#     # Reshape
#     final = normalized.reshape(1, 28, 28, 1)

#     return final

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None

#     if request.method == "POST":
#         file = request.files["image"]
#         image = Image.open(file)

#         processed_image = preprocess_image(image)

#         if processed_image is None:
#             prediction = "No digit detected"
#         else:
#             preds = model.predict(processed_image)
#             prediction = int(np.argmax(preds))

#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)