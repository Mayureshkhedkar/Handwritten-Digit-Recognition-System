# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# from PIL import Image, ImageOps
# from flask_cors import CORS



# app = Flask(__name__)
# CORS(app)
# model = tf.keras.models.load_model("11epochs.h5")


# def preprocess_image(image):
#     # Convert to grayscale
#     image = image.convert("L")

#     # Resize to 28x28
#     image = image.resize((28, 28))

#     # Optional: invert colors if needed (MNIST = white digit on black)
#     image = ImageOps.invert(image)

#     # Convert to array
#     image = np.array(image)

#     # Normalize
#     image = image / 255.0

#     # Reshape for CNN
#     image = image.reshape(1, 28, 28, 1)

#     return image

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None

#     if request.method == "POST":
#         file = request.files["image"]
#         image = Image.open(file)
#         processed_image = preprocess_image(image)

#         preds = model.predict(processed_image)
#         prediction = np.argmax(preds)

#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)









# # Load your trained model
# # model = tf.keras.models.load_model("digit_model.h5")
# # model = tf.keras.models.load_model("15epochs.h5")
# # model = tf.keras.models.load_model("8epochs.h5")
# # model = tf.keras.models.load_model("10epochs.h5")


from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from flask_cors import CORS
import gc

# 1. CRITICAL: Restrict TensorFlow threads to stop CPU & RAM spikes on the free tier
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
CORS(app)

# Load your trained model once globally at startup
model = tf.keras.models.load_model("11epochs.h5")

def preprocess_image(image):
    # Handle transparent backgrounds from HTML canvases
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

    # Convert to grayscale and resize
    image = image.convert("L")
    image = image.resize((28, 28))

    # Invert colors (MNIST expects white digit on black background)
    image = ImageOps.invert(image)

    # Convert to array and normalize
    image = np.array(image) / 255.0

    # Reshape for CNN
    return image.reshape(1, 28, 28, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        processed_image = preprocess_image(image)

        # 2. CRITICAL FIX: Use model(..., training=False) instead of model.predict()
        # This prevents the memory spike that triggers the 502 OOM crash!
        preds = model(processed_image, training=False)
        prediction = int(np.argmax(preds))

        # 3. Immediately free up memory after the request is done
        del processed_image, preds
        gc.collect()

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)