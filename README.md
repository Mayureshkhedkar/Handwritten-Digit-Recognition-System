# Handwritten Digit Recognition using CNN

## Project Description
This project is a **Handwritten Digit Recognition System** developed using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The system allows users to upload an image of a handwritten digit through a web interface and predicts the digit in real time using a trained deep learning model.

The project combines:
- Deep Learning (CNN)
- Image Processing
- Flask Web Development
- Frontend Design using HTML & CSS

The final deployed model was trained for **10 epochs**, achieving:
- **99.08% accuracy on MNIST test data**
- **~80–82% accuracy on real-world handwritten digit samples**

---

# Technology Stack and Tools Used

## Programming Language
- Python 3.10

## Deep Learning Framework
- TensorFlow
- Keras

## Backend Framework
- Flask
- Flask-CORS

## Frontend Technologies
- HTML5
- CSS3

## Image Processing
- Pillow (PIL)

## Numerical Computing
- NumPy

## Dataset
- MNIST Handwritten Digit Dataset

## Development Tools
- Visual Studio Code (VS Code)
- Jupyter Notebook

---

# Features and Functionalities Implemented

- Upload handwritten digit image through web interface
- CNN-based digit prediction
- Real-time inference using Flask backend
- Image preprocessing pipeline:
  - Grayscale conversion
  - Image resizing (28×28)
  - Color inversion
  - Pixel normalization
  - Reshaping for CNN input
- Responsive frontend interface
- Prediction result display
- Multiple epoch testing and evaluation
- Model performance comparison

---

# CNN Model Architecture

The CNN model contains:
1. Conv2D Layer (32 filters, ReLU)
2. MaxPooling Layer
3. Dropout Layer
4. Conv2D Layer (64 filters, ReLU)
5. MaxPooling Layer
6. Dropout Layer
7. Flatten Layer
8. Dense Layer (128 neurons, ReLU)
9. Dropout Layer
10. Output Dense Layer (10 neurons, Softmax)

---

# Model Performance

## Epoch Testing Results
- 8 Epochs → Underfitting
- 10 Epochs → Best performance (Selected Model)
- 11 Epochs → Slight instability
- 12 Epochs → Reduced real-world accuracy
- 15 Epochs → Overfitting

## Final Selected Model
- Test Accuracy: **99.08%**
- Test Loss: **0.028**
- Real-world Accuracy: **~80–82%**

---

# Project Structure

```bash
Handwritten-Digit-Recognition/
│
├── app.py
├── 10epochs.h5
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── README.md
└── requirements.txt
