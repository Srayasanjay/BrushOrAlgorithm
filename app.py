import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
MODEL_PATH = "ai_art_classifier.tflite"

# Function to load TFLite model
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Function to make prediction
def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return "AI-Generated Art" if prediction > 0.5 else "Human-Made Art"

# Streamlit UI
st.title("AI Art Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load TFLite model
    interpreter = load_model()

    # Preprocess image & predict
    processed_image = preprocess_image(image)
    result = predict(processed_image, interpreter)

    st.write(f"Prediction: **{result}**")
