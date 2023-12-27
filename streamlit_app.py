import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

"""
Allergy Detection.


This is demonstration of my CNN Model for detecting which the skin have allergy or not.
The model is not train for detecting what kind of allergy is.

The model need a picture with full (atleast 80%) of skin image.
"""

# Load trained model
loaded_model = tf.keras.models.load_model('model_allergy.h5')
input_shape = (56,56)
label = ["Non Allergy", "Allergy"]

img_file_buffer = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Choose a file")

if img_file_buffer is not None or uploaded_file is not None:
    if uploaded_file is None:
        input_file = img_file_buffer
    else: input_file = uploaded_file
    # To read image file buffer as a PIL Image:
    img = Image.open(input_file)
    img_array = np.array(img)

    image = Image.open(input_file)
    # Resize img to fit in model input shape
    image = image.resize(input_shape)
    # Convert the image to a NumPy array
    image_array = np.array(image)
    image_array = image_array[:, :, :3]  # Keep only the first 3 channels (RGB)
    # Normalize the pixel values
    image_array = image_array / 255.0
    # Expand dimensions to match the input shape dimension of the model
    input_image = np.expand_dims(image_array, axis=0)

    predictions = loaded_model.predict(input_image).tolist()[0]
    predicted_label = np.argmax(predictions, axis=-1)
    
    st.write("Result:")
    st.write(f"{100*predictions[predicted_label]:.2f}%",label[predicted_label])

    st.write(pd.DataFrame({
        'Label': label,
        'Result': predictions,
    }))