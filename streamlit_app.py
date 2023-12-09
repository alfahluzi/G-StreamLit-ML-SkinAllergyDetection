import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

"""
Allergy Detection
"""

# Load trained model
loaded_model = tf.keras.models.load_model('model.h5')
input_shape = (128, 128)
label = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    image = Image.open(img_file_buffer)
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