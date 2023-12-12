import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as vgg19_preprocess_input,
)
import numpy as np
import os
import io
import h5py

model = load_model("vgg19.hdf5")  # Replace with the actual path to your HDF5 file


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Extract class and confidence
    class_label = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]

    return class_label, confidence


# Streamlit UI
st.title("Fruit and Vegetable Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader(
    "Choose a fruit or vegetable image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Perform classification when the 'Classify' button is clicked
    if st.button("Classify"):
        try:
            # Get predictions
            class_label, confidence = predict_image(uploaded_file)

            # Display results
            st.success(f"Prediction: {class_label}")
            st.info(f"Confidence: {confidence:.2%}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
