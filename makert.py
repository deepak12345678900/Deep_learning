import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Image Classifier Elephant vs Tiger")

uploaded_file = st.file_uploader("Select an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    predict_button = st.button("Predict Image")
    if predict_button:
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0

        model = tf.keras.models.load_model('cats_model.h5')  # Load the model
        predictions = model.predict(np.expand_dims(img, axis=0))
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        class_name = "Elephant" if confidence < 0.50 else "Tiger"
        
        st.title("Prediction:")
        st.header(f"Predicted class: {class_name}")
        st.snow()




