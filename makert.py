import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Image Classifier Elephant vs Tiger")
import base64

@st.cache_resource()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')
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




