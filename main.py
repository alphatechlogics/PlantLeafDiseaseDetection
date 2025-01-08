import streamlit as st
from cv2 import cv2
import numpy as np
import keras

# Page Configuration
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")

# Title and Description
st.title("🌿 Leaf Disease Detection")
st.markdown("""
This **Leaf Disease Detection** app uses deep learning to identify diseases from leaf images. 
The model recognizes 33 different types of leaf conditions with high accuracy. 

**Instructions**:
- Upload a clear image of a leaf.
- Wait for the model to predict the disease.
""")

# Load Model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')
label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 
    'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 
    'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot', 
    'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot', 
    'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 
    'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# File Uploader
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload a leaf image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Display Uploaded Image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        
        # Preprocess Image
        normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

        # Prediction
        with st.spinner("Analyzing... Please wait!"):
            predictions = model.predict(normalized_image)
        
        # Display Results
        confidence = predictions[0][np.argmax(predictions)] * 100
        if confidence >= 80:
            st.success(f"**Prediction**: {label_name[np.argmax(predictions)]}")
            st.write(f"**Confidence**: {confidence:.2f}%")
        else:
            st.warning("The model is uncertain. Please try another image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload an image to get started!")
