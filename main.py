import streamlit as st
import cv2
import numpy as np
from model import predict_emotion

def local_css(filename):
    with open(filename) as file:
        st.markdown(f"<style>{file.read()}</style>",unsafe_allow_html=True)

local_css("styles/style.css")

st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊")

st.title("😊 Facial Emotion Detection Web App")
st.write("Upload an image, and the model will detect the dominant emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing emotion..."):
        result = predict_emotion(image)

    st.success(f"**Detected Emotion:** {result['emotion']}")
    if "confidence" in result:
        st.write(f"Confidence: {result['confidence'] * 100:.1f}%")
else:
    st.info("Please upload an image to start.")

