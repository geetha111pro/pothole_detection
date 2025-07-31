import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

model_path = os.path.abspath("best.pt")
print("Model path:", model_path)
model = YOLO(model_path)



st.title("🕳️ Pothole Detection using YOLOv8")
st.sidebar.title("Input Options")
input_type = st.sidebar.radio("Choose input type:", ["Image", "Video", "Webcam"])

# Image upload and detection
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)

            # Use predict for better accuracy
            results = model.predict(source=temp.name, conf=0.5)
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result", use_column_width=True)

# Video upload and detection
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict frame-by-frame
            results = model.predict(source=frame, conf=0.5)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()

# Webcam detection (only works locally)
elif input_type == "Webcam":
    st.warning("⚠️ Webcam only works in local environment. Not supported on Streamlit Cloud.")
    run_webcam = st.button("Start Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=0.5)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()
