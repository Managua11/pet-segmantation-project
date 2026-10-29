import os
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.predict import load_model, overlay, predict

CHECKPOINT = "artifacts/frozen_encoder/best.pt"


@st.cache_resource
def get_model():
    return load_model(CHECKPOINT)


st.title("Pet Segmentation")

if not os.path.exists(CHECKPOINT):
    st.error(f"checkpoint ვერ მოიძებნა: {CHECKPOINT}")
    st.stop()

model, img_size = get_model()
uploaded = st.file_uploader("ატვირთე სურათი", type=["jpg", "jpeg", "png"])
threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)

if uploaded is not None:
    image = np.array(ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB"))
    start = time.perf_counter()
    mask = predict(model, image, img_size, threshold)
    ms = (time.perf_counter() - start) * 1000

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original")
    col2.image(overlay(image, mask), caption="Overlay")
    st.write(f"Inference: {ms:.0f} ms")
