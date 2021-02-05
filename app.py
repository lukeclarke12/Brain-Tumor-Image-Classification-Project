#!/usr/local/bin/python3
from PIL import Image, ImageOps
import keras
import numpy as np
import streamlit as st
from img_classification import brain_tumor_classification

st.title("Brain Tumor Image Classification Model")
st.header("Brain Tumor MRI Classification Example")
st.text("Upload a brain MRI Image for image classification as tumor or no tumor")

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = brain_tumor_classification(image,
                                           'brain_tumor_classification.h5')
    if label == 0:
        st.write("The MRI has a brain tumor")

    else:
        st.write("The MRI scan is healthy")


