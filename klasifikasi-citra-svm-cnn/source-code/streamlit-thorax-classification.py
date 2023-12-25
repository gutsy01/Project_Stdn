import cv2
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score

class_names = ['Normal', 'Tuberculosis', 'Pneumonia']

def load_model():
    model = tf.keras.models.load_model('./collab-ekstraksi-80-10-10.h5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.title("Disease Classification on Lung X-ray Images")

file = st.file_uploader("Please upload an x-ray picture of your lungs", type=["jpg", "png", "jpeg"])

# Membagi layar menjadi dua kolom
col1, col2 = st.columns(2)

# Load the saved SVM model
svm_model_path = "./collab-svm-manual.pkl"
with open(svm_model_path, 'rb') as f:
    svm_weights, svm_biases = pickle.load(f)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clahe_image

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img)
    img_reshape = img_clahe[np.newaxis, ...]
    
    # Extract features using the CNN model
    features = model.predict(img_reshape)
    features = features[0]  # Menjadikan dimensi (256,)
    
    # Predict using the SVM model
    scores = np.dot(features, svm_weights.T) + svm_biases
    prediction = np.argmax(scores)
    confidence = scores[prediction]
    
    # Display preprocessed image
    with col2:
        st.write("After Preprocessing :")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.imshow(img_clahe)
        plt.axis('on')
        st.pyplot()

    return prediction, confidence, scores

if file is None:
    st.write("Please upload an image file")
else:
    with col1:
        st.write("Uploaded Image :")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        image = Image.open(file)
        plt.imshow(image)
        plt.axis('on')
        st.pyplot()

    prediction, confidence, scores = import_and_predict(image, model)
    st.write("Classification Label :", prediction)
    st.write("Confidence :", confidence)
    
    confidence_data = {'Class Name': class_names, 'Confidence': scores}
    confidence_df = pd.DataFrame(confidence_data)
    
    st.markdown("<h5 style='text-align: center;'>Class-wise Confidence:</h5>", unsafe_allow_html=True)
    st.table(confidence_df)
    
    st.success('The image has been analyzed as {}'.format(class_names[prediction]))
    st.set_option('deprecation.showPyplotGlobalUse', False)
