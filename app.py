import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('feelings.keras')

def predict_image(image):
    image = image.resize((48,48))
    img_array = img_to_array(image) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

st.title("Image Classification with CNN")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predictions = predict_image(image)
    print("Raw Predictions:", predictions) 
    class_idx = np.argmax(predictions)
    class_names = ['surprise', 'disgust', 'fear', 'happy', 'natural', 'sad', 'angry']
    
    st.markdown(
        f"<p style='font-size:50px;'>Feeling Prediction: {class_names[class_idx]}</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size:50px;'>Confidence: {predictions[0][class_idx]:.2f}</p>",
        unsafe_allow_html=True
    )

