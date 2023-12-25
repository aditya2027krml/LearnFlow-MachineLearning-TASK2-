import streamlit as st
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the pre-trained model
model = load_model('./training/TSR.h5')

st.text("AUTHOR : <ADITYA KUMAR>")
st.text("TASK 2")
st.text("Traffic Sign Recognition")


# Classes of traffic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }

def preprocess_image(img_path):
    data = []
    try:
        if img_path.endswith('.npy'):
            # Handle loading images from a NumPy file
            data = np.load(img_path)
        else:
            # Handle loading a single image
            image = Image.open(img_path)
            image = image.resize((30, 30))
            data.append(np.array(image))
    except Exception as e:
        print(f"Error processing image: {e}")
    return np.array(data)

def predict_traffic_sign(img_path):
    try:
        X_test = preprocess_image(img_path)
        Y_pred_probabilities = model.predict(X_test)
        Y_pred_classes = np.argmax(Y_pred_probabilities, axis=1)
        predicted_class_index = Y_pred_classes[0]
        predicted_class_name = classes.get(predicted_class_index, 'Unknown Class')
        return predicted_class_name
    except Exception as e:
        print(f"Error predicting traffic sign: {e}")
        return 'Prediction Error'

# Streamlit web app
st.title("Traffic Sign Recognition Web App")
st.sidebar.title("Enter Image Path")

img_path = st.sidebar.text_input("Enter the image path...")

if img_path:
    # Display the specified image
    try:
        image = Image.open(img_path)
        st.image(image, caption="Specified Image.", use_column_width=True)

        # Make predictions on the specified image path
        prediction = predict_traffic_sign(img_path)

        # Display the predicted class
        st.subheader("Prediction:")
        st.subheader(prediction)
    except Exception as e:
        st.write(f"Error: {e}. Please make sure the file path is correct.")
else:
    st.sidebar.write("Please enter the image path.")

st.success("SUCCESSFULLY COMPLETED!")
