import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
import subprocess

subprocess.run(['pip', 'install', 'tensorflow'])
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Pneumonia detection
         """
         )

file = st.file_uploader("Please upload the image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        size = (150,150)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        #img_resize = (cv2.resize(image, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction
if file is None:
    st.text("Please upload an image")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    pneumonia_probability = predictions[0]  # Modify this based on your model's output

    # You can set a threshold to decide if pneumonia is detected or not
    threshold = 0.5  # Modify this threshold as needed

    # Determine if pneumonia is detected based on the threshold
    pneumonia_detected = pneumonia_probability >= threshold

    # Display the result
    if pneumonia_detected:
        st.write("No Pneumonia Detected")
    else:
        st.write("Pneumonia Detected")


