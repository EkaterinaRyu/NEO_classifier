import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import streamlit as st
import logging

logging.basicConfig(level=logging.DEBUG)

# --- SETTING CONFIGURATION ---
st.set_page_config(page_title="NEA classifier")
st.title('NASA Nearest Earth Objects (a.k.a asteroids) classifier')

try:
    model = joblib.load('res\GradientBoostingClassifier.joblib')
    print("model successfully loaded")
except Exception as e:
    print(e)


def preprocessing(data):
    try:
        names = data['name']
    except:
        names = data.index
    data = data[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']]
    scaler = StandardScaler()
    print(data.columns)
    return names, scaler.fit_transform(data)

upload_message = "note: data must contain columns labeled 'absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance'. \nIt can also contain column 'name', where you can add names or ids for asteroids."
uploaded_file = st.file_uploader("Upload CSV", type='csv')
st.write(upload_message)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Here's your data:")
    st.write(data)

    if st.button('Predict'):
        names, data = preprocessing(data)
        st.write('Preprocessed data:')
        st.write(pd.DataFrame(data, columns=['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']))
        predictions = model.predict(data)
        st.write('here are your predictions')
        st.write(pd.concat([names, pd.DataFrame(predictions, columns=['pred'])], axis=1))

