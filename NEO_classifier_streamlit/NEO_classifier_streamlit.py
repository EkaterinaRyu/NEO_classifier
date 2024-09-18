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
st.title('NASA Nearest Earth Objects \n(a.k.a asteroids) classifier')

page_bg_img = '''
<style>
.stAppViewContainer {
    background-image: url("https://cdn.mos.cms.futurecdn.net/vFBAa88j4EKfDXrsKsUFuh-1200-80.png");
    background-size: cover;
}
.stAppViewBlockContainer {
    -webkit-box-align: center;
    margin-top: 100px;
    margin-bottom: 50px;
    padding-top: 10px;
    background-color: rgb(240, 242, 246);
    border-radius: 0.5rem;
    color: rgb(49, 51, 63);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    model = joblib.load('NEO_classifier_streamlit\/res\GradientBoostingClassifier.joblib')
    print("model successfully loaded")
except Exception as e:
    print(e)


def preprocessing(data):
    try:
        names = data['name']
    except:
        names = data.index
    try:
        trues = data['is_hazardous']
    except:
        trues = None

    data = data[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']]
    scaler = StandardScaler()
    print(data.columns)
    return names, scaler.fit_transform(data), trues

upload_message = "note: data must contain columns labeled 'absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance'. \nIt can also contain column 'name', where you can add names or ids for asteroids."
uploaded_file = st.file_uploader("Here you can upload your CSV", type='csv')
st.write(upload_message)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Here's your data:")
    st.write(data)

    if st.button('Predict'):
        names, data, trues = preprocessing(data)
        st.write('Preprocessed data:')
        st.write(pd.DataFrame(data, columns=['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']))
        predictions = model.predict(data)
        st.write('here are your predictions')

        if trues is not None:
            st.write(pd.concat([names, pd.DataFrame(predictions, columns=['pred']), trues], axis=1))
        else:
            st.write(pd.concat([names, pd.DataFrame(predictions, columns=['pred'])], axis=1))

