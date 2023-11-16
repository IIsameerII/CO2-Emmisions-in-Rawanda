import streamlit as st
import pandas as pd
import config
import pickle
import random
import numpy as np

st.set_page_config(page_title='Predict using models')
st.header('Predict using our Trained Models')

# Define custom CSS for justification
st.markdown("""
<style>
.justify-text {
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)

test_df = pd.read_csv(config.project_path+r'dataset\CO2 Emmisions in Rawanda\train.csv')


data_prep_df = test_df.iloc[:,1:5] # Got the required columns
# st.write(data_prep_df)


# random_idx = random.randint(0,len(data_prep_df))

# test_df = data_prep_df[random_idx,:]
random_row = np.array(data_prep_df.sample(n=1))

st.write(random_row) 

with open(config.project_path + r'models\model_time_series.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

    loaded_model.predict(random_row)