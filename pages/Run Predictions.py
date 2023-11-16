import streamlit as st
import pandas as pd
import config
import pickle
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

test_df = pd.read_csv(r'./dataset/CO2 Emmisions in Rawanda/train.csv')

num_times = st.slider("Choose how many predictions are needed",min_value=0,max_value=5)

for num in range(0,num_times):
    # Assuming df is your DataFrame
    random_row = test_df.sample(n=1)

    # Extracting only the 'longitude', 'latitude', and 'week_no' columns
    longitude = random_row['longitude'].iloc[0]
    latitude = random_row['latitude'].iloc[0]
    week_no = random_row['week_no'].iloc[0]
    emission = random_row['emission'].iloc[0]

    # If you want these in a single row DataFrame
    selected_data = random_row[['longitude', 'latitude', 'week_no']]

    st.write(selected_data)

    pred_list = []
    pred_list.append(['ground truth',emission])

    

    with open(config.project_path + r'./models/LinearRegressor1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

        output = loaded_model.predict(selected_data)

        # st.write(output)
        pred_list.append(['Linear Regression',output])

    with open(config.project_path + r'./models/Lasso1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

        output = loaded_model.predict(selected_data)

        # st.write(output)
        pred_list.append(['Lasso',output])

    with open(config.project_path + r'./models/GradientBoostingRegressor1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)


        output = loaded_model.predict(selected_data)

        # st.write(output)
        pred_list.append(['Gradient Boosting Regressor',output])

    with open(config.project_path + r'./models/DesicionTree1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

        output = loaded_model.predict(selected_data)

        # st.write(output)
        pred_list.append(['Desicion Tree',output])

    with open(config.project_path + r'./models/KernelRidge1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

        output = loaded_model.predict(selected_data)

        # st.write(output)
        pred_list.append(['Kernel Ridge Regressor',output])

    # with open(config.project_path + r'models\RandomForestRegressor1.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)

    #     output = loaded_model.predict(selected_data)

    #     # st.write(output)
    #     pred_list.append(['Random Forest Regressor',output])




    # st.write(str(pred_list))

    # Extract model names and their corresponding predictions
    models = [item[0] for item in pred_list]
    predictions = [item[1] if isinstance(item[1], float) else item[1][0] for item in pred_list]

    # Normalize the predictions (unwrap from array if needed)
    normalized_data = [[item[0], item[1] if isinstance(item[1], float) else item[1][0]] for item in pred_list]

    # Convert to DataFrame
    df = pd.DataFrame(normalized_data, columns=['Model', 'Prediction'])

    df = df.set_index('Model')

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.write(df)

        with col2:
            st.bar_chart(df)
    st.write('-----------------')
