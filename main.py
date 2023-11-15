import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt


project_path = r'C:\Users\SameerAhamed\Documents\GitHub\MACHINE-LEARNING-DSCI6601-PROJECT\\'

train = pd.read_csv(project_path + r'dataset\CO2 Emmisions in Rawanda\train.csv',
                    index_col='ID_LAT_LON_YEAR_WEEK')


# Define custom CSS for justification
st.markdown("""
<style>
.justify-text {
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)

# Setting the title of the page
st.title('Prediction of CO2 Emissions in Rwanda')

# Setting the contributor name
st.subheader('Contributors \n * Sameer Ahamed \n * Peter Utomakili \n * Adejoke Adeoye')
st.write('')

# Project Overview
st.subheader('Project Overview')
st.markdown("""
<div class="justify-text">
    Our primary focus is on greenhouse gas emissions, particularly carbon dioxide (CO2). These emissions result from a variety of human activities, including the use of fossil fuels for energy, transportation, manufacturing, and deforestation. Rising global temperatures, more frequent and severe extreme weather events, the melting of the polar ice caps, an increase in sea level, and disruptions to ecosystems and biodiversity are some of the negative repercussions that follow. To ensure a healthy and habitable planet for both the present and future generations, addressing the emissions problem is a global necessity that requires cooperative efforts to reduce and ultimately eradicate these harmful emissions.
</div>
""", unsafe_allow_html=True)
st.write('')


# Project Novelty 
st.subheader('Novelty of the project')
st.markdown("""
<div class="justify-text">
    Our primary focus is on greenhouse gas emissions, 
    particularly carbon dioxide (CO2). These emissions
    result from a variety of human activities, including
    the use of fossil fuels for energy, transportation
    , manufacturing, and deforestation. Rising global 
    temperatures, more frequent and severe extreme weather
    events, the melting of the polar ice caps, an increase
    in sea level, and disruptions to ecosystems and 
    biodiversity are some of the negative repercussions 
    that follow. To ensure a healthy and habitable planet
    for both the present and future generations, addressing 
    the emissions problem is a global necessity that requires
    cooperative efforts to reduce and ultimately eradicate 
    these harmful emissions.
</div>
""", unsafe_allow_html=True)
st.write('')

# Methodology
st.subheader('Methodology')

st.markdown('**1. Preliminary Analysis of the variables in Dataset**')
st.markdown("""
<div class="justify-text">
The factors we’ve identified encompass a wide range of variables, both atmospheric and pollutant-related, which can significantly influence CO2 emissions. Relevance of each factor is discussed briefly below:

1.	Sulfur Dioxide (SO2): SO2 is a precursor to sulfuric acid formation and can contribute to air pollution. Understanding its levels can shed light on industrial emissions and their environmental impact.
2.	Carbon Monoxide (CO): CO is a product of incomplete combustion and is a key indicator of combustion-related emissions. Monitoring CO levels helps assess the efficiency of energy and transportation systems.
3.	Nitrogen Dioxide (NO2): NO2 is a major component of smog and contributes to respiratory issues. It's an important pollutant to track due to its links with emissions from vehicles and industrial sources.
4.	Formaldehyde: Formaldehyde is a volatile organic compound (VOC) and is a crucial air quality parameter. It is linked to various health concerns and is used to gauge indoor and outdoor pollution sources.
5.	Aerosol Content: Aerosols affect climate and air quality. Measuring aerosol content is valuable for understanding their role in emissions and atmospheric processes.
6.	Ozone: Ozone levels are relevant as they are both a greenhouse gas and an air pollutant. Its presence in the atmosphere is a key factor in air quality and climate change.
7.	Atmospheric Features: Monitoring atmospheric features such as cloud cover can provide insights into how weather patterns impact emissions. Clouds can affect the amount of solar radiation reaching the Earth's surface, which, in turn, influences CO2 concentrations.

The dataset in question provides a comprehensive overview of geospatial data, as it 
            includes the latitude and longitude coordinates for each observation. 
            This precise locational information allows for detailed geographical 
            analysis and mapping. Additionally, the dataset is temporally structured, 
            featuring a chronological component that tracks the week number of 
            each observation. These weekly records span from week 0 to week 52, 
            offering a year-long perspective on the data. This combination of spatial 
            and temporal elements makes the dataset particularly valuable for studies 
            that require an understanding of both where and when the observations occurred.
            
Moreover, we found that all of these obervation have been taken **once a week over a 3
            year period from 2019 to 2021.**

The shape of the data is (79023, 75). That means we have 75 variables (columns) and 79023 data instances (rows) 

There are 497 unique coordinates from where observations were took
</div>
""", unsafe_allow_html=True)

st.info('The output of the **df.describe()** is shown below')
df_describe = pd.read_csv(project_path + r'df_describe.csv',index_col='Unnamed: 0')
st.write(df_describe)

st.markdown("""
<div class="justify-text">
    For every geographical point, there are 159 rows with observations in train and 49 rows in test.
The 159 training rows correspond to three years (2019, 2020, 2021) with 
            53 weeks each (numbered from 0 to 52).
The 49 test rows correspond to weeks 0 to 48 of 2022.
497 * 3 * 53 = 79023, the size of the training dataset.
</div>
""", unsafe_allow_html=True)

st.write('')

st.info('The map below visually shows the areas of observation')

# Taken from https://www.kaggle.com/code/inversion/getting-started-eda

train_coords = train.drop_duplicates(subset = ['latitude', 'longitude'])
geometry = gpd.points_from_xy(train_coords.longitude, train_coords.latitude)
geo_df = gpd.GeoDataFrame(
    train_coords[["latitude", "longitude"]], geometry=geometry
)
# Create a canvas to plot your map on
all_data_map = folium.Map(prefer_canvas=True)

# Create a geometry list from the GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

# Iterate through list and add a marker for each location
for coordinates in geo_df_list:

    # Place the markers 
    all_data_map.add_child(
        folium.CircleMarker(
            location=coordinates,
            radius = 1,
            weight = 4,
            zoom =10,
            color =  "red"),
        )
all_data_map.fit_bounds(all_data_map.get_bounds())
folium_static(all_data_map)

st.markdown("""
<div class="justify-text">
Since our project involves geospatial data, leveraging the measurements 
            from surrounding areas could enhance our ability to forecast 
            the desired outcome. An effective machine learning model should 
            go beyond utilizing the 74 attributes of each individual 
            row for prediction, and should incorporate data from 
            adjacent locations for more accurate results. Additionally, 
            the dataset is structured as a time series, suggesting that 
            historical data points might be useful in predicting present values.
</div>
""", unsafe_allow_html=True)

st.write('')
st.info('Visiualization of missingness in the dataset')
df_missingness = pd.read_csv(project_path + r'df_missingness.csv',index_col='Key')
st.bar_chart(df_missingness)

st.write()
st.info('Distribution of data instances in years')
img = Image.open(project_path+r'Distribution of data wrt years.png')
st.image(img,use_column_width=True)

st.write()
st.info('Distribution of the Emission Target Value')
img = Image.open(project_path+r'Target Value Histogram.png')
st.image(img,use_column_width=True)


st.markdown('**2. Data Cleaning and Preperation**')
