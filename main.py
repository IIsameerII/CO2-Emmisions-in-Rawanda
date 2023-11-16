import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from PIL import Image
import matplotlib
import numpy as np
import config


# project_path = r'C:\Users\SameerAhamed\Documents\GitHub\MACHINE-LEARNING-DSCI6601-PROJECT\\'
st.set_page_config(page_title='main',initial_sidebar_state='expanded')
train = pd.read_csv(r'./dataset/CO2 Emmisions in Rawanda/train.csv',
                    index_col='ID_LAT_LON_YEAR_WEEK')
# train = pd.read_csv(r'./dataset/CO2 Emmisions in Rawanda/train.csv',
#                     index_col='ID_LAT_LON_YEAR_WEEK')

img0 = Image.open(r'Streamlit app QR Code.svg')

# Display the image in the sidebar
st.sidebar.image(img0, caption='Click here to view our app on your phone')


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
st.subheader('Contributors \n * [Sameer Ahamed](https://www.linkedin.com/in/sameer-ahamed-904032165/) \n * [Peter Utomakili](http://www.linkedin.com/in/peter-utomakili-a12a6b34) \n * [Adejoke Adeoye](https://www.linkedin.com/in/adejokeadeoye/)')
st.write('--------------------------')

# Project Overview
st.subheader('Project Overview')
st.markdown("""
<div class="justify-text">
    Our primary focus is on greenhouse gas emissions, particularly carbon dioxide (CO2). These emissions result from a variety of human activities, including the use of fossil fuels for energy, transportation, manufacturing, and deforestation. Rising global temperatures, more frequent and severe extreme weather events, the melting of the polar ice caps, an increase in sea level, and disruptions to ecosystems and biodiversity are some of the negative repercussions that follow. To ensure a healthy and habitable planet for both the present and future generations, addressing the emissions problem is a global necessity that requires cooperative efforts to reduce and ultimately eradicate these harmful emissions.
</div>
""", unsafe_allow_html=True)
st.write('-------------------------')


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
st.write('-------------------------')

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
df_describe = pd.read_csv(config.project_path + r'df_describe.csv',index_col='Unnamed: 0')
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
df_missingness = pd.read_csv(r'df_missingness.csv',index_col='Key')
st.bar_chart(df_missingness)

st.write()
st.info('Distribution of data instances in years')
img1 = Image.open('Distribution of data wrt years.png')
st.image(img1,use_column_width=True)

st.write()
st.info('Distribution of the Emission Target Value')
img2 = Image.open('Target Value Histogram.png')
st.image(img2,use_column_width=True)

st.info('Visualizations of Emmisions in each unique locations')
def rgba_to_hex(color):
    """Return color as #rrggbb for the given color values."""
    red, green, blue, alpha = color
    return f"#{int(red*255):02x}{int(green*255):02x}{int(blue*255):02x}"

temp = train.groupby(['latitude', 'longitude']).emission.mean().reset_index()
geometry = gpd.points_from_xy(temp.longitude, temp.latitude)

cmap = matplotlib.colormaps['coolwarm']
normalizer = matplotlib.colors.Normalize(vmin=np.log1p(temp.emission.min()), vmax=np.log1p(temp.emission.max()))

# Create a canvas to plot your map on
all_data_map = folium.Map(prefer_canvas=True)

# Create a geometry list from the GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geometry]

# Iterate through list and add a marker for each location
for coordinates, emission in zip(geo_df_list, temp.emission):
#     print(emission, normalizer(emission), rgba_to_hex(cmap(normalizer(emission))))
    # Place the markers 
    all_data_map.add_child(
        folium.CircleMarker(
            location=coordinates,
            radius = 1,
            weight = 4,
            zoom =10,
            color = rgba_to_hex(cmap(normalizer(np.log1p(emission))))),
        )
all_data_map.fit_bounds(all_data_map.get_bounds())
folium_static(all_data_map)

st.write('----------------------')

st.markdown('**2. Data Cleaning and Preperation**')

st.markdown('**2.1. Feature selection and Evaluation Approach**')

st.markdown("""
There are 7 features we need to compare with the outcome.

* Sulphur Dioxide

* Carbon Monoxide

* Nitrogen Dioxide

* Formaldehyde

* UV Aerosol Index

* Ozone

* Cloud
""")

st.markdown("""
<div class="justify-text">
We need to deal with missing values in the features columns for exploratory data analysis. 
Therefore, we dropped all the variables that had more than 40% of missing values.
</div>
""", unsafe_allow_html=True)

st.write('')

# Button to show the text
if st.button('Click here to see features left after dropping'):
    # Display the text when the button is clicked
    st.write("""NitrogenDioxide_tropospheric_NO2_column_number_density
             
NitrogenDioxide_stratospheric_NO2_column_number_density
             
NitrogenDioxide_NO2_slant_column_number_density
             
NitrogenDioxide_tropopause_pressure
             
NitrogenDioxide_absorbing_aerosol_index
             
NitrogenDioxide_cloud_fraction
             
NitrogenDioxide_sensor_altitude
             
NitrogenDioxide_sensor_azimuth_angle
             
NitrogenDioxide_sensor_zenith_angle
             
NitrogenDioxide_solar_azimuth_angle
             
NitrogenDioxide_solar_zenith_angle
             
Formaldehyde_tropospheric_HCHO_column_number_density
             
Formaldehyde_tropospheric_HCHO_column_number_density_amf
             
Formaldehyde_HCHO_slant_column_number_density
             
Formaldehyde_cloud_fraction
             
Formaldehyde_solar_zenith_angle
             
Formaldehyde_solar_azimuth_angle
             
Formaldehyde_sensor_zenith_angle
             
Formaldehyde_sensor_azimuth_angle
             
UvAerosolIndex_absorbing_aerosol_index
             
UvAerosolIndex_sensor_altitude
             
UvAerosolIndex_sensor_azimuth_angle
             
UvAerosolIndex_sensor_zenith_angle
             
UvAerosolIndex_solar_azimuth_angle
             
UvAerosolIndex_solar_zenith_angle
             
Ozone_O3_column_number_density
             
Ozone_O3_column_number_density_amf
             
Ozone_O3_slant_column_number_density
             
Ozone_O3_effective_temperature
             
Ozone_cloud_fraction
             
Ozone_sensor_azimuth_angle
             
Ozone_sensor_zenith_angle
             
Ozone_solar_azimuth_angle
             
Ozone_solar_zenith_angle
             
Cloud_cloud_fraction
             
Cloud_cloud_top_pressure
             
Cloud_cloud_top_height
             
Cloud_cloud_base_pressure
             
Cloud_cloud_base_height
             
Cloud_cloud_optical_depth
             
Cloud_surface_albedo
             
Cloud_sensor_azimuth_angle
             
Cloud_sensor_zenith_angle
             
Cloud_solar_azimuth_angle
             
Cloud_solar_zenith_angle
             
emission""")
    

st.write('')
    

st.markdown("""
<div class="justify-text">
In the process of developing our machine learning model, 
            we are meticulously examining a range of 
            environmental features to better understand 
            their relationship with our target variable, 
            which is emission levels. Our dataset encompasses 
            seven primary features: Sulphur Dioxide, Carbon Monoxide, 
            Nitrogen Dioxide, Formaldehyde, UV Aerosol Index, Ozone, 
            and Cloud. Each of these primary features is further broken
             down into various sub-features, providing a more granular 
            view and enabling a thorough analysis.

Our objective is to identify a single, representative feature for each 
            primary feature category that most significantly correlates 
            with emission levels. To achieve this, we are employing 
            heatmaps as a key analytical tool. Heatmaps offer a visual 
            representation of correlation data, making it easier to pinpoint 
            which specific sub-features have the strongest correlations with 
            emissions. By focusing on these highly correlated sub-features, we 
            aim to enhance the predictive accuracy of our machine learning model,
             ensuring that it is both efficient and effective in predicting emission
             levels based on environmental factors. This approach allows us 
            to streamline our feature set, reducing complexity while maintaining,
             or even improving, the model's performance.
</div>
""", unsafe_allow_html=True)

# Get the Images for all the heatmaps
carbon_img = Image.open(r'Carbon Monoxide heatmap.png')
sulphur_img = Image.open(r'Sulphur dioxide heatmap.png')
nitrogen_img = Image.open(r'Nitrogen Dioxide heatmap.png')
formaldehyde_img = Image.open(r'Formaldehyde Heatmap.png')
uv_img = Image.open(r'UV Aerosol Heatmap.png')
ozone_img = Image.open(r'Ozone Heatmap.png')
cloud_img = Image.open(r'Cloud Heatmap.png')


st.image(sulphur_img,caption='Correlation of Sulphur Dioxide Features to emission')
st.info('Selected Feature: **SulphurDioxide_sensor_azimuth_angle**')


st.image(carbon_img,caption='Correlation of Carbon Monoxide Features to emission')
st.info('Selected Feature: **CarbonMonoxide_H2O_column_number_density**')

st.image(nitrogen_img,caption='Correlation of Nitrogen Features to emission')
st.info('Selected Feature: **NitrogenDioxide_sensor_altitude**')

st.image(formaldehyde_img,caption='Correlation of Formaldehyde Features to emission')
st.info('Selected Feature: **Formaldehyde_tropospheric_HCHO_column_number_density_amf**')

st.image(uv_img,caption='Correlation of UV Aerosol Features to emission')
st.info('Selected Feature: **UvAerosolIndex_solar_azimuth_angle**')

st.image(ozone_img,caption='Correlation of Ozone Features to emission')
st.info('Selected Feature: **Ozone_solar_azimuth_angle**')

st.image(cloud_img,caption='Correlation of Cloud Features to emission')
st.info('Selected Feature: **Cloud_solar_azimuth_angle**')


st.markdown("""
<div class="justify-text">
These are the features we are left with -

* latitude

* longitude

* year

* week_no

* SulphurDioxide_sensor_azimuth_angle

* CarbonMonoxide_H2O_column_number_density

* NitrogenDioxide_sensor_altitude
            
* Formaldehyde_tropospheric_HCHO_column_number_density_amf
            
* UvAerosolIndex_solar_azimuth_angle
            
* Ozone_solar_azimuth_angle
            
* Cloud_solar_azimuth_angle
            
* emission
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="justify-text">
There are 2 ways which we tried to deal with the missing values of the features.
            
1. Using KNN imputer
            
2. Checking the distribution to determine the appropriate measure of central tendency
            : mean and median
            


</div>
""", unsafe_allow_html=True)



st.write('---------')

st.markdown('**2.2. Using domain knowledge about emissions and environments**')
img3 = Image.open(r'time series foe every location.png')

st.image(img3,use_column_width=True,caption='Time Series graph of the Emission over 2019 to 2021 in numbered weeks')

folium_static(all_data_map)


st.markdown("""
<div class="justify-text">
In the process of refining our dataset for analysis, 
            we selectively chose to retain only the 
            columns pertaining to latitude, longitude, 
            and week number and year. This decision was significantly 
            influenced by insights gained from a domain expert 
            in environmental science, who is currently pursuing 
            a master's degree. According to their expertise, 
            emissions are intricately linked to time-series 
            variations. This knowledge prompted us to focus 
            on the spatial and temporal dimensions of our 
            data. By incorporating latitude and longitude, 
            we ensure that our analysis accounts for the 
            geographic distribution of emissions, recognizing 
            that environmental impacts can vary greatly across 
            different locations. The inclusion of the week number 
            is equally critical, as it allows us to investigate how 
            emissions fluctuate over time, from week 0 to week 52. 
            This time component is essential to understand seasonal 
            patterns, trends, and anomalies in emission levels. By 
            narrowing down to these specific columns, our analysis 
            is tailored to explore the relationship between emissions 
            and their temporal and spatial dynamics, thereby aligning 
            with the specialized insights provided by the 
            environmental science expert.
""", unsafe_allow_html=True)

st.write('----------------------')

st.markdown('**3. Model Scores**')

# Create a DataFrame
df = pd.DataFrame({
    'Model': [ 'Linear Regression', 'Lasso', 
              'Gradient Boosting Regressor', 'Decision Tree', 
              'Kernel Ridge Regressor', 'Random Forest Regressor'],
    'Test Scores': [0.0124, 0.0121, 0.7975, 
                   0.9676, 0.1298, 0.9760]
})

# Set 'Model' as the index of the DataFrame
df = df.set_index('Model')

# Create a bar chart
st.bar_chart(df)
