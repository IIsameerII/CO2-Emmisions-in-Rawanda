import streamlit as st


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
st.markdown("""
<div class="justify-text">

</div>
""", unsafe_allow_html=True)


st.markdown("**1. Load the Dataset**")

code1 = """
#loading the train dataset
train_data = pd.read_csv("../dataset/CO2 Emmisions in Rawanda/train.csv", index_col='ID_LAT_LON_YEAR_WEEK')
train_data.head()

"""

st.code(code1,language='python')