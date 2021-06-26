import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.title('Covid-19 India Cases')
st.write("It shows ***Coronavirus Cases*** in India")
st.sidebar.title("Selector")
image = Image.open("Coronavirus.jpg")
st.image(image,use_column_width=True)
st.markdown('<style>body{background-color: lightblue;}</style>',unsafe_allow_html=True)

@st.cache
def load_data():
    df = pd.read_csv("india_covid_19_state_wise_info.csv")
    return df

df = load_data()

visualization = st.sidebar.selectbox('Select a Chart type',('Bar Chart','Pie Chart','Line Chart'))
state_select = st.sidebar.selectbox('Select a state',df['state'].unique())
status_select = st.sidebar.radio('Covid-19 patient status',('confirmed_cases','active_cases','recovered_cases','death_cases'))
#select = st.sidebar.selectbox('Covid-19 patient status',('confirmed_cases','active_cases','recovered_cases','death_cases'))
selected_state = df[df['state']==state_select]
st.markdown("## **State level analysis**")

def get_total_dataframe(df):
    total_dataframe = pd.DataFrame({
    'Status':['Confirmed', 'Recovered', 'Deaths','Active'],
    'Number of cases':(df.iloc[0]['confirmed_cases'],
    df.iloc[0]['active_cases'], 
    df.iloc[0]['recovered_cases'],df.iloc[0]['death_cases'])})
    return total_dataframe
state_total = get_total_dataframe(selected_state)
if visualization=='Bar Chart':
    state_total_graph = px.bar(state_total, x='Status',y='Number of cases',
                               labels={'Number of cases':'Number of cases in %s' % (state_select)},color='Status')
    st.plotly_chart(state_total_graph)
elif visualization=='Pie Chart':
    if status_select=='confirmed_cases':
        st.title("Total Confirmed Cases ")
        fig = px.pie(df, values=df['confirmed_cases'], names=df['state'])
        st.plotly_chart(fig)
    elif status_select=='active_cases':
        st.title("Total Active Cases ")
        fig = px.pie(df, values=df['active_cases'], names=df['state'])
        st.plotly_chart(fig)
    elif status_select=='death_cases':
        st.title("Total Death Cases ")
        fig = px.pie(df, values=df['death_cases'], names=df['state'])
        st.plotly_chart(fig)
    else:
        st.title("Total Recovered Cases ")
        fig = px.pie(df, values=df['recovered_cases'], names=df['state'])
        st.plotly_chart(fig)
elif visualization =='Line Chart':
    if status_select == 'death_cases':
        st.title("Total Death Cases Among states")
        fig = px.line(df,x='state',y=df['death_cases'])
        st.plotly_chart(fig)
    elif status_select =='confirmed_cases':
        st.title("Total Confirmed Cases Among states")
        fig = px.line(df,x='state',y=df['confirmed_cases'])
        st.plotly_chart(fig)
    elif status_select =='recovered_cases':
        st.title("Total Recovered Cases Among states")
        fig = px.line(df,x='state',y=df['recovered_cases'])
        st.plotly_chart(fig)
    else:
        st.title("Total Active Cases Among states")
        fig = px.line(df,x='state',y=df['active_cases'])
        st.plotly_chart(fig)
        
def get_table():
    datatable = df[['state', 'confirmed_cases', 'recovered_cases', 'death_cases','active_cases']].sort_values(by=['confirmed_cases'],ascending =False)
    return datatable

datatable = get_table()
st.dataframe(datatable)

