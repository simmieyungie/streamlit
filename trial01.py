# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 23:33:29 2021

@author: SIMIYOUNG
"""

#import streamlit
import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
model_training  = st.container()


#Header bit
with header:
    st.title('welcome to my awesome data science project')
    st.text("In this project I look into transactions of taxis in NYC..")

#dataset display bit
with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on .....")
    
    #read in dataset
    df = pd.read_csv("https://raw.githubusercontent.com/MicrosoftDocs/ml-basics/master/data/diabetes.csv")
    
    #write to app
    st.write(df.head())

    #age distribution
    age_dist = df.Age.value_counts().head(30)
    
    #add header to chart
    st.subheader("Distribution of patient ages")
    
    #plot barchart
    st.bar_chart(age_dist)
    
with features:
    st.header("The features I created")
    
    
with model_training:
    st.header("The model I train")
    st.text("Here you get to choose the hyperparameters")
    
    sel_col, disp_col = st.columns(2)
    
    #create slider
    sel_col.slider("What should be the age?", min_value = 15, max_value = 100, value = 20, step = 1)
    
    n_estimators = sel_col.selectbox("How many trees should there be?", options = [100, 200, 300, "No list"])
    
    input_feature = sel_col.text_input("Which feature should be used as the input feature?", "diabetic")
    